#!/usr/bin/env python
"""TRACE-Kin training entry point.

Drives the v1 / v3 architecture switch and the cross-dataset pooling that
applies to both. (The earlier v2 attempt regressed on 9 of 11 reruns and was
removed; see PROJECT.md history.) Tier 1 and Tier 2 hyperparameter knobs
that the failed ablation explored (``--patience``, ``--warmup_iters``,
``--min_lrate``, ``--lr_decay_iters``, ``--dropout``,
``--use_gated_prot_fusion``, ``--deep_regression_head``,
``--learnable_aux_loss``) are intentionally removed — they did not move the
needle and are now dead weight. The only training-time switches that matter
are ``--model_version``, ``--use_swa``, and ``--pool_train_csvs``.

The triple DataFrame load bug from the legacy script (load for embedding
detection, load for seq2feat, load for create_data_loaders) is fixed: each
input file is read at most once.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Local package imports
from training.data_utils import (
    DataLoader,
    CustomWeightedRandomSampler,
    compute_pna_degrees,
    virtual_screening,
)
from training.dataset import ProteinMoleculeDataset
from training.ligand_init import ligand_init
from training.metrics import evaluate_reg
from training.protein_init_with_embedding import protein_init_with_embedding
from training.trainer import Trainer

# Both architecture variants are exposed by the package; we instantiate one
# based on the config's ``model_version`` field.
from models.trace_kin import TraceKinV1, TraceKinV3


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TRACE-Kin (v1 baseline or v3 dual-head) on a kinetic-regression dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    parser.add_argument("--datafolder", type=str, required=True,
                        help="Primary dataset folder (must contain train.{parquet,csv} and test.{parquet,csv}).")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Where to write checkpoints, predictions, logs.")
    parser.add_argument("--config_path", type=str, default="training/config_v3.json",
                        help="Path to model config JSON.")
    parser.add_argument("--protein_col", type=str, default="Sequence")
    parser.add_argument("--feature_col", type=str, default="protein_features")
    parser.add_argument("--ligand_col", type=str, default="Smiles")
    parser.add_argument("--label_col", type=str, default="Label")
    parser.add_argument("--metabolite_feature_col", type=str, default="metabolite_features",
                        help="Per-row column with the pre-computed ChemBERT (768) molecular embedding "
                             "consumed by the v3 FP-MLP head. Required for v3.")

    # Cross-dataset pooling for v2 catalytic kinetics. Each entry is the path
    # to an additional dataset folder whose train.{parquet,csv} is appended to
    # the primary training set. Test/val always come from --datafolder only.
    parser.add_argument("--pool_train_csvs", type=str, default="",
                        help="Comma-separated list of additional dataset folders to pool into the training set.")

    # Compute
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--evaluate_epoch", type=int, default=1)
    parser.add_argument("--lrate", type=float, default=1e-4,
                        help="Optimizer learning rate. Defaults match v1 baseline (1e-4).")

    # Task
    parser.add_argument("--regression_task", action="store_true", default=False)
    parser.add_argument("--classification_task", action="store_true", default=False)
    parser.add_argument("--mclassification_task", type=int, default=0)

    # Architecture / training behaviour switches
    parser.add_argument("--model_version", type=str, choices=["v1", "v3"], default=None,
                        help="Override config 'model_version'. Defaults to whatever the config file specifies.")
    parser.add_argument("--use_swa", action="store_true", default=False,
                        help="Enable SWA (overrides config swa.use_swa).")
    parser.add_argument("--no_swa", action="store_true", default=False,
                        help="Disable SWA even if the config enables it.")

    # Preprocessing cache
    parser.add_argument("--force_rebuild", action="store_true", default=False,
                        help="Always regenerate protein.pt and ligand.pt instead of reusing the cache.")

    # Optional pretrained warm start
    parser.add_argument("--trained_model_path", type=str, default="")
    parser.add_argument("--finetune_modules", type=str, default=None)

    # Quick smoke testing
    parser.add_argument("--sample_train", type=int, default=None)
    parser.add_argument("--sample_test", type=int, default=None)
    parser.add_argument("--sample_valid", type=int, default=None)

    parser.add_argument("--save_interpret", action="store_true", default=True)
    parser.add_argument("--sampling_col", type=str, default="")

    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def find_data_file(folder: str, base_name: str) -> str | None:
    for ext in (".parquet", ".csv", ".tsv"):
        path = os.path.join(folder, f"{base_name}{ext}")
        if os.path.exists(path):
            return path
    return None


def load_dataframe(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def coerce_features_to_numpy(raw):
    """Best-effort conversion of a protein_features cell to a 1-D or 2-D numpy array."""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return np.asarray(raw, dtype=np.float32)
    if isinstance(raw, np.ndarray):
        return raw.astype(np.float32)
    if isinstance(raw, str):
        try:
            return np.asarray(ast.literal_eval(raw), dtype=np.float32)
        except (ValueError, SyntaxError):
            return np.asarray(json.loads(raw), dtype=np.float32)
    raise TypeError(f"Unsupported protein_features type: {type(raw)}")


def detect_embedding_dim(df: pd.DataFrame, feature_col: str) -> int:
    """Find the first non-null protein_features entry and return its trailing dim."""
    for _, row in df.iterrows():
        raw = row[feature_col]
        if raw is None:
            continue
        if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) == 0:
            continue
        if not isinstance(raw, (list, tuple, np.ndarray)) and pd.isna(raw):
            continue
        arr = coerce_features_to_numpy(raw)
        return int(arr.shape[-1] if arr.ndim >= 1 else arr.size)
    raise ValueError(f"No valid {feature_col!r} entry found in dataframe.")


def build_seq2feat(*dfs: pd.DataFrame, protein_col: str, feature_col: str) -> dict:
    """Map sequence -> numpy embedding from one or more DataFrames."""
    seq2feat: dict = {}
    for df in dfs:
        if df is None:
            continue
        for seq, raw in zip(df[protein_col], df[feature_col]):
            if raw is None:
                continue
            if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) == 0:
                continue
            if not isinstance(raw, (list, tuple, np.ndarray)) and pd.isna(raw):
                continue
            seq2feat[seq] = coerce_features_to_numpy(raw)
    return seq2feat


def detect_chembert_dim(df: pd.DataFrame, feature_col: str) -> int:
    """Same shape as detect_embedding_dim, but for the ChemBERT column.

    Returns the trailing dim of the first non-null entry. ChemBERT
    embeddings are sequence-level (1D); we still take .shape[-1] so a 2D
    encoding wouldn't silently break.
    """
    for _, row in df.iterrows():
        raw = row[feature_col]
        if raw is None:
            continue
        if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) == 0:
            continue
        if not isinstance(raw, (list, tuple, np.ndarray)) and pd.isna(raw):
            continue
        arr = coerce_features_to_numpy(raw)
        return int(arr.shape[-1] if arr.ndim >= 1 else arr.size)
    raise ValueError(f"No valid {feature_col!r} entry found in dataframe.")


def build_smi2chembert(*dfs: pd.DataFrame, ligand_col: str, feature_col: str) -> dict:
    """Map SMILES -> ChemBERT (768,) numpy embedding from one or more DataFrames.

    Fails loudly if the same SMILES appears with materially different
    embedding vectors across rows — silent last-write-wins would mask data
    pipeline bugs.
    """
    smi2chembert: dict = {}
    for df in dfs:
        if df is None:
            continue
        for smi, raw in zip(df[ligand_col], df[feature_col]):
            if raw is None:
                continue
            if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) == 0:
                continue
            if not isinstance(raw, (list, tuple, np.ndarray)) and pd.isna(raw):
                continue
            arr = coerce_features_to_numpy(raw).reshape(-1)
            existing = smi2chembert.get(smi)
            if existing is not None and not np.allclose(existing, arr, atol=1e-5):
                raise ValueError(
                    f"ChemBERT collision for SMILES {smi!r}: two different "
                    f"vectors observed across rows (max abs diff "
                    f"{float(np.max(np.abs(existing - arr))):.4g})."
                )
            smi2chembert[smi] = arr
    return smi2chembert


def normalize_columns(df: pd.DataFrame, protein_col: str, ligand_col: str, label_col: str) -> pd.DataFrame:
    """Rename input columns to the canonical names that ProteinMoleculeDataset expects."""
    rename_map = {}
    if protein_col != "Protein":
        rename_map[protein_col] = "Protein"
    if ligand_col != "Ligand":
        rename_map[ligand_col] = "Ligand"
    if label_col not in ("regression_label", "classification_label", "multiclass_label"):
        rename_map[label_col] = "regression_label"
    return df.rename(columns=rename_map) if rename_map else df


def load_split_dataframes(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None,
                                          list[str], list[str], dict, dict, int, int]:
    """Load primary train/test/val once, build seq2feat + smi2chembert, return everything the rest of main() needs."""
    train_path = find_data_file(args.datafolder, "train")
    test_path = find_data_file(args.datafolder, "test")
    val_path = find_data_file(args.datafolder, "val")

    if train_path is None or test_path is None:
        raise FileNotFoundError(
            f"Missing train/test in {args.datafolder} (looked for .parquet/.csv/.tsv)."
        )

    print(f"Loading primary train: {train_path}")
    train_df = load_dataframe(train_path)
    if args.sample_train:
        train_df = train_df.sample(args.sample_train, random_state=args.seed)
    train_df = train_df.reset_index(drop=True)

    print(f"Loading primary test: {test_path}")
    test_df = load_dataframe(test_path)
    if args.sample_test:
        test_df = test_df.sample(args.sample_test, random_state=args.seed)
    test_df = test_df.reset_index(drop=True)

    val_df = None
    if val_path is not None:
        print(f"Loading primary val: {val_path}")
        val_df = load_dataframe(val_path)
        if args.sample_valid:
            val_df = val_df.sample(args.sample_valid, random_state=args.seed)
        val_df = val_df.reset_index(drop=True)

    # Cross-dataset pooling: append additional training rows from other dataset folders.
    pool_paths = [p.strip() for p in args.pool_train_csvs.split(",") if p.strip()]
    pooled_dfs: list[pd.DataFrame] = []
    for folder in pool_paths:
        extra_train_path = find_data_file(folder, "train")
        if extra_train_path is None:
            print(f"WARN: pool entry {folder} has no train file, skipping.")
            continue
        print(f"Pooling additional train from: {extra_train_path}")
        pooled_dfs.append(load_dataframe(extra_train_path))
    if pooled_dfs:
        before = len(train_df)
        train_df = pd.concat([train_df] + pooled_dfs, ignore_index=True)
        print(f"Pooled training set: {before} -> {len(train_df)} rows from {len(pool_paths)} extra dataset(s).")

    # Detect embedding dim from the (possibly pooled) train df. All inputs must
    # share a single embedding dim; mixing 1280-dim ESM with 1024-dim ProteinCLIP
    # is a user error, not a fall-through.
    embedding_dim = detect_embedding_dim(train_df, args.feature_col)
    print(f"Detected prot_evo_channels = {embedding_dim}")

    seq2feat = build_seq2feat(train_df, test_df, val_df,
                              protein_col=args.protein_col, feature_col=args.feature_col)
    print(f"seq2feat: {len(seq2feat)} unique sequences.")

    # ChemBERT (768) molecular embeddings for the v3 FP-MLP head. The
    # parquet files written by the user pipeline carry these per-row in
    # metabolite_features. Detect dim from train, then build the SMILES
    # -> vector dict across all splits.
    chembert_dim = detect_chembert_dim(train_df, args.metabolite_feature_col)
    print(f"Detected chembert_dim = {chembert_dim}")
    smi2chembert = build_smi2chembert(train_df, test_df, val_df,
                                      ligand_col=args.ligand_col,
                                      feature_col=args.metabolite_feature_col)
    print(f"smi2chembert: {len(smi2chembert)} unique SMILES.")

    # Standardize column names for downstream code.
    train_df = normalize_columns(train_df, args.protein_col, args.ligand_col, args.label_col)
    test_df = normalize_columns(test_df, args.protein_col, args.ligand_col, args.label_col)
    if val_df is not None:
        val_df = normalize_columns(val_df, args.protein_col, args.ligand_col, args.label_col)

    # Collect the unique pairs that need preprocessing. Every sequence used in
    # any split must have an entry in seq2feat (otherwise embedding lookup
    # fails downstream).
    proteins = sorted({*train_df["Protein"], *test_df["Protein"]})
    ligands = sorted({*train_df["Ligand"], *test_df["Ligand"]})
    if val_df is not None:
        proteins = sorted(set(proteins).union(val_df["Protein"]))
        ligands = sorted(set(ligands).union(val_df["Ligand"]))

    missing = [s for s in proteins if s not in seq2feat]
    if missing:
        raise ValueError(
            f"{len(missing)} sequences are missing protein_features (e.g. {missing[:3]!r})."
        )

    missing_smi = [s for s in ligands if s not in smi2chembert]
    if missing_smi:
        raise ValueError(
            f"{len(missing_smi)} SMILES are missing metabolite_features "
            f"(e.g. {missing_smi[:3]!r})."
        )

    return (train_df, test_df, val_df, proteins, ligands,
            seq2feat, smi2chembert, embedding_dim, chembert_dim)


# ---------------------------------------------------------------------------
# Preprocessing cache (protein.pt / ligand.pt)
# ---------------------------------------------------------------------------
def preprocess(folder: str, proteins: list[str], ligands: list[str],
               seq2feat: dict, smi2chembert: dict,
               force_rebuild: bool) -> tuple[dict, dict]:
    protein_path = os.path.join(folder, "protein.pt")
    ligand_path = os.path.join(folder, "ligand.pt")

    if os.path.exists(protein_path) and not force_rebuild:
        print(f"Reusing protein cache: {protein_path}")
        protein_dict = torch.load(protein_path)
    else:
        print("Building protein graphs from scratch (this can be slow)...")
        protein_dict = protein_init_with_embedding(proteins, seq2feat=seq2feat)
        torch.save(protein_dict, protein_path)

    if os.path.exists(ligand_path) and not force_rebuild:
        print(f"Reusing ligand cache: {ligand_path}")
        ligand_dict = torch.load(ligand_path)
        # The cache may predate the ChemBERT cutover. v3 needs chembert_fp on
        # every ligand; backfill in-memory from smi2chembert (no disk write —
        # next --force_rebuild will replace the on-disk cache cleanly).
        missing = [s for s in ligand_dict if 'chembert_fp' not in ligand_dict[s]]
        if missing:
            print(f"Backfilling chembert_fp on {len(missing)} cached ligands...")
            for s in missing:
                if s not in smi2chembert:
                    raise KeyError(
                        f"chembert_fp missing in cache and in smi2chembert "
                        f"for SMILES {s!r}; rerun with --force_rebuild."
                    )
                vec = np.asarray(smi2chembert[s], dtype=np.float32).reshape(-1)
                ligand_dict[s]['chembert_fp'] = torch.from_numpy(vec).unsqueeze(0)
    else:
        print("Building ligand graphs from scratch...")
        ligand_dict = ligand_init(ligands, smi2chembert=smi2chembert)
        torch.save(ligand_dict, ligand_path)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return protein_dict, ligand_dict


# ---------------------------------------------------------------------------
# DataLoader construction
# ---------------------------------------------------------------------------
def make_loaders(train_df, test_df, val_df, ligand_dict, protein_dict,
                 device, batch_size, sampling_col):
    """Filter rows whose graphs failed to build, then build PyG DataLoaders."""
    def filter_and_reset(df, name):
        valid = df["Ligand"].isin(ligand_dict) & df["Protein"].isin(protein_dict)
        dropped = (~valid).sum()
        if dropped:
            print(f"WARN: dropping {dropped} {name} rows with missing protein/ligand graphs.")
        return df[valid].reset_index(drop=True)

    train_df = filter_and_reset(train_df, "train")
    test_df = filter_and_reset(test_df, "test")
    val_df = filter_and_reset(val_df, "val") if val_df is not None else None

    train_ds = ProteinMoleculeDataset(train_df, ligand_dict, protein_dict, device=device)
    test_ds = ProteinMoleculeDataset(test_df, ligand_dict, protein_dict, device=device)
    val_ds = (
        ProteinMoleculeDataset(val_df, ligand_dict, protein_dict, device=device)
        if val_df is not None else None
    )

    train_sampler = None
    train_shuffle = True
    if sampling_col:
        weights = torch.from_numpy(train_df[sampling_col].values)
        train_sampler = CustomWeightedRandomSampler(weights, len(weights), replacement=True)
        train_shuffle = False

    follow = ["mol_x", "clique_x", "prot_node_aa"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=train_shuffle,
                              sampler=train_sampler, follow_batch=follow)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, follow_batch=follow)
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, follow_batch=follow)
        if val_ds is not None else None
    )
    return train_loader, test_loader, val_loader, train_df, test_df, val_df


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_model(model_config: dict, mol_deg, prot_deg, device: str):
    """Instantiate v1 or v3 from the config."""
    version = model_config.get("model_version", "v1")
    params = model_config["params"]
    tasks = model_config["tasks"]

    common_kwargs = dict(
        mol_in_channels=params["mol_in_channels"],
        prot_in_channels=params["prot_in_channels"],
        prot_evo_channels=params["prot_evo_channels"],
        hidden_channels=params["hidden_channels"],
        pre_layers=params["pre_layers"],
        post_layers=params["post_layers"],
        aggregators=params["aggregators"],
        scalers=params["scalers"],
        total_layer=params["total_layer"],
        K=params["K"],
        heads=params["heads"],
        dropout=params.get("dropout", 0),
        dropout_attn_score=params.get("dropout_attn_score", 0.2),
        regression_head=tasks["regression_task"],
        classification_head=tasks["classification_task"],
        multiclassification_head=tasks["mclassification_task"],
        device=device,
    )

    if version == "v1":
        model = TraceKinV1(
            mol_deg, prot_deg,
            **common_kwargs,
            use_gated_prot_fusion=params.get("use_gated_prot_fusion", False),
            deep_regression_head=params.get("deep_regression_head", False),
            learnable_aux_loss=params.get("learnable_aux_loss", False),
        )
    elif version == "v3":
        model = TraceKinV3(
            mol_deg, prot_deg,
            **common_kwargs,
            chembert_dim=params.get("chembert_dim", 768),
            rf_head_hidden=tuple(params.get("rf_head_hidden", [512, 128])),
            gate_hidden=params.get("gate_hidden", 64),
            gate_init_bias=params.get("gate_init_bias", 0.0),
        )
    else:
        raise ValueError(f"Unknown model_version: {version!r}")

    model = model.to(device)
    model.reset_parameters()
    return model, version


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = torch.device(args.device)

    print("=" * 60)
    print("TRACE-Kin training")
    print("=" * 60)
    print(f"Datafolder:        {args.datafolder}")
    print(f"Result path:       {args.result_path}")
    print(f"Config:            {args.config_path}")
    print(f"Pool train CSVs:   {args.pool_train_csvs or '(none)'}")
    print("=" * 60)

    # 1. Load splits + build seq2feat / smi2chembert (single-pass loading).
    (train_df, test_df, val_df, proteins, ligands,
     seq2feat, smi2chembert, embedding_dim, chembert_dim) = load_split_dataframes(args)

    # 2. Load and update model config.
    with open(args.config_path) as f:
        model_config = json.load(f)
    model_config["params"]["prot_evo_channels"] = embedding_dim
    model_config["params"]["chembert_dim"] = chembert_dim
    model_config["optimizer"]["lrate"] = args.lrate
    model_config["tasks"]["regression_task"] = args.regression_task
    model_config["tasks"]["classification_task"] = args.classification_task
    model_config["tasks"]["mclassification_task"] = args.mclassification_task
    if args.model_version is not None:
        model_config["model_version"] = args.model_version

    swa_block = model_config.setdefault("swa", {"use_swa": False, "swa_start_frac": 0.75, "swa_lr_factor": 0.1})
    if args.use_swa:
        swa_block["use_swa"] = True
    if args.no_swa:
        swa_block["use_swa"] = False

    print(f"model_version: {model_config['model_version']}, use_swa: {swa_block['use_swa']}")

    # 3. Preprocess proteins/ligands (with cache).
    protein_dict, ligand_dict = preprocess(
        args.datafolder, proteins, ligands, seq2feat, smi2chembert, args.force_rebuild
    )

    # 4. Build DataLoaders (with row filtering).
    train_loader, test_loader, val_loader, train_df, test_df, val_df = make_loaders(
        train_df, test_df, val_df, ligand_dict, protein_dict,
        device, args.batch_size, args.sampling_col
    )
    print(f"loaders ready: train={len(train_loader)} test={len(test_loader)} "
          f"val={len(val_loader) if val_loader is not None else 0}")

    # 5. PNA degree statistics (computed once on train, cached per dataset).
    degree_path = os.path.join(args.datafolder, "degree.pt")
    if os.path.exists(degree_path) and not args.force_rebuild:
        degree_dict = torch.load(degree_path)
    else:
        print("Computing PNA degrees from training data...")
        mol_deg, clique_deg, prot_deg = compute_pna_degrees(train_loader)
        degree_dict = {"ligand_deg": mol_deg, "clique_deg": clique_deg, "protein_deg": prot_deg}
        torch.save(degree_dict, degree_path)
    mol_deg = degree_dict["ligand_deg"]
    prot_deg = degree_dict["protein_deg"]

    # 6. Build model + dump config beside the checkpoint for reproducibility.
    os.makedirs(args.result_path, exist_ok=True)
    model_dir = os.path.join(args.result_path, f"save_model_seed{args.seed}")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(degree_dict, os.path.join(model_dir, "degree.pt"))

    model, model_version = build_model(model_config, mol_deg, prot_deg, args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Built TraceKin{model_version.upper()} with {n_params:,} parameters")

    if args.trained_model_path:
        param_path = os.path.join(args.trained_model_path, "model.pt")
        state = torch.load(param_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"Loaded pretrained weights from {param_path}")

    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    # 7. Determine evaluation metric.
    if model_config["tasks"]["regression_task"]:
        evaluate_metric = "rmse"
    elif model_config["tasks"]["classification_task"]:
        evaluate_metric = "roc"
    elif model_config["tasks"]["mclassification_task"]:
        evaluate_metric = "macro_f1"
    else:
        raise ValueError("No task selected (--regression_task / --classification_task / --mclassification_task).")

    # 8. Trainer.
    finetune_modules = ast.literal_eval(args.finetune_modules) if args.finetune_modules else None
    engine = Trainer(
        model=model,
        lrate=model_config["optimizer"]["lrate"],
        min_lrate=model_config["optimizer"]["min_lrate"],
        wdecay=model_config["optimizer"]["weight_decay"],
        betas=tuple(model_config["optimizer"]["betas"]),
        eps=model_config["optimizer"]["eps"],
        amsgrad=model_config["optimizer"]["amsgrad"],
        clip=model_config["optimizer"]["clip"],
        steps_per_epoch=len(train_loader),
        num_epochs=args.epochs,
        total_iters=None,
        warmup_iters=model_config["optimizer"]["warmup_iters"],
        lr_decay_iters=model_config["optimizer"]["lr_decay_iters"],
        schedule_lr=model_config["optimizer"]["schedule_lr"],
        regression_weight=1,
        classification_weight=1,
        evaluate_metric=evaluate_metric,
        result_path=args.result_path,
        runid=args.seed,
        finetune_modules=finetune_modules,
        device=device,
        patience=model_config["optimizer"].get("patience", 0),
        use_swa=swa_block["use_swa"],
        swa_start_frac=swa_block.get("swa_start_frac", 0.75),
        swa_lr_factor=swa_block.get("swa_lr_factor", 0.1),
    )

    # 9. Train.
    engine.train_epoch(
        train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        evaluate_epoch=args.evaluate_epoch,
    )

    # 10. Final evaluation on test set with the best checkpoint.
    best_checkpoint = os.path.join(model_dir, "model.pt")
    model.load_state_dict(torch.load(best_checkpoint, map_location=device))
    interpret_path = os.path.join(args.result_path, f"interpretation_result_seed{args.seed}")
    os.makedirs(interpret_path, exist_ok=True)
    screen_df = virtual_screening(
        test_df, model, test_loader,
        result_path=interpret_path, save_interpret=args.save_interpret,
        ligand_dict=ligand_dict, device=device,
    )
    pred_csv = os.path.join(args.result_path, f"test_prediction_seed{args.seed}.csv")
    screen_df.to_csv(pred_csv, index=False)

    if args.regression_task and "regression_label" in screen_df and "predicted_binding_affinity" in screen_df:
        y = screen_df["regression_label"].values.astype(float)
        p = screen_df["predicted_binding_affinity"].values.astype(float)
        mask = ~np.isnan(y) & ~np.isnan(p)
        if mask.any():
            metrics = evaluate_reg(y[mask], p[mask])
            print(f"Final test metrics: rmse={metrics['rmse']:.4f}, pearson={metrics['pearson']:.4f}, "
                  f"r2_proxy(mse)={metrics['mse']:.4f}")

    print(f"Done. Predictions: {pred_csv}; checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    main()

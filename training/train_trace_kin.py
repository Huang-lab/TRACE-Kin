#!/usr/bin/env python
"""TRACE-Kin training entry point.

Drives the v1 / v4 architecture switch and the cross-dataset pooling that
applies to both. The v2 (shortcut concat) and v3 (FP-MLP / ChemBERT
context-residual) attempts were removed after empirical results showed
they didn't beat v1 on enough cells to justify keeping them; see
PROJECT.md history. The only training-time switches that matter are
``--model_version``, ``--use_swa``, ``--pool_train_csvs``, and
``--molformer_path``.
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
from training.aa_typical import load_or_compute_aa_typical, aa_typical_to_buffers, seq_to_aa_idx
from training.dataset import ProteinMoleculeDataset
from training.ligand_init import ligand_init
from training.metrics import evaluate_reg
from training.molformer_init import compute_molformer_embeddings
from training.protein_init_with_embedding import protein_init_with_embedding
from training.trainer import Trainer

# All architecture variants are exposed by the package; we instantiate one
# based on the config's ``model_version`` field.
from models.trace_kin import TraceKinV1, TraceKinV4, TraceKinV5, TraceKinV5T, TraceKinV6C, TraceKinV7


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TRACE-Kin (v1 baseline or v4 MAP-GNN) on a kinetic-regression dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    parser.add_argument("--datafolder", type=str, required=True,
                        help="Primary dataset folder (must contain train.{parquet,csv} and test.{parquet,csv}).")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Where to write checkpoints, predictions, logs.")
    parser.add_argument("--config_path", type=str, default="training/config_v4.json",
                        help="Path to model config JSON.")
    parser.add_argument("--protein_col", type=str, default="Sequence")
    parser.add_argument("--feature_col", type=str, default="protein_features")
    parser.add_argument("--ligand_col", type=str, default="Smiles")
    parser.add_argument("--label_col", type=str, default="Label")
    parser.add_argument("--metabolite_feature_col", type=str, default="metabolite_features",
                        help="Per-row parquet column carrying a pre-computed molecular "
                             "embedding (768-d). Used by v4's chembert_proj when "
                             "--molformer_path is NOT set. Ignored when --molformer_path "
                             "is set (live MoLFormer-XL inference replaces this column).")
    parser.add_argument("--molformer_path", type=str, default="",
                        help="Local path to a MoLFormer-XL HuggingFace snapshot (e.g., "
                             "/sc/arion/.../transformersmodels/ibm/MoLFormer-XL-both-10pct). "
                             "When set, molecular embeddings are computed via live "
                             "MoLFormer inference and cached to <datafolder>/molformer_emb.pt, "
                             "replacing the parquet metabolite_features column.")
    
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
    parser.add_argument("--model_version", type=str, choices=["v1", "v4", "v5", "v5t", "v6c", "v7"], default=None,
                        help="Override config 'model_version'. Defaults to whatever the config file specifies.")
    parser.add_argument("--use_swa", action="store_true", default=False,
                        help="Enable SWA (overrides config swa.use_swa).")
    parser.add_argument("--no_swa", action="store_true", default=False,
                        help="Disable SWA even if the config enables it.")
    parser.add_argument("--amp", action="store_true", default=False,
                        help="Enable AMP (bf16 mixed precision) to reduce GPU memory.")

    # Preprocessing cache
    parser.add_argument("--force_rebuild", action="store_true", default=False,
                        help="Always regenerate protein.pt and ligand.pt instead of reusing the cache.")

    parser.add_argument("--ensemble_size", type=int, default=1,
                        help="Train N models in one invocation with seeds "
                             "--seed, --seed+1, ... N-1. Preprocessing, DataLoaders and "
                             "PNA degree stats are built once and shared, so the cost is "
                             "N trainings, not N full runs. Members land in "
                             "save_model_seed<seed>/ and are averaged by "
                             "inference/predict_ensemble.py.")

    parser.add_argument("--preprocess_only", action="store_true", default=False,
                        help="Build protein.pt / ligand.pt / degree.pt and exit before training. "
                             "Use this once before launching a multi-seed ensemble array so the "
                             "seeds share a warm cache instead of racing to build it.")

    # Optional pretrained warm start
    parser.add_argument("--trained_model_path", type=str, default="")
    parser.add_argument("--finetune_modules", type=str, default=None)

    # Quick smoke testing
    parser.add_argument("--sample_train", type=int, default=None)
    parser.add_argument("--sample_test", type=int, default=None)
    parser.add_argument("--sample_valid", type=int, default=None)

    parser.add_argument("--save_interpret", action="store_true", default=True)

    parser.add_argument("--no_save_interpret", dest="save_interpret", action="store_false",
                        help="Skip the per-pair interpretation directories (one PAIR_*/ per test "
                             "pair, each with protein.csv / ligand.csv / fingerprint.npy). Those "
                             "are thousands of small files and exhaust disk/inode quota; the test "
                             "prediction CSV is still written. Needed because --save_interpret is "
                             "store_true with default=True and so cannot otherwise be disabled.")

    parser.add_argument("--sampling_col", type=str, default="")
    
    parser.add_argument("--contact_source", type=str, choices=["esm2", "esmc"], 
                        default="esm2"
                       )

    # Node-feature source
    parser.add_argument("--embedding_source", type=str, choices=["parquet", "esmc"],
                        default="parquet",
                        help="Where per-residue node features come from. 'parquet' reads "
                             "--feature_col (default, existing behavior). 'esmc' extracts "
                             "per-residue embeddings from the same ESMC forward pass that "
                             "computes contacts — requires --esmc_contact_ckpt.")


    parser.add_argument("--esmc_contact_ckpt", type=str, default=None )
    
    parser.add_argument("--esmc_model_name", type=str, default="biohub/ESMC-6B")
        
                         

                        
                        
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def print_mem(label: str) -> None:
    """Print process resident memory at a checkpoint. ru_maxrss is KB on
    Linux (the only OS we run on at HPC), bytes on macOS — we report the
    Linux interpretation since the LSF runs there."""
    try:
        import resource
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"[mem] {label:<40s} max RSS = {rss_kb / (1024 * 1024):.2f} GB")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def find_data_file(folder: str, base_name: str) -> str | None:
    for ext in (".parquet", ".csv", ".tsv"):
        path = os.path.join(folder, f"{base_name}{ext}")
        if os.path.exists(path):
            return path
    return None


def load_dataframe(path: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Load parquet or CSV. ``columns=`` pushes the column projection down
    to the reader so heavy columns (e.g. per-residue MutaPLM at 4096-d)
    are never materialized when not needed."""
    ext = Path(path).suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t", usecols=columns)
    return pd.read_csv(path, usecols=columns)


def detect_dim_from_parquet(path: str, col: str) -> int:
    """Read just one row from `path` to learn the trailing dim of `col`.

    Uses pyarrow iter_batches with batch_size=1 so we don't accidentally
    materialize a full row group. For a list[float] / list[list[float]]
    column at MutaPLM's 4096-d, a single full-row read is ~1.6 MB; a
    full-row-group read could be GBs.
    """
    import pyarrow.parquet as pq
    if not path.endswith(".parquet"):
        # CSV/TSV fallback: read a small head with the column subset
        df_head = load_dataframe(path, columns=[col]).head(1)
        return detect_embedding_dim(df_head, col)
    pf = pq.ParquetFile(path)
    batch = next(pf.iter_batches(batch_size=1, columns=[col]))
    rows = batch.to_pylist()
    if not rows:
        raise ValueError(f"Parquet {path} is empty; cannot detect dim of {col!r}.")
    raw = rows[0][col]
    arr = coerce_features_to_numpy(raw)
    return int(arr.shape[-1] if arr.ndim >= 1 else arr.size)


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
    """Map sequence -> numpy float16 embedding from one or more DataFrames.

    Stored as float16 because protein_init_with_embedding converts to
    torch then ``.half()`` immediately. Halving here (instead of after a
    detour through float32) saves 21 GB at MutaPLM's 4096-d × 6607
    unique sequences scale. Skips duplicates so the dict only grows up
    to the unique-sequence count.
    """
    seq2feat: dict = {}
    for df in dfs:
        if df is None:
            continue
        for seq, raw in zip(df[protein_col], df[feature_col]):
            if seq in seq2feat:
                continue  # dedupe — saves time and memory on multi-row repeats
            if raw is None:
                continue
            if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) == 0:
                continue
            if not isinstance(raw, (list, tuple, np.ndarray)) and pd.isna(raw):
                continue
            seq2feat[seq] = coerce_features_to_numpy(raw).astype(np.float16)
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

    When the same SMILES appears with different ChemBERT vectors across
    rows (data integrity issue in the upstream metabolite_features
    column), keep the first-seen vector and emit a warning summarizing
    the worst collisions. This is preferred over hard-failing because:
    (a) the underlying parquet may have stale duplicates (e.g. small
    SMILES like 'O' embedded inconsistently), and (b) a single bad
    SMILES shouldn't kill an otherwise-valid training run. Investigate
    the warnings if results look off.
    """
    smi2chembert: dict = {}
    collisions: dict = {}  # smi -> max abs diff
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
            if existing is not None:
                if not np.allclose(existing, arr, atol=1e-5):
                    diff = float(np.max(np.abs(existing - arr)))
                    collisions[smi] = max(collisions.get(smi, 0.0), diff)
                continue  # keep first-seen vector
            smi2chembert[smi] = arr
    if collisions:
        worst = sorted(collisions.items(), key=lambda kv: -kv[1])[:5]
        print(f"WARN: {len(collisions)} SMILES had inconsistent ChemBERT "
              f"vectors across rows; keeping first-seen vector for each. "
              f"Worst 5 (smi, max abs diff): "
              + ", ".join(f"({s!r}, {d:.3g})" for s, d in worst))
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
                                          list[str], list[str],
                                          int, int, dict]:
    """Load primary train/test/val with ESSENTIAL columns only.

    The big columns (``protein_features`` at 4096-d MutaPLM, plus
    ``metabolite_features`` at 768-d) are NOT loaded here — they are
    constructed lazily inside :func:`preprocess` only when the
    corresponding cache (protein.pt / ligand.pt) misses. This avoids
    holding tens-of-GB of feature arrays in CPU RAM through the whole
    training run when the caches already exist.

    Returns the lite DataFrames + protein/ligand lists + detected dims +
    a ``paths`` dict so :func:`preprocess` can re-read the parquets with
    the heavy columns when (and only when) it needs them.
    """
    train_path = find_data_file(args.datafolder, "train")
    test_path = find_data_file(args.datafolder, "test")
    val_path = find_data_file(args.datafolder, "val")

    if train_path is None or test_path is None:
        raise FileNotFoundError(
            f"Missing train/test in {args.datafolder} (looked for .parquet/.csv/.tsv)."
        )

    # Cross-dataset pool paths (per-folder runs leave this empty).
    pool_paths = [p.strip() for p in args.pool_train_csvs.split(",") if p.strip()]
    pool_train_paths = []
    for folder in pool_paths:
        extra_train_path = find_data_file(folder, "train")
        if extra_train_path is None:
            print(f"WARN: pool entry {folder} has no train file, skipping.")
            continue
        pool_train_paths.append(extra_train_path)

    # --- Lite load: essential columns only -----------------------------------
    essential = [args.protein_col, args.ligand_col, args.label_col]

    print(f"Loading primary train (essential cols): {train_path}")
    train_df = load_dataframe(train_path, columns=essential)
    if args.sample_train:
        train_df = train_df.sample(args.sample_train, random_state=args.seed)
    train_df = train_df.reset_index(drop=True)

    print(f"Loading primary test (essential cols): {test_path}")
    test_df = load_dataframe(test_path, columns=essential)
    if args.sample_test:
        test_df = test_df.sample(args.sample_test, random_state=args.seed)
    test_df = test_df.reset_index(drop=True)

    val_df = None
    if val_path is not None:
        print(f"Loading primary val (essential cols): {val_path}")
        val_df = load_dataframe(val_path, columns=essential)
        if args.sample_valid:
            val_df = val_df.sample(args.sample_valid, random_state=args.seed)
        val_df = val_df.reset_index(drop=True)

    pooled_dfs: list[pd.DataFrame] = []
    for p in pool_train_paths:
        print(f"Pooling additional train (essential cols): {p}")
        pooled_dfs.append(load_dataframe(p, columns=essential))
    if pooled_dfs:
        before = len(train_df)
        train_df = pd.concat([train_df] + pooled_dfs, ignore_index=True)
        print(f"Pooled training set: {before} -> {len(train_df)} rows from {len(pool_train_paths)} extra dataset(s).")

    # --- Dim detection from 1-row samples ------------------------------------
    # --- Dim detection from 1-row samples ------------------------------------
    if args.embedding_source == "esmc":                          # ← NEW branch
        # Node features come from the ESMC forward pass, not --feature_col.
        # The real dim is read from protein.pt after preprocess() (Edit 3).
        embedding_dim = -1
        print("embedding_source=esmc: skipping parquet dim detection "
              "(prot_evo_channels will be set from protein.pt)")
    else:
        embedding_dim = detect_dim_from_parquet(train_path, args.feature_col)
        print(f"Detected prot_evo_channels = {embedding_dim} (1-row sample)")

    print(f"Detected prot_evo_channels = {embedding_dim} (1-row sample)")
    if args.molformer_path:
        chembert_dim = 768  # MoLFormer-XL hidden size; doesn't need parquet read
    else:
        chembert_dim = detect_dim_from_parquet(train_path, args.metabolite_feature_col)
        print(f"Detected chembert_dim = {chembert_dim} (1-row sample, {args.metabolite_feature_col})")

    # --- Standardize columns + collect unique pairs --------------------------
    train_df = normalize_columns(train_df, args.protein_col, args.ligand_col, args.label_col)
    test_df = normalize_columns(test_df, args.protein_col, args.ligand_col, args.label_col)
    if val_df is not None:
        val_df = normalize_columns(val_df, args.protein_col, args.ligand_col, args.label_col)

    proteins = sorted({*train_df["Protein"], *test_df["Protein"]})
    ligands = sorted({*train_df["Ligand"], *test_df["Ligand"]})
    if val_df is not None:
        proteins = sorted(set(proteins).union(val_df["Protein"]))
        ligands = sorted(set(ligands).union(val_df["Ligand"]))

    paths = {
        "train": train_path,
        "test": test_path,
        "val": val_path,
        "pool_train": pool_train_paths,
    }
    return train_df, test_df, val_df, proteins, ligands, embedding_dim, chembert_dim, paths


def _build_seq2feat_lazy(args, paths) -> dict:
    """Re-read parquets with just (protein_col, feature_col) and build seq2feat.

    Called only by :func:`preprocess` on protein.pt cache miss. Held in
    memory just long enough to construct protein_dict, then dropped.
    """
    cols = [args.protein_col, args.feature_col]
    print(f"  reloading {args.feature_col} for seq2feat construction (transient)...")
    train_df = load_dataframe(paths["train"], columns=cols)
    if args.sample_train:
        train_df = train_df.sample(args.sample_train, random_state=args.seed)
    test_df = load_dataframe(paths["test"], columns=cols)
    if args.sample_test:
        test_df = test_df.sample(args.sample_test, random_state=args.seed)
    val_df = None
    if paths["val"] is not None:
        val_df = load_dataframe(paths["val"], columns=cols)
        if args.sample_valid:
            val_df = val_df.sample(args.sample_valid, random_state=args.seed)
    pooled = []
    for p in paths.get("pool_train", []):
        pooled.append(load_dataframe(p, columns=cols))
    if pooled:
        train_df = pd.concat([train_df] + pooled, ignore_index=True)
    return build_seq2feat(train_df, test_df, val_df,
                          protein_col=args.protein_col, feature_col=args.feature_col)


def _build_smi2chembert_lazy(args, paths, ligands) -> dict:
    """Compute MoLFormer or read metabolite_features parquet column for smi2chembert.

    Called only by :func:`preprocess` when ligand.pt cache misses or
    needs chembert_fp backfill.
    """
    if args.molformer_path:
        molformer_cache = os.path.join(args.datafolder, "molformer_emb.pt")
        return compute_molformer_embeddings(
            ligands,
            model_path=args.molformer_path,
            cache_path=molformer_cache,
            force_rebuild=args.force_rebuild,
            device=args.device,
        )
    cols = [args.ligand_col, args.metabolite_feature_col]
    print(f"  reloading {args.metabolite_feature_col} for smi2chembert construction (transient)...")
    train_df = load_dataframe(paths["train"], columns=cols)
    if args.sample_train:
        train_df = train_df.sample(args.sample_train, random_state=args.seed)
    test_df = load_dataframe(paths["test"], columns=cols)
    if args.sample_test:
        test_df = test_df.sample(args.sample_test, random_state=args.seed)
    val_df = None
    if paths["val"] is not None:
        val_df = load_dataframe(paths["val"], columns=cols)
        if args.sample_valid:
            val_df = val_df.sample(args.sample_valid, random_state=args.seed)
    return build_smi2chembert(train_df, test_df, val_df,
                              ligand_col=args.ligand_col,
                              feature_col=args.metabolite_feature_col)


# ---------------------------------------------------------------------------
# Preprocessing cache (protein.pt / ligand.pt)
# ---------------------------------------------------------------------------
def preprocess(args, paths: dict, proteins: list[str], ligands: list[str],
               model_version: str = "v4") -> tuple[dict, dict, dict | None]:
    """Build/load protein.pt, ligand.pt, and (v4) aa_typical caches.

    seq2feat (~21 GB at MutaPLM 4096-d × 6607 unique seqs) is built
    LAZILY — only when ``protein.pt`` cache misses or ``force_rebuild``
    is set. After ``protein_dict`` is constructed we drop seq2feat
    immediately so it doesn't sit in RAM through training. Same pattern
    for smi2chembert (loaded on ligand.pt cache miss only).

    With cache hits (typical for seeds 2/3 after seed 1 builds), peak
    extra CPU RAM = just protein.pt size (~21 GB at MutaPLM). With cache
    miss, peak ≈ 2× protein.pt size briefly during the protein_init
    build (~42 GB), then drops to ~21 GB after seq2feat is freed.

     When ``args.embedding_source == "esmc"``, seq2feat is skipped entirely —
    per-residue node features come from the ESMC forward pass that also
    produces the contact map, so the parquet feature column is never read.
    """
    folder = args.datafolder
    force_rebuild = args.force_rebuild
    protein_path = os.path.join(folder, "protein.pt")
    ligand_path = os.path.join(folder, "ligand.pt")

    # === Protein cache ===
    if os.path.exists(protein_path) and not force_rebuild:
        print(f"Reusing protein cache: {protein_path}")
        protein_dict = torch.load(protein_path)
    else:
        if args.embedding_source == 'esmc':
            seq2feat = None
            print("Cache miss  -- building protein.pt (embedding_source = esmc :"
                  "per residue features + contacts from one  ESMC pass; seq2feat skipped)")
        else:
            
            print("Cache miss — loading seq2feat (transient) to build protein.pt...")
            seq2feat = _build_seq2feat_lazy(args, paths)
            print(f"  seq2feat: {len(seq2feat)} unique sequences (transient)")
            # Validate before consuming.
            missing = [s for s in proteins if s not in seq2feat]
            if missing:
                raise ValueError(
                    f"{len(missing)} sequences are missing protein_features "
                    f"(e.g. {missing[:3]!r}). Check the parquet's {args.feature_col} column."
                )
        protein_dict = protein_init_with_embedding(
            proteins, seq2feat = seq2feat,
            contact_source = args.contact_source,
            embedding_source = args.embedding_source,
            esmc_ckpt = (args.esmc_contact_ckpt or None),
            esmc_model_name = args.esmc_model_name,
            device = args.device,
        )
        
        torch.save(protein_dict, protein_path)
        del seq2feat
        import gc; gc.collect()
        print(f"Saved protein cache: {protein_path}")

    # === aa_typical (v4 only) ===
    aa_typical_buffers = None
    if model_version == "v4":
        aa_typical_path = os.path.join(folder, "aa_typical.pt")
        aa_typical = load_or_compute_aa_typical(
            cache_path=aa_typical_path,
            protein_dict=protein_dict,
            force_rebuild=force_rebuild,
        )
        aa_to_idx, aa_means, aa_stds = aa_typical_to_buffers(aa_typical)
        print(f"  aa_typical buffers: means/stds shape {tuple(aa_means.shape)}, "
              f"{len(aa_to_idx)} known + 1 unknown AA slot")
        aa_typical_buffers = {
            "aa_to_idx": aa_to_idx,
            "aa_means": aa_means,
            "aa_stds": aa_stds,
        }
        # Attach per-residue AA-index (L,) per protein — int64, ~L bytes per
        # protein. PyG batches along dim=0 alongside prot_node_evo.
        for seq, prot in protein_dict.items():
            prot["prot_aa_idx"] = seq_to_aa_idx(prot["seq"], aa_to_idx)

    # === Ligand cache ===
    if os.path.exists(ligand_path) and not force_rebuild:
        print(f"Reusing ligand cache: {ligand_path}")
        ligand_dict = torch.load(ligand_path)
        # Backfill chembert_fp if a legacy cache predates the ChemBERT cutover.
        missing_chembert = [s for s in ligand_dict if 'chembert_fp' not in ligand_dict[s]]
        if missing_chembert:
            print(f"Backfilling chembert_fp on {len(missing_chembert)} cached ligands "
                  f"(loading smi2chembert transiently)...")
            smi2chembert = _build_smi2chembert_lazy(args, paths, ligands)
            for s in missing_chembert:
                if s not in smi2chembert:
                    raise KeyError(
                        f"chembert_fp missing in cache and in smi2chembert "
                        f"for SMILES {s!r}; rerun with --force_rebuild."
                    )
                vec = np.asarray(smi2chembert[s], dtype=np.float32).reshape(-1)
                ligand_dict[s]['chembert_fp'] = torch.from_numpy(vec).unsqueeze(0)
            del smi2chembert
            # Persist the backfilled cache so subsequent runs don't repeat
            # the smi2chembert load + backfill loop.
            torch.save(ligand_dict, ligand_path)
            print(f"  persisted backfilled ligand cache: {ligand_path}")
    else:
        print("Cache miss — loading smi2chembert (transient) to build ligand.pt...")
        smi2chembert = _build_smi2chembert_lazy(args, paths, ligands)
        # Validate before consuming.
        missing_smi = [s for s in ligands if s not in smi2chembert]
        if missing_smi:
            raise ValueError(
                f"{len(missing_smi)} SMILES are missing molecular embeddings "
                f"(e.g. {missing_smi[:3]!r})."
            )
        ligand_dict = ligand_init(ligands, smi2chembert=smi2chembert)
        torch.save(ligand_dict, ligand_path)
        del smi2chembert
        import gc; gc.collect()
        print(f"Saved ligand cache: {ligand_path}; freed smi2chembert.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return protein_dict, ligand_dict, aa_typical_buffers


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
def build_model(model_config: dict, mol_deg, prot_deg, device: str,
                aa_typical_buffers: dict | None = None):
    """Instantiate v1 or v4 from the config.

    For v4, ``aa_typical_buffers`` (containing ``aa_means`` and ``aa_stds``
    tensors) is registered onto the model as buffers via ``register_buffer``
    so the per-residue mean/std lookup happens inside the GPU forward pass
    without per-protein duplication.
    """
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
    elif version == "v4":
        model = TraceKinV4(
            mol_deg, prot_deg,
            **common_kwargs,
            chembert_dim=params.get("chembert_dim", 768),
            pocket_hidden=params.get("pocket_hidden", 64),
        )
        if aa_typical_buffers is None:
            raise ValueError(
                "v4 requires aa_typical_buffers (computed in preprocess); "
                "got None. Check that preprocess() ran with model_version='v4'."
            )
        model.register_buffer("aa_typical_means", aa_typical_buffers["aa_means"])
        model.register_buffer("aa_typical_stds", aa_typical_buffers["aa_stds"])
    elif version == "v5":
        model = TraceKinV5(
            mol_deg, prot_deg,
            prot_evo_channels=params["prot_evo_channels"],
            d_model=params.get("d_model", 512),
            n_mamba_layers=params.get("n_mamba_layers", 4),
            mamba_expand=params.get("mamba_expand", 2),
            mamba_d_state=params.get("mamba_d_state", 16),
            mamba_d_conv=params.get("mamba_d_conv", 4),
            graph_pe_rwse_steps=params.get("graph_pe_rwse_steps", 16),
            mol_in_channels=params["mol_in_channels"],
            n_drug_pna_layers=params.get("n_drug_pna_layers", 3),
            n_cross_heads=params.get("n_cross_heads", 8),
            chembert_dim=params.get("chembert_dim", 768),
            dropout=params.get("dropout", 0.1),
            regression_head=tasks["regression_task"],
            classification_head=tasks["classification_task"],
            multiclassification_head=tasks["mclassification_task"],
            device=device,
            heads=params.get("heads", 8),
        )
    elif version == "v5t":
        model = TraceKinV5T(
            mol_deg, prot_deg,
            prot_evo_channels=params["prot_evo_channels"],
            d_model=params.get("d_model", 512),
            n_transformer_layers=params.get("n_transformer_layers", 4),
            n_self_attn_heads=params.get("n_self_attn_heads", 8),
            ffn_expand=params.get("ffn_expand", 4),
            graph_pe_rwse_steps=params.get("graph_pe_rwse_steps", 16),
            mol_in_channels=params["mol_in_channels"],
            n_drug_pna_layers=params.get("n_drug_pna_layers", 3),
            n_cross_heads=params.get("n_cross_heads", 8),
            chembert_dim=params.get("chembert_dim", 768),
            dropout=params.get("dropout", 0.1),
            input_dropout=params.get("input_dropout", 0.15),
            regression_head=tasks["regression_task"],
            classification_head=tasks["classification_task"],
            multiclassification_head=tasks["mclassification_task"],
            device=device,
            heads=params.get("heads", 8),
        )
    elif version == "v6c":
        model = TraceKinV6C(
            mol_deg, prot_deg,
            prot_evo_channels=params["prot_evo_channels"],
            d_model=params.get("d_model", 512),
            n_gat_layers=params.get("n_gat_layers", 3),
            n_gat_heads=params.get("n_gat_heads", 8),
            gat_rbf_dim=params.get("gat_rbf_dim", 16),
            graph_pe_rwse_steps=params.get("graph_pe_rwse_steps", 16),
            pocket_k=params.get("pocket_k", 64),
            mol_in_channels=params["mol_in_channels"],
            n_drug_pna_layers=params.get("n_drug_pna_layers", 3),
            n_cross_heads=params.get("n_cross_heads", 8),
            chembert_dim=params.get("chembert_dim", 768),
            dropout=params.get("dropout", 0.1),
            regression_head=tasks["regression_task"],
            classification_head=tasks["classification_task"],
            multiclassification_head=tasks["mclassification_task"],
            device=device,
            heads=params.get("heads", 8),
        )
    elif version == "v7":
        model = TraceKinV7(
            mol_deg, prot_deg,
            prot_evo_channels=params["prot_evo_channels"],
            d_model=params.get("d_model", 512),
            n_gat_layers=params.get("n_gat_layers", 3),
            n_gat_heads=params.get("n_gat_heads", 8),
            n_global_heads=params.get("n_global_heads", 8),
            gat_rbf_dim=params.get("gat_rbf_dim", 16),
            graph_pe_rwse_steps=params.get("graph_pe_rwse_steps", 16),
            pocket_k=params.get("pocket_k", 64),
            pcer_k_catalytic=params.get("pcer_k_catalytic", 16),
            pcer_k_pocket=params.get("pcer_k_pocket", 64),
            use_mutation_gate=params.get("use_mutation_gate", True),
            use_graph_transformer=params.get("use_graph_transformer", True),
            n_pred_heads=params.get("n_pred_heads", 3),
            emb_recon_weight=params.get("emb_recon_weight", 0.1),
            mol_in_channels=params["mol_in_channels"],
            n_drug_pna_layers=params.get("n_drug_pna_layers", 3),
            mol_pe_mode=params.get("mol_pe_mode", "none"),
            mol_pe_steps=params.get("mol_pe_steps", 8),
            mol_pe_dim=params.get("mol_pe_dim", 16),
            n_cross_heads=params.get("n_cross_heads", 8),
            chembert_dim=params.get("chembert_dim", 768),
            dropout=params.get("dropout", 0.1),
            regression_head=tasks["regression_task"],
            classification_head=tasks["classification_task"],
            multiclassification_head=tasks["mclassification_task"],
            device=device,
            heads=params.get("heads", 8),
        )
    else:
        raise ValueError(f"Unknown model_version: {version!r}")

    model = model.to(device)
    model.reset_parameters()
    return model, version
# ---------------------------------------------------------------------------
# One ensemble member
# ---------------------------------------------------------------------------
def train_one_member(args, model_config, swa_block, device, seed,
                     mol_deg, prot_deg, degree_dict, aa_typical_buffers,
                     train_loader, val_loader, test_loader, test_df, ligand_dict):
    """Build, train and evaluate a single model, keyed by ``seed``.

    Everything upstream (caches, DataLoaders, degree stats) is built once by
    main() and shared, so an N-member ensemble costs N trainings — not N
    preprocessing runs. Artifacts are written per seed:
    ``save_model_seed<seed>/`` and ``test_prediction_seed<seed>.csv``.
    """
    # 6. Build model + dump config beside the checkpoint for reproducibility.
    os.makedirs(args.result_path, exist_ok=True)
    model_dir = os.path.join(args.result_path, f"save_model_seed{seed}")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(degree_dict, os.path.join(model_dir, "degree.pt"))

    model, model_version = build_model(
        model_config, mol_deg, prot_deg, args.device,
        aa_typical_buffers=aa_typical_buffers,
    )
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
        runid=seed,
        finetune_modules=finetune_modules,
        device=device,
        patience=model_config["optimizer"].get("patience", 0),
        use_swa=swa_block["use_swa"],
        swa_start_frac=swa_block.get("swa_start_frac", 0.75),
        swa_lr_factor=swa_block.get("swa_lr_factor", 0.1),
        use_amp=args.amp,
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
    interpret_path = os.path.join(args.result_path, f"interpretation_result_seed{seed}")
    os.makedirs(interpret_path, exist_ok=True)
    # virtual_screening mutates the frame it is handed (adds ID + prediction
    # columns), so each member gets its own copy.
    screen_df = virtual_screening(
        test_df.copy(), model, test_loader,
        result_path=interpret_path, save_interpret=args.save_interpret,
        ligand_dict=ligand_dict, device=device,
    )
    pred_csv = os.path.join(args.result_path, f"test_prediction_seed{seed}.csv")
    screen_df.to_csv(pred_csv, index=False)

    rmse = None
    if args.regression_task and "regression_label" in screen_df and "predicted_binding_affinity" in screen_df:
        y = screen_df["regression_label"].values.astype(float)
        p = screen_df["predicted_binding_affinity"].values.astype(float)
        mask = ~np.isnan(y) & ~np.isnan(p)
        if mask.any():
            metrics = evaluate_reg(y[mask], p[mask])
            rmse = metrics["rmse"]
            print(f"Final test metrics: rmse={metrics['rmse']:.4f}, pearson={metrics['pearson']:.4f}, "
                  f"r2_proxy(mse)={metrics['mse']:.4f}")

    print(f"Done. Predictions: {pred_csv}; checkpoint: {best_checkpoint}")

    del model, engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rmse



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

    print_mem("before load_split_dataframes")
    # 1. Lite parquet load (essential cols only) + dim detection.
    #    seq2feat / smi2chembert are deferred to preprocess (lazy on cache miss).
    (train_df, test_df, val_df, proteins, ligands,
     embedding_dim, chembert_dim, paths) = load_split_dataframes(args)
    print_mem("after load_split_dataframes (lite — feature cols not loaded)")

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

    # 3. Preprocess proteins/ligands (with cache). On cache hit (typical
    #    for seeds 2/3 after seed 1 builds), seq2feat/smi2chembert are
    #    NEVER loaded. On cache miss they're built transiently inside
    #    preprocess and freed immediately after the .pt is saved.
    #    model_version controls whether v4-specific aa_typical is computed.
    protein_dict, ligand_dict, aa_typical_buffers = preprocess(
        args, paths, proteins, ligands,
        model_version=model_config["model_version"],
    )
    print_mem("after preprocess")

    # ── NEW: prot_evo_channels comes from what protein.pt ACTUALLY holds. ──
    # The parquet-based detection at step 2 is only a guess: features may come
    # from ESMC (embedding_source='esmc') or from a hand-precomputed cache, in
    # which case --feature_col never described the real node features.
    if not protein_dict:
        raise ValueError("preprocess() returned an empty protein_dict.")
    _sample = next(iter(protein_dict.values()))["token_representation"]
    _true_dim = int(_sample.shape[-1])
    if _true_dim != model_config["params"]["prot_evo_channels"]:
        print(f"WARNING: parquet-detected prot_evo_channels="
              f"{model_config['params']['prot_evo_channels']} != protein.pt dim {_true_dim}. "
              f"Using {_true_dim}. If you did NOT intend this, you are probably reusing a "
              f"stale protein.pt built with a different embedding — rebuild the cache.")
    model_config["params"]["prot_evo_channels"] = _true_dim
    print(f"prot_evo_channels = {_true_dim} (from protein.pt, "
          f"per-residue={_sample.shape[0] > 1})")

    # 4. Build DataLoaders (with row filtering).
    train_loader, test_loader, val_loader, train_df, test_df, val_df = make_loaders(
        train_df, test_df, val_df, ligand_dict, protein_dict,
        device, args.batch_size, args.sampling_col
    )
    print(f"loaders ready: train={len(train_loader)} test={len(test_loader)} "
          f"val={len(val_loader) if val_loader is not None else 0}")
    print_mem("after make_loaders")

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

    # 6-10. Train each ensemble member. Seeds are args.seed, args.seed+1, ...
    #       so every member is independently reproducible (CatPred/Chemprop
    #       instead rely on the global RNG advancing between members, which
    #       gives diversity but no per-member reproducibility).
    seeds = [args.seed + i for i in range(max(1, args.ensemble_size))]
    if len(seeds) > 1:
        print("=" * 60)
        print(f"Ensemble training: {len(seeds)} members, seeds {seeds}")
        print("Caches, DataLoaders and degree stats are shared across members.")
        print("=" * 60)

    member_rmse = []
    for i, seed in enumerate(seeds):
        if len(seeds) > 1:
            print(f"\n----- member {i + 1}/{len(seeds)} (seed {seed}) -----")
        set_random_seed(seed)
        rmse = train_one_member(
            args, model_config, swa_block, device, seed,
            mol_deg, prot_deg, degree_dict, aa_typical_buffers,
            train_loader, val_loader, test_loader, test_df, ligand_dict,
        )
        member_rmse.append(rmse)

    if len(seeds) > 1:
        scored = [r for r in member_rmse if r is not None]
        print("\n" + "=" * 60)
        print(f"Ensemble training complete: {len(seeds)} members in {args.result_path}")
        if scored:
            print(f"  per-member rmse: " + ", ".join(f"{r:.4f}" for r in scored))
            print(f"  mean={np.mean(scored):.4f} best={min(scored):.4f} worst={max(scored):.4f}")
        print("  average them with:")
        print(f"    python inference/predict_ensemble.py \\")
        print(f"      --weights_glob '{args.result_path}/save_model_seed*' \\")
        print(f"      --datafolder {args.datafolder} --split test \\")
        print(f"      --output {args.result_path}/ensemble_test.csv")
        print("=" * 60)


if __name__ == "__main__":
    main()

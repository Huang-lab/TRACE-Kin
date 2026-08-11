#!/usr/bin/env python
"""Standalone TRACE-Kin inference — score a parquet/csv with a trained checkpoint.

Replaces the v1/v2-only ``trace_kin_predictor.py`` for the live model versions
(v1, v4, v5, v5t, v6c, v7). No training loop, no optimizer, no Trainer.

Design notes
------------
* **Single source of truth for model construction.** This imports
  :func:`training.train_trace_kin.build_model`, the same dispatcher training
  uses, so a new ``net_v*`` never needs a second code path here (the old
  predictor drifted precisely because it duplicated the dispatch and only knew
  v1/v2).
* **Reuses the preprocessing caches.** Protein/ligand graphs are read from
  ``<datafolder>/protein.pt`` and ``ligand.pt``. Nothing re-runs ESM2/ESMC or
  MoLFormer, so inference is cheap even for a 6B contact model.
* **Dimensions come from the checkpoint's own ``config.json``**, which training
  writes after resolving ``prot_evo_channels`` from the real features — so a
  per-residue or ESMC-sourced run is reconstructed exactly.
* **Interpretation output is OFF by default** (the opposite of training).
  ``--interpret_dir`` opts in. Per-pair ``PAIR_*/`` directories are thousands
  of small files and readily exhaust disk/inode quota.

Usage
-----
Score the test split of a dataset folder::

    python inference/predict.py \
        --weights_dir results/kcat_v7/save_model_seed1 \
        --datafolder  data/kcat_v7 \
        --split test \
        --output results/kcat_v7/test_prediction_seed1.csv \
        --protein_col sequence --ligand_col smiles --label_col log10_value

Score an arbitrary table (its proteins/ligands must exist in the caches)::

    python inference/predict.py --weights_dir ... --datafolder ... \
        --input_file /path/to/new_pairs.parquet --output preds.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

# Make `training` / `models` importable when run as a script from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.loader import DataLoader

from training.dataset import ProteinMoleculeDataset
from training.data_utils import virtual_screening
from training.train_trace_kin import (build_model, find_data_file, load_dataframe,
                                      normalize_columns)
from training.metrics import evaluate_reg


def parse_args():
    p = argparse.ArgumentParser(description="Standalone TRACE-Kin inference (v1/v4/v5/v5t/v6c/v7).")
    p.add_argument("--weights_dir", required=True,
                   help="Directory holding config.json, degree.pt and model.pt "
                        "(training writes these to <result_path>/save_model_seed<N>/).")
    p.add_argument("--datafolder", required=True,
                   help="Dataset folder holding protein.pt / ligand.pt caches (and the split file).")
    p.add_argument("--split", default="test", choices=["train", "val", "test"],
                   help="Which split in --datafolder to score (ignored if --input_file is given).")
    p.add_argument("--input_file", default="",
                   help="Score this table instead of a split. Proteins/ligands must be "
                        "present in the datafolder caches.")
    p.add_argument("--output", required=True, help="Destination CSV for predictions.")

    p.add_argument("--protein_col", default="Protein")
    p.add_argument("--ligand_col", default="Ligand")
    p.add_argument("--label_col", default="regression_label",
                   help="Ground-truth column, if present. Metrics are reported when it exists.")

    p.add_argument("--interpret_dir", default="",
                   help="Opt in to per-pair interpretation output (cross-attention scores + "
                        "interaction fingerprints) in this directory. WARNING: one "
                        "subdirectory per pair — thousands of small files.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available()
                          or not str(args.device).startswith("cuda") else "cpu")
    if str(device) != args.device:
        print(f"WARN: CUDA unavailable; using {device}")

    # ---------------- checkpoint artifacts ----------------
    config_path = os.path.join(args.weights_dir, "config.json")
    degree_path = os.path.join(args.weights_dir, "degree.pt")
    ckpt_path = os.path.join(args.weights_dir, "model.pt")
    for path in (config_path, degree_path, ckpt_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"missing {path}. --weights_dir must be a training run's "
                f"save_model_seed<N>/ directory."
            )

    with open(config_path) as f:
        model_config = json.load(f)
    version = model_config.get("model_version", "v1")

    degree_dict = torch.load(degree_path, map_location="cpu")
    mol_deg = degree_dict["ligand_deg"]
    prot_deg = degree_dict["protein_deg"]

    # ---------------- feature caches ----------------
    protein_path = os.path.join(args.datafolder, "protein.pt")
    ligand_path = os.path.join(args.datafolder, "ligand.pt")
    for path in (protein_path, ligand_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"missing {path}. Run training/preprocessing once for this datafolder "
                f"so the graph caches exist; inference does not build them."
            )
    print(f"Loading caches from {args.datafolder} ...")
    protein_dict = torch.load(protein_path)
    ligand_dict = torch.load(ligand_path)
    print(f"  proteins={len(protein_dict)} ligands={len(ligand_dict)}")

    # v4 needs the aa_typical buffers registered onto the model.
    aa_typical_buffers = None
    if version == "v4":
        from training.aa_typical import load_or_compute_aa_typical, aa_typical_to_buffers
        aa_typical = load_or_compute_aa_typical(
            cache_path=os.path.join(args.datafolder, "aa_typical.pt"),
            protein_dict=protein_dict, force_rebuild=False)
        _, aa_means, aa_stds = aa_typical_to_buffers(aa_typical)
        aa_typical_buffers = {"aa_means": aa_means, "aa_stds": aa_stds}

    # ---------------- rows to score ----------------
    if args.input_file:
        table_path = args.input_file
    else:
        table_path = find_data_file(args.datafolder, args.split)
        if table_path is None:
            raise FileNotFoundError(f"no {args.split}.[parquet|csv|tsv] in {args.datafolder}")
    print(f"Scoring: {table_path}")

    # Read every column: the label column name is user-supplied, and heavy
    # feature columns are irrelevant here (graphs come from the caches).
    df = load_dataframe(table_path, columns=None)
    df = normalize_columns(df, args.protein_col, args.ligand_col, args.label_col)

    before = len(df)
    valid = df["Ligand"].isin(ligand_dict) & df["Protein"].isin(protein_dict)
    if (~valid).any():
        print(f"WARN: dropping {int((~valid).sum())} of {before} rows whose protein/ligand "
              f"is absent from the caches.")
    df = df[valid].reset_index(drop=True)
    if df.empty:
        raise ValueError("no rows left to score after cache filtering.")
    print(f"  {len(df)} rows")

    # ---------------- model ----------------
    model, built_version = build_model(model_config, mol_deg, prot_deg, str(device),
                                       aa_typical_buffers)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded TraceKin{built_version.upper()} ({n_params:,} params) from {ckpt_path}")
    print(f"  prot_evo_channels={model_config['params']['prot_evo_channels']}, "
          f"chembert_dim={model_config['params'].get('chembert_dim')}")

    # ---------------- inference ----------------
    ds = ProteinMoleculeDataset(df, ligand_dict, protein_dict, device=device)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        follow_batch=["mol_x", "clique_x", "prot_node_aa"])

    save_interpret = bool(args.interpret_dir)
    interpret_path = args.interpret_dir or args.datafolder  # unused when save_interpret=False
    if save_interpret:
        os.makedirs(interpret_path, exist_ok=True)
        print(f"Interpretation output ON -> {interpret_path} (one dir per pair)")

    pred_df = virtual_screening(
        df, model, loader,
        result_path=interpret_path,
        save_interpret=save_interpret,
        ligand_dict=ligand_dict,
        device=device,
    )

    # ---------------- write + report ----------------
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pred_df.to_csv(args.output, index=False)
    print(f"Wrote {len(pred_df)} predictions -> {args.output}")

    if "regression_label" in pred_df and "predicted_binding_affinity" in pred_df:
        y = pred_df["regression_label"].values.astype(float)
        yhat = pred_df["predicted_binding_affinity"].values.astype(float)
        mask = ~np.isnan(y) & ~np.isnan(yhat)
        if mask.any():
            m = evaluate_reg(y[mask], yhat[mask])
            print(f"Metrics on {int(mask.sum())} labelled rows: "
                  f"rmse={m['rmse']:.4f} pearson={m['pearson']:.4f} mse={m['mse']:.4f}")
        else:
            print("No labelled rows — metrics skipped (predictions still written).")


if __name__ == "__main__":
    main()

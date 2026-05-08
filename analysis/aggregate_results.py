#!/usr/bin/env python
"""Aggregate v3 rerun results into a single CSV.

After the v3 array runs on HPC, the result tree looks like::

    TRACE_Kin_Results_v3/
        MPEK_dataset/MPEK_kcat_ESMv1_embedding_random/
            test_prediction_seed1.csv
            test_prediction_seed2.csv     # if a second seed has been run
            test_prediction_seed3.csv     # if a third seed has been run
            full_result-1.txt
        ...

This script walks the tree, detects all available seeds per task, **averages
the predictions across seeds before computing RMSE** (the v3 paper config is
the multi-seed ensemble — RF gets bagging for free, so v3 must too), and
joins against the cleaned baseline CSV to attach v1 RMSE and the historical
RF best for each dataset.

Output: ``data/benchmark/trace_kin_v3_results.csv``.

Tasks with only one seed file fall through unchanged (single-seed RMSE).
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


# Map dataset folder name -> (k_type, split_mode, embedding) for joining.
NAME_RE = re.compile(
    r"^(?P<dataset_name>[A-Za-z]+)_(?P<k_type>[A-Za-z]+)_(?P<embedding>[A-Za-z0-9]+)_embedding_(?P<split>[a-z_]+)$"
)
# k_type aliases used in dataset directory names (kkm vs kcat_km, Kd vs kd...).
K_TYPE_DIR_TO_CANONICAL = {
    "kkm": "kcat_km",
    "kcat": "kcat",
    "km": "km",
    "ki": "ki",
    "Kd": "kd",
    "kd": "kd",
}
EMBEDDING_DIR_TO_CANONICAL = {
    "ESM2": "ESM2",
    "ESMv1": "ESMv1",
    "ESM1v": "ESMv1",
    "MUTAPLM": "MutaPLM",
    "MutaPLM": "MutaPLM",
    "ProteinCLIP": "ProteinCLIP",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", required=True,
                   help="Directory containing per-dataset v3 result subfolders.")
    p.add_argument("--clean_csv", default="data/benchmark/kinetic_regress_benchmark_clean.csv",
                   help="Baseline benchmark CSV; used to look up v1 RMSE and RF best.")
    p.add_argument("--output", default="data/benchmark/trace_kin_v3_results.csv")
    return p.parse_args()


def parse_dataset_name(folder_name: str) -> dict | None:
    m = NAME_RE.match(folder_name)
    if not m:
        return None
    parts = m.groupdict()
    parts["k_type"] = K_TYPE_DIR_TO_CANONICAL.get(parts["k_type"], parts["k_type"].lower())
    parts["embedding"] = EMBEDDING_DIR_TO_CANONICAL.get(parts["embedding"], parts["embedding"])
    parts["split"] = parts["split"].strip("_")  # cold_drug, cold_protein, random
    parts["dataset_name"] = parts["dataset_name"].lower() if parts["dataset_name"].lower() == "inhouse" else parts["dataset_name"]
    return parts


def compute_rmse_from_seeds(pred_csvs: list[Path]) -> tuple[float | None, int]:
    """Recompute RMSE from one or more test_prediction_seed*.csv files.

    When multiple seed files are present, predictions are averaged row-by-row
    (matched on (Protein, Ligand) keys when present, otherwise on row order)
    *before* RMSE is computed. The returned int is the number of seeds
    successfully averaged.
    """
    frames = []
    for pred_csv in sorted(pred_csvs):
        df = pd.read_csv(pred_csv)
        label_col = next((c for c in ("regression_label", "Label") if c in df.columns), None)
        pred_col = next((c for c in ("predicted_binding_affinity",) if c in df.columns), None)
        if label_col is None or pred_col is None:
            continue
        keep_cols = [label_col, pred_col]
        if "Protein" in df.columns and "Ligand" in df.columns:
            keep_cols = ["Protein", "Ligand"] + keep_cols
        frames.append(df[keep_cols].rename(columns={label_col: "label", pred_col: "pred"}))

    if not frames:
        return None, 0

    if len(frames) == 1:
        valid = frames[0][["label", "pred"]].dropna()
        if valid.empty:
            return None, 0
        return float(np.sqrt(mean_squared_error(valid["label"], valid["pred"]))), 1

    # Multi-seed: align on (Protein, Ligand) keys when available, otherwise
    # assume row-aligned (every seed wrote rows in the same order).
    if "Protein" in frames[0].columns and "Ligand" in frames[0].columns:
        merged = frames[0].rename(columns={"pred": "pred_0"})
        for i, fr in enumerate(frames[1:], start=1):
            merged = merged.merge(
                fr.rename(columns={"pred": f"pred_{i}"})[["Protein", "Ligand", f"pred_{i}"]],
                on=["Protein", "Ligand"], how="inner",
            )
        pred_cols = [c for c in merged.columns if c.startswith("pred_")]
    else:
        # Row-aligned fallback. This is brittle (assumes the trainer wrote
        # rows in the same order across seeds), but the trainer's
        # virtual_screening function does iterate the test loader in the
        # fixed dataset order, so this is fine in practice.
        merged = frames[0].rename(columns={"pred": "pred_0"})
        for i, fr in enumerate(frames[1:], start=1):
            merged[f"pred_{i}"] = fr["pred"].values
        pred_cols = [c for c in merged.columns if c.startswith("pred_")]

    merged = merged.dropna(subset=["label"] + pred_cols)
    if merged.empty:
        return None, 0
    avg_pred = merged[pred_cols].mean(axis=1)
    return float(np.sqrt(mean_squared_error(merged["label"], avg_pred))), len(pred_cols)


def lookup_v1_and_rf(clean: pd.DataFrame, k_type: str, split: str,
                     embedding: str, dataset_name: str) -> tuple[float | None, float | None]:
    test_only = clean[(clean["Dataset"] == "test")
                       & (clean["k_type"] == k_type)
                       & (clean["split_mode"] == split)
                       & (clean["embedding_type"] == embedding)
                       & (clean["dataset_name"] == dataset_name)]
    psichic = test_only[test_only["Model"] == "PSICHIC"]["RMSE"].min()
    rf = test_only[test_only["Model"] == "Random Forest"]["RMSE"].min()
    psichic = float(psichic) if pd.notna(psichic) else None
    rf = float(rf) if pd.notna(rf) else None
    return psichic, rf


def main() -> None:
    args = parse_args()
    root = Path(args.results_root)
    if not root.exists():
        raise FileNotFoundError(f"results root does not exist: {root}")
    clean = pd.read_csv(args.clean_csv)

    rows = []
    for parent_dir in sorted(root.iterdir()):
        if not parent_dir.is_dir():
            continue
        for ds_dir in sorted(parent_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            pred_files = list(ds_dir.glob("test_prediction_seed*.csv"))
            if not pred_files:
                continue
            v3_rmse, n_seeds = compute_rmse_from_seeds(pred_files)
            parsed = parse_dataset_name(ds_dir.name)
            if parsed is None:
                print(f"WARN: cannot parse dataset name {ds_dir.name}, skipping")
                continue
            v1_rmse, rf_rmse = lookup_v1_and_rf(
                clean, parsed["k_type"], parsed["split"],
                parsed["embedding"], parsed["dataset_name"],
            )
            row = {
                "dataset_folder": ds_dir.name,
                "dataset_name": parsed["dataset_name"],
                "k_type": parsed["k_type"],
                "split_mode": parsed["split"],
                "embedding": parsed["embedding"],
                "n_seeds": n_seeds,
                "v3_rmse": v3_rmse,
                "v1_rmse": v1_rmse,
                "rf_best_rmse": rf_rmse,
                "v3_minus_v1": (v3_rmse - v1_rmse) if (v3_rmse is not None and v1_rmse is not None) else None,
                "v3_minus_rf": (v3_rmse - rf_rmse) if (v3_rmse is not None and rf_rmse is not None) else None,
            }
            rows.append(row)

    if not rows:
        print("WARN: no v3 results found.")
    out = pd.DataFrame(rows).sort_values(["k_type", "split_mode"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(out)} rows)")


if __name__ == "__main__":
    main()

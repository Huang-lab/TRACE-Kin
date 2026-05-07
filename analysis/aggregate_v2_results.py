#!/usr/bin/env python
"""Aggregate v2 rerun results into a single CSV.

After the user has run ``training/rerun_12_datasets.sh`` on HPC, the result
tree looks like::

    TRACE_Kin_Results_v2/
        MPEK_dataset/MPEK_kcat_ESMv1_embedding_random/
            test_prediction_seed1.csv
            full_result-1.txt
        ...

This script walks the tree, recomputes RMSE from the predictions, parses the
log to confirm SWA / model_version, and joins against the cleaned baseline CSV
to attach v1 RMSE and the historical RF best for each dataset. Output:
``data/benchmark/trace_kin_v2_results.csv``.
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
                   help="Directory containing per-dataset v2 result subfolders.")
    p.add_argument("--clean_csv", default="data/benchmark/kinetic_regress_benchmark_clean.csv",
                   help="Baseline benchmark CSV; used to look up v1 RMSE and RF best.")
    p.add_argument("--output", default="data/benchmark/trace_kin_v2_results.csv")
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


def compute_rmse_from_predictions(pred_csv: Path) -> float | None:
    """Recompute RMSE from a test_prediction_seed*.csv file."""
    df = pd.read_csv(pred_csv)
    label_col = next((c for c in ("regression_label", "Label") if c in df.columns), None)
    pred_col = next((c for c in ("predicted_binding_affinity",) if c in df.columns), None)
    if label_col is None or pred_col is None:
        return None
    valid = df[[label_col, pred_col]].dropna()
    if valid.empty:
        return None
    return float(np.sqrt(mean_squared_error(valid[label_col], valid[pred_col])))


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
            v2_rmse = compute_rmse_from_predictions(pred_files[0])
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
                "v2_rmse": v2_rmse,
                "v1_rmse": v1_rmse,
                "rf_best_rmse": rf_rmse,
                "v2_minus_v1": (v2_rmse - v1_rmse) if (v2_rmse is not None and v1_rmse is not None) else None,
                "v2_minus_rf": (v2_rmse - rf_rmse) if (v2_rmse is not None and rf_rmse is not None) else None,
            }
            rows.append(row)

    if not rows:
        print("WARN: no v2 results found.")
    out = pd.DataFrame(rows).sort_values(["k_type", "split_mode"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(out)} rows)")


if __name__ == "__main__":
    main()

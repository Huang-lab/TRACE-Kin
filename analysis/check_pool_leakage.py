#!/usr/bin/env python
"""Check cross-dataset pool leakage for the v3 pooled tasks (1-5).

For each pooled task in the JOBS array, compute three leakage metrics:

  1. Exact pair leakage: |test ∩ pool_train| on (Protein, Smiles) keys.
     Direct training-test memorization risk.

  2. Cold-drug leakage (only for cold_drug splits): |test_smiles ∩ pool_train_smiles|.
     Defeats the "model has never seen this drug" guarantee.

  3. Cold-protein leakage (only for cold_protein splits): |test_proteins ∩ pool_train_proteins|.
     Defeats the "model has never seen this protein" guarantee.

Output: stdout table + optional CSV.

Usage:
    python analysis/check_pool_leakage.py \
        --enzyme_base /sc/arion/projects/.../enzyme_embeddings_dataset \
        [--output data/benchmark/pool_leakage.csv]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import pandas as pd


# Mirror of the JOBS array from training/run_v3_array.lsf — pooled tasks 1-5.
# (source_dataset, primary_dataset_name, [pool_dataset_relpaths])
POOLED_JOBS = [
    ("EITLEM_dataset", "EITLEM_km_ESMv1_embedding_random",
        ["MPEK_dataset/MPEK_km_ESMv1_embedding_random",
         "catpred_dataset/catpred_km_ESMv1_embedding_random"]),
    ("EITLEM_dataset", "EITLEM_km_ProteinCLIP_embedding_cold_protein",
        ["MPEK_dataset/MPEK_km_ProteinCLIP_embedding_cold_protein",
         "catpred_dataset/catpred_km_ProteinCLIP_embedding_cold_protein"]),
    ("MPEK_dataset", "MPEK_kcat_ESMv1_embedding_random",
        ["EITLEM_dataset/EITLEM_kcat_ESMv1_embedding_random",
         "catpred_dataset/catpred_kcat_ESMv1_embedding_random"]),
    ("MPEK_dataset", "MPEK_kcat_ESMv1_embedding_cold_drug",
        ["EITLEM_dataset/EITLEM_kcat_ESMv1_embedding_cold_drug",
         "catpred_dataset/catpred_kcat_ESMv1_embedding_cold_drug"]),
    ("MPEK_dataset", "MPEK_kcat_ESM2_embedding_cold_protein",
        ["EITLEM_dataset/EITLEM_kcat_ESM2_embedding_cold_protein",
         "catpred_dataset/catpred_kcat_ESM2_embedding_cold_protein"]),
]


def load_split(folder: Path, split_name: str) -> pd.DataFrame | None:
    """Load test/train/val with parquet preferred, csv fallback."""
    for ext in (".parquet", ".csv", ".tsv"):
        path = folder / f"{split_name}{ext}"
        if path.exists():
            if ext == ".parquet":
                return pd.read_parquet(path)
            return pd.read_csv(path)
    return None


def detect_split_mode(name: str) -> str:
    if "cold_drug" in name:
        return "cold_drug"
    if "cold_protein" in name:
        return "cold_protein"
    if "random" in name:
        return "random"
    return "unknown"


def keys_pair(df: pd.DataFrame) -> set:
    return set(zip(df["Protein"].astype(str), df["Smiles"].astype(str)))


def keys_smiles(df: pd.DataFrame) -> set:
    return set(df["Smiles"].astype(str))


def keys_protein(df: pd.DataFrame) -> set:
    return set(df["Protein"].astype(str))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enzyme_base", required=True,
                    help="Root of the enzyme_embeddings_dataset tree.")
    ap.add_argument("--output", default=None,
                    help="Optional CSV summary path.")
    args = ap.parse_args()

    base = Path(args.enzyme_base)
    if not base.exists():
        print(f"ERROR: enzyme_base does not exist: {base}", file=sys.stderr)
        return 2

    rows = []
    for src, primary, pool_relpaths in POOLED_JOBS:
        primary_dir = base / src / primary
        split = detect_split_mode(primary)
        print(f"\n=== {primary}  (split={split}) ===")
        print(f"  primary: {primary_dir}")

        test_df = load_split(primary_dir, "test")
        if test_df is None:
            print(f"  ERROR: no test.parquet/csv found in {primary_dir}")
            continue

        n_test = len(test_df)
        test_pairs = keys_pair(test_df)
        test_smi = keys_smiles(test_df)
        test_prot = keys_protein(test_df)
        print(f"  primary test rows: {n_test}  (unique pairs: {len(test_pairs)}, "
              f"unique drugs: {len(test_smi)}, unique proteins: {len(test_prot)})")

        # Aggregate pool train across all pool sources
        pool_pairs: set = set()
        pool_smi: set = set()
        pool_prot: set = set()
        pool_total_rows = 0
        for rel in pool_relpaths:
            pool_dir = base / rel
            train_df = load_split(pool_dir, "train")
            if train_df is None:
                print(f"  WARN: no train file in {pool_dir} — skipping")
                continue
            pool_total_rows += len(train_df)
            pool_pairs |= keys_pair(train_df)
            pool_smi |= keys_smiles(train_df)
            pool_prot |= keys_protein(train_df)
            print(f"  pool source: {rel}  (train rows: {len(train_df)})")

        if not pool_pairs:
            print("  no pool data found, nothing to check")
            continue

        # Type 1: exact pair leakage
        exact_leak = test_pairs & pool_pairs
        exact_pct = len(exact_leak) * 100 / len(test_pairs) if test_pairs else 0

        # Type 2/3: split-specific leakage
        if split == "cold_drug":
            soft_leak_keys = test_smi & pool_smi
            soft_label = "cold_drug — drugs in pool_train"
            soft_total = len(test_smi)
        elif split == "cold_protein":
            soft_leak_keys = test_prot & pool_prot
            soft_label = "cold_protein — proteins in pool_train"
            soft_total = len(test_prot)
        else:
            soft_leak_keys = set()
            soft_label = f"({split}: no soft-leak check applies)"
            soft_total = 0
        soft_pct = (len(soft_leak_keys) * 100 / soft_total) if soft_total else 0

        print(f"  EXACT pair leakage:  {len(exact_leak):>6} of {n_test:>6} test rows "
              f"({exact_pct:5.1f}%)")
        if soft_total > 0:
            print(f"  SOFT  leakage:       {len(soft_leak_keys):>6} of {soft_total:>6} unique test "
                  f"{'drugs' if split == 'cold_drug' else 'proteins'} "
                  f"({soft_pct:5.1f}%) — {soft_label}")
        else:
            print(f"  SOFT  leakage:       {soft_label}")

        # Verdict
        if exact_pct > 5 or soft_pct > 10:
            verdict = "⚠️  CONCERNING"
        elif exact_pct > 1 or soft_pct > 5:
            verdict = "moderate"
        else:
            verdict = "clean"
        print(f"  Verdict: {verdict}")

        rows.append({
            "task": primary,
            "split_mode": split,
            "n_test_rows": n_test,
            "n_pool_train_rows": pool_total_rows,
            "exact_leak_count": len(exact_leak),
            "exact_leak_pct": round(exact_pct, 2),
            "soft_leak_count": len(soft_leak_keys) if soft_total else 0,
            "soft_leak_pct": round(soft_pct, 2) if soft_total else "",
            "verdict": verdict,
        })

    if args.output and rows:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.output}")

    print("\n" + "=" * 80)
    print("Summary:")
    for r in rows:
        print(f"  {r['task']:<55} exact={r['exact_leak_pct']:>5}%  "
              f"soft={r['soft_leak_pct']:>5}%  ({r['verdict']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

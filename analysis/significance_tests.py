#!/usr/bin/env python
"""Wilcoxon paired-t significance tests on the cleaned benchmark.

Two test families:
  (a) For each pair of embeddings (ESM2 vs MutaPLM, ESM2 vs ProteinCLIP, ...),
      build matched (Model, k_type, split_mode, dataset_name) rows and run a
      Wilcoxon signed-rank test on the RMSE differences.
  (b) If v2 results are available, run Wilcoxon on the 14 paired (v1 RMSE,
      v2 RMSE) rows from the redesign rerun.

Outputs ``data/benchmark/significance.csv`` with columns:
``family, label_a, label_b, n_pairs, statistic, p_value, mean_delta_rmse``.
"""
from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clean_csv", default="data/benchmark/kinetic_regress_benchmark_clean.csv")
    p.add_argument("--v2_csv", default="data/benchmark/trace_kin_v2_results.csv")
    p.add_argument("--output", default="data/benchmark/significance.csv")
    return p.parse_args()


def pairwise_embedding_tests(df: pd.DataFrame) -> list[dict]:
    """Wilcoxon on RMSE diffs between embedding pairs, matched on (Model, k, split, dataset)."""
    test_only = df[df["Dataset"] == "test"].copy()
    embeddings = sorted(test_only["embedding_type"].dropna().unique())
    pivot = test_only.pivot_table(
        index=["Model", "k_type", "split_mode", "dataset_name"],
        columns="embedding_type",
        values="RMSE",
        aggfunc="min",
    )
    rows = []
    for a, b in itertools.combinations(embeddings, 2):
        if a not in pivot.columns or b not in pivot.columns:
            continue
        paired = pivot[[a, b]].dropna()
        if len(paired) < 5:  # too few to test
            continue
        try:
            stat, pval = stats.wilcoxon(paired[a], paired[b])
        except ValueError:
            continue  # all-zero diffs, no test
        rows.append({
            "family": "embedding_pair",
            "label_a": a,
            "label_b": b,
            "n_pairs": len(paired),
            "statistic": float(stat),
            "p_value": float(pval),
            "mean_delta_rmse": float((paired[a] - paired[b]).mean()),
        })
    return rows


def v1_vs_v2_test(v2_csv: str) -> list[dict]:
    if not os.path.exists(v2_csv):
        return []
    df = pd.read_csv(v2_csv)
    if "v1_rmse" not in df.columns or "v2_rmse" not in df.columns:
        return []
    paired = df.dropna(subset=["v1_rmse", "v2_rmse"])
    if len(paired) < 3:
        return []
    stat, pval = stats.wilcoxon(paired["v1_rmse"], paired["v2_rmse"])
    return [{
        "family": "v1_vs_v2",
        "label_a": "TRACE-Kin v1",
        "label_b": "TRACE-Kin v2",
        "n_pairs": len(paired),
        "statistic": float(stat),
        "p_value": float(pval),
        "mean_delta_rmse": float((paired["v1_rmse"] - paired["v2_rmse"]).mean()),
    }]


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.clean_csv)
    rows = pairwise_embedding_tests(df) + v1_vs_v2_test(args.v2_csv)
    out = pd.DataFrame(rows)
    if out.empty:
        print("WARN: no significance results produced.")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(out)} test results)")


if __name__ == "__main__":
    main()

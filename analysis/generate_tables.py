#!/usr/bin/env python
"""Generate publication-ready benchmark tables.

Inputs:
  - data/benchmark/kinetic_regress_benchmark_clean.csv (from clean_benchmark.py)
  - data/benchmark/trace_kin_v2_results.csv (optional — from aggregate_v2_results.py)

Outputs (LaTeX + Markdown side by side, in data/benchmark/tables/):
  - table_main_best_per_kinetic_split  Best RMSE/R²/Pearson per (kinetic × split)
                                        across all models/embeddings, with v2
                                        column where reran.
  - table_supp_full                     Full per-config benchmark.
  - table_supp_psichic_v2_vs_rf         PSICHIC v1 vs v2 vs RF gap analysis.
  - table_supp_mutaplm_vs_esm2          Head-to-head MutaPLM vs ESM2 wins/losses.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clean_csv", default="data/benchmark/kinetic_regress_benchmark_clean.csv")
    p.add_argument("--v2_csv", default="data/benchmark/trace_kin_v2_results.csv",
                   help="Optional v2 rerun results; columns: dataset_name,k_type,split_mode,v2_rmse,v1_rmse,rf_best_rmse")
    p.add_argument("--output_dir", default="data/benchmark/tables")
    return p.parse_args()


def best_per_kinetic_split(df: pd.DataFrame) -> pd.DataFrame:
    """Lowest RMSE per (k_type, split_mode) across all models/embeddings/datasets."""
    test_only = df[df["Dataset"] == "test"].copy()
    rows = []
    for (k, split), grp in test_only.groupby(["k_type", "split_mode"]):
        idx = grp["RMSE"].idxmin()
        best = grp.loc[idx]
        rows.append({
            "k_type": k,
            "split_mode": split,
            "best_RMSE": best["RMSE"],
            "best_R2": best["R2"],
            "best_Pearson": best["Pearson"],
            "best_Model": best["Model"],
            "best_embedding": best["embedding_type"],
            "best_dataset": best["dataset_name"],
        })
    return pd.DataFrame(rows).sort_values(["k_type", "split_mode"]).reset_index(drop=True)


def psichic_vs_rf_gap(df: pd.DataFrame) -> pd.DataFrame:
    """For each (k_type, split, embedding, dataset) row, compare PSICHIC RMSE to RF RMSE."""
    test_only = df[df["Dataset"] == "test"].copy()
    pivot = test_only.pivot_table(
        index=["k_type", "split_mode", "embedding_type", "dataset_name"],
        columns="Model",
        values="RMSE",
        aggfunc="min",
    ).reset_index()
    if "PSICHIC" not in pivot.columns or "Random Forest" not in pivot.columns:
        return pd.DataFrame()
    pivot["gap_psichic_minus_rf"] = pivot["PSICHIC"] - pivot["Random Forest"]
    pivot["winner"] = np.where(
        pivot["gap_psichic_minus_rf"] < 0, "PSICHIC",
        np.where(pivot["gap_psichic_minus_rf"] > 0, "Random Forest", "Tie"),
    )
    return pivot.sort_values("gap_psichic_minus_rf").reset_index(drop=True)


def mutaplm_vs_esm2(df: pd.DataFrame) -> pd.DataFrame:
    """MutaPLM vs ESM2 head-to-head wins/losses by (Model, k_type, split_mode)."""
    test_only = df[df["Dataset"] == "test"].copy()
    grp = test_only[test_only["embedding_type"].isin(["MutaPLM", "ESM2"])]
    pivot = grp.pivot_table(
        index=["Model", "k_type", "split_mode", "dataset_name"],
        columns="embedding_type",
        values="RMSE",
        aggfunc="min",
    ).reset_index()
    if "MutaPLM" not in pivot.columns or "ESM2" not in pivot.columns:
        return pd.DataFrame()
    pivot = pivot.dropna(subset=["MutaPLM", "ESM2"])
    pivot["delta_mutaplm_minus_esm2"] = pivot["MutaPLM"] - pivot["ESM2"]
    return pivot.sort_values("delta_mutaplm_minus_esm2").reset_index(drop=True)


def merge_v2(main_table: pd.DataFrame, v2_path: str) -> pd.DataFrame:
    """If v2 results exist, attach a v2_RMSE column to the main table."""
    if not os.path.exists(v2_path):
        return main_table
    v2 = pd.read_csv(v2_path)
    out = main_table.merge(
        v2[["k_type", "split_mode", "v2_rmse"]],
        on=["k_type", "split_mode"], how="left",
    )
    return out


def df_to_markdown(df: pd.DataFrame) -> str:
    """Hand-rolled GFM table writer so we don't depend on ``tabulate``."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append("" if pd.isna(v) else f"{v:.4f}")
            else:
                cells.append("" if pd.isna(v) else str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def write_table(df: pd.DataFrame, name: str, output_dir: str) -> None:
    if df.empty:
        print(f"WARN: skipping {name}, empty dataframe")
        return
    md_path = os.path.join(output_dir, f"{name}.md")
    tex_path = os.path.join(output_dir, f"{name}.tex")
    df_round = df.copy()
    for col in df_round.select_dtypes(include="number").columns:
        df_round[col] = df_round[col].round(4)
    with open(md_path, "w") as f:
        f.write(df_to_markdown(df_round))
        f.write("\n")
    with open(tex_path, "w") as f:
        f.write(df_round.to_latex(index=False, float_format="%.4f"))
    print(f"Wrote {md_path} + {tex_path}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.clean_csv)

    main_tbl = best_per_kinetic_split(df)
    main_tbl = merge_v2(main_tbl, args.v2_csv)
    write_table(main_tbl, "table_main_best_per_kinetic_split", args.output_dir)

    write_table(df, "table_supp_full", args.output_dir)

    gap = psichic_vs_rf_gap(df)
    write_table(gap, "table_supp_psichic_v2_vs_rf", args.output_dir)

    mp_vs_esm = mutaplm_vs_esm2(df)
    write_table(mp_vs_esm, "table_supp_mutaplm_vs_esm2", args.output_dir)

    print(f"All tables written to {args.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Generate publication figures from the cleaned benchmark + v2 results.

Five figures (PDF + PNG, written to figures/output/):

* ``fig2b_heatmap_best_rmse``    — Heatmap of best RMSE per (k_type, embedding, split).
* ``fig2c_embedding_scatter``    — ESM2 vs MutaPLM RMSE scatter colored by split.
* ``fig2d_cold_degradation``     — Per-embedding RMSE growth random → cold_drug → cold_protein.
* ``fig_supp_radar_per_dataset`` — Radar chart per dataset across kinetic types.
* ``fig_supp_v1_vs_v2_bar``      — v1 vs v2 RMSE bars on the 14 rerun rows (only if v2 results exist).

Skip the figures that depend on v2 results until the user has run the rerun
on HPC.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

K_ORDER = ["km", "kcat", "ki", "kd", "kcat_km"]
SPLIT_ORDER = ["random", "cold_drug", "cold_protein"]
EMBEDDING_ORDER = ["ESM2", "ESMv1", "MutaPLM", "ProteinCLIP"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clean_csv", default="data/benchmark/kinetic_regress_benchmark_clean.csv")
    p.add_argument("--v2_csv", default="data/benchmark/trace_kin_v2_results.csv")
    p.add_argument("--output_dir", default="figures/output")
    return p.parse_args()


def save(fig, name: str, output_dir: str) -> None:
    fig.savefig(os.path.join(output_dir, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"{name}.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Wrote {name}.pdf + {name}.png")


def fig_heatmap(df: pd.DataFrame, output_dir: str) -> None:
    test_only = df[df["Dataset"] == "test"].copy()
    fig, axes = plt.subplots(1, len(SPLIT_ORDER), figsize=(15, 4.5), sharey=True)
    for ax, split in zip(axes, SPLIT_ORDER):
        sub = test_only[test_only["split_mode"] == split]
        pivot = sub.groupby(["k_type", "embedding_type"])["RMSE"].min().unstack()
        pivot = pivot.reindex(index=K_ORDER, columns=EMBEDDING_ORDER)
        im = ax.imshow(pivot.values, cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(len(EMBEDDING_ORDER)))
        ax.set_xticklabels(EMBEDDING_ORDER, rotation=45, ha="right")
        ax.set_yticks(range(len(K_ORDER)))
        ax.set_yticklabels(K_ORDER)
        ax.set_title(split)
        for i, k in enumerate(K_ORDER):
            for j, e in enumerate(EMBEDDING_ORDER):
                v = pivot.loc[k, e] if (k in pivot.index and e in pivot.columns) else np.nan
                if pd.notna(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if v > pivot.values[~np.isnan(pivot.values)].mean() else "black",
                            fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle("Best test RMSE per (kinetic, embedding, split)")
    save(fig, "fig2b_heatmap_best_rmse", output_dir)


def fig_embedding_scatter(df: pd.DataFrame, output_dir: str) -> None:
    test_only = df[df["Dataset"] == "test"].copy()
    pivot = test_only.pivot_table(
        index=["Model", "k_type", "split_mode", "dataset_name"],
        columns="embedding_type",
        values="RMSE",
        aggfunc="min",
    ).reset_index()
    if "ESM2" not in pivot.columns or "MutaPLM" not in pivot.columns:
        print("WARN: skipping embedding scatter — ESM2 or MutaPLM column missing")
        return
    pivot = pivot.dropna(subset=["ESM2", "MutaPLM"])
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {"random": "#1f77b4", "cold_drug": "#d62728", "cold_protein": "#2ca02c"}
    for split, sub in pivot.groupby("split_mode"):
        ax.scatter(sub["ESM2"], sub["MutaPLM"], color=colors.get(split, "gray"),
                   alpha=0.7, label=split, s=30)
    lo = min(pivot["ESM2"].min(), pivot["MutaPLM"].min()) - 0.05
    hi = max(pivot["ESM2"].max(), pivot["MutaPLM"].max()) + 0.05
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("ESM2 RMSE")
    ax.set_ylabel("MutaPLM RMSE")
    ax.set_title("ESM2 vs MutaPLM (matched tasks)")
    ax.legend()
    save(fig, "fig2c_embedding_scatter", output_dir)


def fig_cold_degradation(df: pd.DataFrame, output_dir: str) -> None:
    test_only = df[df["Dataset"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    for emb in EMBEDDING_ORDER:
        means = []
        for split in SPLIT_ORDER:
            sub = test_only[(test_only["embedding_type"] == emb) & (test_only["split_mode"] == split)]
            means.append(sub["RMSE"].mean())
        ax.plot(SPLIT_ORDER, means, marker="o", label=emb)
    ax.set_xlabel("Split")
    ax.set_ylabel("Mean RMSE across all (model, k_type, dataset)")
    ax.set_title("Cold-split RMSE degradation per embedding")
    ax.legend()
    save(fig, "fig2d_cold_degradation", output_dir)


def fig_radar(df: pd.DataFrame, output_dir: str) -> None:
    test_only = df[df["Dataset"] == "test"].copy()
    datasets = sorted(test_only["dataset_name"].dropna().unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4),
                              subplot_kw={"projection": "polar"})
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        sub = test_only[test_only["dataset_name"] == ds]
        means = [sub[sub["k_type"] == k]["RMSE"].min() if not sub[sub["k_type"] == k].empty else np.nan
                 for k in K_ORDER]
        means = [m if pd.notna(m) else 0 for m in means]
        angles = np.linspace(0, 2 * np.pi, len(K_ORDER), endpoint=False).tolist()
        means.append(means[0])
        angles.append(angles[0])
        ax.plot(angles, means, "o-")
        ax.fill(angles, means, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(K_ORDER)
        ax.set_title(ds)
    save(fig, "fig_supp_radar_per_dataset", output_dir)


def fig_v1_vs_v2(v2_csv: str, output_dir: str) -> None:
    if not os.path.exists(v2_csv):
        print(f"INFO: {v2_csv} not found — skipping v1-vs-v2 figure (run rerun first)")
        return
    df = pd.read_csv(v2_csv)
    df = df.dropna(subset=["v1_rmse", "v2_rmse"])
    if df.empty:
        print("INFO: no v1/v2 paired rows — skipping figure")
        return
    fig, ax = plt.subplots(figsize=(10, 4 + 0.2 * len(df)))
    y = np.arange(len(df))
    ax.barh(y - 0.2, df["v1_rmse"], height=0.4, color="#888", label="v1 RMSE")
    ax.barh(y + 0.2, df["v2_rmse"], height=0.4, color="#1f77b4", label="v2 RMSE")
    if "rf_best_rmse" in df.columns:
        ax.scatter(df["rf_best_rmse"], y, color="red", marker="x", s=80, label="RF best")
    ax.set_yticks(y)
    ax.set_yticklabels(df["dataset_folder"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE")
    ax.set_title("v1 vs v2 vs RF best on rerun datasets")
    ax.legend()
    save(fig, "fig_supp_v1_vs_v2_bar", output_dir)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.clean_csv)

    fig_heatmap(df, args.output_dir)
    fig_embedding_scatter(df, args.output_dir)
    fig_cold_degradation(df, args.output_dir)
    fig_radar(df, args.output_dir)
    fig_v1_vs_v2(args.v2_csv, args.output_dir)

    print(f"All figures written to {args.output_dir}")


if __name__ == "__main__":
    main()

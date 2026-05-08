#!/usr/bin/env python
"""Human-readable improvement summary for the v3 rerun.

Reads ``data/benchmark/trace_kin_v3_results.csv`` (produced by
``analysis/aggregate_results.py``) and produces a single concise report
answering "did the v3 redesign actually improve over v1, and does it close
the RF gap?" Results go to stdout and to
``data/benchmark/improvement_summary.md``.

Sections of the report:

* **Per-task table** — one row per (dataset, kinetic, split) with v1 RMSE,
  v3 RMSE, RF best, Δv1 = v3−v1, Δrf = v3−rf, and a per-row verdict
  (``beats RF``, ``ties RF``, ``worse than RF``, ``worse than v1``).
* **Per-kinetic aggregate** — mean Δv1, mean Δrf, # splits where v3 ≤ RF,
  per-kinetic verdict.
* **Headline summary** — one paragraph fit for status updates.
* **Exit code** — 0 if the catalytic gate passes (≥3 of 4 kinetics PASS),
  1 otherwise. Mirrors ``promotion_gate.py`` so this script can stand in for
  CI checks.

Compared to ``promotion_gate.py``, this script is **descriptive** (always
prints the full table) rather than **prescriptive** (just verdict). Use both
in series: ``check_improvement.py`` for the human review,
``promotion_gate.py`` for the binding decision.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd


CATALYTIC_KINETICS = ["kcat", "km", "kd", "kcat_km"]
TIE_TOLERANCE = 0.005  # |Δrf| ≤ 0.005 RMSE counts as a tie


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--v3_csv", default="data/benchmark/trace_kin_v3_results.csv")
    p.add_argument("--output", default="data/benchmark/improvement_summary.md")
    return p.parse_args()


def row_verdict(v1: float | None, v3: float | None, rf: float | None) -> str:
    if v3 is None or pd.isna(v3):
        return "no v3 result"
    if v1 is not None and not pd.isna(v1) and v3 > v1:
        return "✗ worse than v1"
    if rf is None or pd.isna(rf):
        return "no RF reference"
    if v3 <= rf:
        return "✓ beats RF"
    if v3 - rf <= TIE_TOLERANCE:
        return "~ ties RF"
    return "✗ worse than RF"


def per_kinetic_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in sorted(df["k_type"].dropna().unique()):
        sub = df[df["k_type"] == k]
        valid_v1 = sub.dropna(subset=["v1_rmse", "v3_rmse"])
        valid_rf = sub.dropna(subset=["v3_rmse", "rf_best_rmse"])
        beats_rf = (valid_rf["v3_rmse"] <= valid_rf["rf_best_rmse"]).sum()
        n_splits = len(valid_rf)
        # Verdict: same logic as promotion_gate.py — ≥2/3 splits beat RF, OR
        # the per-kinetic mean v3 ≤ mean RF.
        by_count = (n_splits >= 3 and beats_rf >= 2) or (n_splits < 3 and beats_rf == n_splits and n_splits > 0)
        by_mean = (valid_rf["v3_rmse"].mean() <= valid_rf["rf_best_rmse"].mean()) if not valid_rf.empty else False
        passed = bool(by_count or by_mean)
        rows.append({
            "k_type": k,
            "n_splits": int(n_splits),
            "beats_rf": int(beats_rf),
            "mean_v1": round(valid_v1["v1_rmse"].mean(), 4) if not valid_v1.empty else None,
            "mean_v3": round(valid_rf["v3_rmse"].mean(), 4) if not valid_rf.empty else None,
            "mean_rf": round(valid_rf["rf_best_rmse"].mean(), 4) if not valid_rf.empty else None,
            "mean_delta_v1": round(valid_v1["v3_minus_v1"].mean(), 4) if not valid_v1.empty else None,
            "mean_delta_rf": round(valid_rf["v3_minus_rf"].mean(), 4) if not valid_rf.empty else None,
            "verdict": "PASS" if passed else "FAIL",
        })
    return pd.DataFrame(rows)


def render_md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append("" if pd.isna(v) else f"{v:.4f}")
            else:
                cells.append("" if pd.isna(v) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def headline(df: pd.DataFrame, kinetic_summary: pd.DataFrame) -> str:
    catalytic = kinetic_summary[kinetic_summary["k_type"].isin(CATALYTIC_KINETICS)]
    n_pass = int((catalytic["verdict"] == "PASS").sum())
    n_total_cat = int(len(catalytic))
    valid = df.dropna(subset=["v1_rmse", "v3_rmse"])
    mean_v1_minus_v3 = (valid["v1_rmse"] - valid["v3_rmse"]).mean() if not valid.empty else 0.0
    valid_rf = df.dropna(subset=["v3_rmse", "rf_best_rmse"])
    n_beats_rf = int((valid_rf["v3_rmse"] <= valid_rf["rf_best_rmse"]).sum())
    n_total_rf = int(len(valid_rf))
    ki_rows = df[df["k_type"] == "ki"].dropna(subset=["v1_rmse", "v3_rmse"])
    ki_max_regress = (ki_rows["v3_rmse"] - ki_rows["v1_rmse"]).max() if not ki_rows.empty else 0.0
    return (
        f"v3 improves over v1 by mean {mean_v1_minus_v3:+.4f} RMSE across "
        f"{len(valid)} paired rows. v3 beats or ties RF on {n_beats_rf}/{n_total_rf} "
        f"reran rows. Catalytic gate: {n_pass}/{n_total_cat} kinetics pass. "
        f"Ki preservation: largest v3−v1 regression = {ki_max_regress:+.4f}."
    )


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.v3_csv):
        print(f"ERROR: {args.v3_csv} not found. Run analysis/aggregate_results.py first.")
        return 2
    df = pd.read_csv(args.v3_csv)
    if df.empty:
        print("ERROR: v3 results CSV is empty.")
        return 2

    df = df.copy()
    df["verdict"] = df.apply(
        lambda r: row_verdict(r.get("v1_rmse"), r.get("v3_rmse"), r.get("rf_best_rmse")),
        axis=1,
    )

    per_task_cols = ["dataset_folder", "k_type", "split_mode", "embedding",
                     "v1_rmse", "v3_rmse", "rf_best_rmse",
                     "v3_minus_v1", "v3_minus_rf", "verdict"]
    per_task = df[[c for c in per_task_cols if c in df.columns]].copy()

    kinetic_summary = per_kinetic_summary(df)
    head = headline(df, kinetic_summary)

    md = [
        "# TRACE-Kin v3 improvement summary", "",
        f"Source: `{args.v3_csv}` ({len(df)} rows)", "",
        "## Headline", "",
        head, "",
        "## Per-task results", "",
        render_md_table(per_task), "",
        "## Per-kinetic aggregate", "",
        render_md_table(kinetic_summary), "",
        "## Legend", "",
        "- `✓ beats RF` — v3 RMSE ≤ RF best RMSE",
        f"- `~ ties RF` — v3 within {TIE_TOLERANCE} RMSE of RF best (above)",
        "- `✗ worse than RF` — v3 RMSE > RF best (and not within tie tolerance)",
        "- `✗ worse than v1` — v3 RMSE > v1 RMSE (regression vs. baseline)",
        "",
        "Per-kinetic verdict mirrors `analysis/promotion_gate.py`: PASS if ≥2/3 "
        "splits beat RF or the per-kinetic mean v3 RMSE ≤ mean RF RMSE.",
        "",
    ]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote {args.output}")
    print()
    print(head)
    print()
    print(render_md_table(kinetic_summary))

    catalytic_pass = int((kinetic_summary[kinetic_summary["k_type"].isin(CATALYTIC_KINETICS)]["verdict"] == "PASS").sum())
    return 0 if catalytic_pass >= 3 else 1


if __name__ == "__main__":
    sys.exit(main())

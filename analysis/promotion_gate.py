#!/usr/bin/env python
"""Decide whether to promote v2 results into the paper.

Gate (mechanical, defined in the project plan):

* For each catalytic kinetic in {kcat, km, kd, kcat_km}, count how many of its
  splits have v2 RMSE <= RF best RMSE. PASS the kinetic if 2 of 3 splits pass,
  OR if the per-kinetic mean v2 RMSE <= per-kinetic mean RF best.
* Overall gate PASSES if at least 3 of the 4 catalytic kinetics PASS.
* Separately: each Ki rerun must have v2 RMSE within +0.02 of v1 RMSE
  (preservation guard). Failures here mean we should ship v1 weights for Ki
  even if the catalytic gate passes.

Output: ``data/benchmark/promotion_gate_decision.md`` with a per-task
breakdown and an overall PASS/FAIL verdict.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

CATALYTIC_KINETICS = ["kcat", "km", "kd", "kcat_km"]
KI_REGRESSION_TOLERANCE = 0.02  # v2 may not be more than 0.02 RMSE worse than v1 on Ki tasks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--v2_csv", default="data/benchmark/trace_kin_v2_results.csv")
    p.add_argument("--output", default="data/benchmark/promotion_gate_decision.md")
    return p.parse_args()


def evaluate_kinetic(kinetic_df: pd.DataFrame) -> tuple[bool, str]:
    """Return (passed, reasoning)."""
    valid = kinetic_df.dropna(subset=["v2_rmse", "rf_best_rmse"])
    if valid.empty:
        return False, "no v2 results vs RF available"
    valid = valid.copy()
    valid["beats_rf"] = valid["v2_rmse"] <= valid["rf_best_rmse"]
    n_total = len(valid)
    n_beats = int(valid["beats_rf"].sum())
    mean_v2 = valid["v2_rmse"].mean()
    mean_rf = valid["rf_best_rmse"].mean()
    by_count = (n_total >= 3 and n_beats >= 2) or (n_total < 3 and n_beats == n_total)
    by_mean = mean_v2 <= mean_rf
    passed = by_count or by_mean
    reasoning = (
        f"{n_beats}/{n_total} splits beat RF; "
        f"mean v2={mean_v2:.4f} vs mean RF={mean_rf:.4f}. "
        f"by_count_pass={by_count}, by_mean_pass={by_mean}"
    )
    return passed, reasoning


def evaluate_ki_guards(ki_df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Ki preservation guard: v2 within +0.02 of v1 on each Ki rerun row."""
    if ki_df.empty:
        return True, ["no Ki rows in v2 results — guard not exercised"]
    notes = []
    all_passed = True
    for _, row in ki_df.iterrows():
        if pd.isna(row["v2_rmse"]) or pd.isna(row["v1_rmse"]):
            notes.append(f"  - {row['dataset_folder']}: missing v2 or v1 (skipped)")
            continue
        delta = row["v2_rmse"] - row["v1_rmse"]
        ok = delta <= KI_REGRESSION_TOLERANCE
        notes.append(
            f"  - {row['dataset_folder']}: v2={row['v2_rmse']:.4f}, "
            f"v1={row['v1_rmse']:.4f}, Δ={delta:+.4f} ({'OK' if ok else 'GUARD FAIL'})"
        )
        all_passed &= ok
    return all_passed, notes


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.v2_csv):
        raise FileNotFoundError(f"v2 results not found: {args.v2_csv}")
    df = pd.read_csv(args.v2_csv)

    lines = ["# TRACE-Kin v2 Promotion Gate Decision", "",
             f"Source: `{args.v2_csv}` ({len(df)} v2 result rows)", ""]

    # Catalytic gate (3 of 4 kinetics).
    n_kinetics_pass = 0
    lines.append("## Catalytic kinetics gate")
    for k in CATALYTIC_KINETICS:
        sub = df[df["k_type"] == k]
        passed, reason = evaluate_kinetic(sub)
        verdict = "PASS" if passed else "FAIL"
        lines.append(f"### `{k}`: {verdict}")
        lines.append(f"- {reason}")
        if not sub.empty:
            for _, row in sub.iterrows():
                gap = (row["v2_rmse"] - row["rf_best_rmse"]) if pd.notna(row["rf_best_rmse"]) else None
                lines.append(
                    f"  - {row['dataset_folder']}: v2={row['v2_rmse']!s}, "
                    f"RF={row['rf_best_rmse']!s}, gap={gap if gap is None else f'{gap:+.4f}'}"
                )
        lines.append("")
        if passed:
            n_kinetics_pass += 1

    catalytic_gate_pass = n_kinetics_pass >= 3
    lines.append(f"**Catalytic gate: {n_kinetics_pass}/{len(CATALYTIC_KINETICS)} kinetics passed → "
                 f"{'PASS' if catalytic_gate_pass else 'FAIL'}**")
    lines.append("")

    # Ki preservation guard.
    ki_df = df[df["k_type"] == "ki"]
    ki_passed, ki_notes = evaluate_ki_guards(ki_df)
    lines.append("## Ki preservation guard")
    lines.append(f"Tolerance: v2 may not be more than +{KI_REGRESSION_TOLERANCE:.2f} RMSE worse than v1 on Ki tasks.")
    lines.extend(ki_notes)
    lines.append(f"**Ki guard: {'PASS' if ki_passed else 'FAIL'}**")
    lines.append("")

    # Overall.
    overall_pass = catalytic_gate_pass and ki_passed
    lines.append("---")
    lines.append(f"## Overall verdict: **{'PASS' if overall_pass else 'FAIL'}**")
    if overall_pass:
        lines.append("- Proceed to Phase 8 packaging deliverables (figures, tables) using v2 weights.")
    else:
        lines.append("- Do NOT generate paper figures/tables yet.")
        if not catalytic_gate_pass:
            lines.append("- Catalytic gate failed: review per-kinetic breakdown above and iterate v2 design.")
        if not ki_passed:
            lines.append("- Ki guard failed: ship v1 weights for Ki tasks; v2 weights for everything else if catalytic gate passed.")
    lines.append("")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {args.output}")
    print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")


if __name__ == "__main__":
    main()

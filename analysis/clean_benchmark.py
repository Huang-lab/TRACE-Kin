#!/usr/bin/env python
"""Normalize the raw 1,847-row benchmark CSV.

Reads ``trace_doc/kinetic_regress_benchmark.csv`` (immutable). Writes
``data/benchmark/kinetic_regress_benchmark_clean.csv`` plus a
``normalization_log.md`` describing every distinct value that was renamed and
every row that was dropped.

The cleanup rules are derived from the patterns seen in the raw distinct-value
distributions: leading/trailing whitespace, casing inconsistencies
(``ESM1V/ESM1v/ESMv1``), space-vs-underscore splits, and synonyms
(``in house`` vs ``inhouse``). After cleaning, downstream tables and figures
can group by canonical names without ambiguity.

Usage::

    python analysis/clean_benchmark.py \\
        --input trace_doc/kinetic_regress_benchmark.csv \\
        --output data/benchmark/kinetic_regress_benchmark_clean.csv \\
        --log data/benchmark/normalization_log.md
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

import pandas as pd


# Canonical mappings. RHS is the post-clean value; LHS is what the raw CSV may contain.
KINETIC_TYPE_MAP = {
    "kcat": "kcat",
    "Kcat": "kcat",
    "KCAT": "kcat",
    "km": "km",
    "Km": "km",
    "ki": "ki",
    "Ki": "ki",
    "kd": "kd",
    "Kd": "kd",
    "kkm": "kcat_km",
    "KKM": "kcat_km",
    "kcat_km": "kcat_km",
}

SPLIT_MODE_MAP = {
    "random": "random",
    "embedding_random": "random",
    "cold_drug": "cold_drug",
    "cold_protein": "cold_protein",
}

EMBEDDING_MAP = {
    "ESM2": "ESM2",
    "ESMv1": "ESMv1",
    "ESM1V": "ESMv1",
    "ESM1v": "ESMv1",
    "MutaPLM": "MutaPLM",
    "MUTAPLM": "MutaPLM",
    "ProteinCLIP": "ProteinCLIP",
    "Protein_CLIP": "ProteinCLIP",
}

DATASET_NAME_MAP = {
    "MPEK": "MPEK",
    "EITLEM": "EITLEM",
    "catpred": "catpred",
    "CatPred": "catpred",
    "inhouse": "inhouse",
    "in_house": "inhouse",
    "in house": "inhouse",
}

DATASET_SPLIT_MAP = {
    "Test": "test",
    "test": "test",
    "Validation": "validation",
    "validation": "validation",
}


def normalize_value(raw, mapping: dict, normalize_spaces: bool = True) -> str | None:
    """Trim, optionally collapse spaces->underscores, then look up canonical form."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if normalize_spaces:
        s_underscored = s.replace(" ", "_")
    else:
        s_underscored = s
    # Try a few variants in order: exact, underscore-collapsed, raw-trimmed.
    for candidate in (s, s_underscored, s.lower(), s_underscored.lower()):
        if candidate in mapping:
            return mapping[candidate]
    # No match — return the trimmed form so the dropping step can flag it.
    return s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input", default="trace_doc/kinetic_regress_benchmark.csv")
    p.add_argument("--output", default="data/benchmark/kinetic_regress_benchmark_clean.csv")
    p.add_argument("--log", default="data/benchmark/normalization_log.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    n_raw = len(df)
    print(f"Loaded raw benchmark: {n_raw} rows × {len(df.columns)} columns")

    rename_log: dict[str, dict[str, str]] = defaultdict(dict)

    def remap(col: str, mapping: dict, normalize_spaces: bool = True):
        before = df[col].astype(object)
        after = before.apply(lambda v: normalize_value(v, mapping, normalize_spaces))
        for orig, mapped in zip(before, after):
            if orig != mapped and pd.notna(orig):
                rename_log[col][str(orig)] = str(mapped) if mapped is not None else "<dropped>"
        df[col] = after

    remap("k_type", KINETIC_TYPE_MAP, normalize_spaces=False)
    remap("split_mode", SPLIT_MODE_MAP, normalize_spaces=True)
    remap("embedding_type", EMBEDDING_MAP, normalize_spaces=False)
    remap("dataset_name", DATASET_NAME_MAP, normalize_spaces=True)
    remap("Dataset", DATASET_SPLIT_MAP, normalize_spaces=False)
    df["Model"] = df["Model"].astype(str).str.strip().replace("", pd.NA)

    # Drop rows that lack a canonical value in any required column.
    required = ["k_type", "split_mode", "embedding_type", "dataset_name", "Dataset", "Model"]
    valid_mask = pd.Series(True, index=df.index)
    drop_reasons: dict[str, int] = defaultdict(int)
    for col in required:
        canonical_set = (
            set(KINETIC_TYPE_MAP.values()) if col == "k_type"
            else set(SPLIT_MODE_MAP.values()) if col == "split_mode"
            else set(EMBEDDING_MAP.values()) if col == "embedding_type"
            else set(DATASET_NAME_MAP.values()) if col == "dataset_name"
            else set(DATASET_SPLIT_MAP.values()) if col == "Dataset"
            else None
        )
        if canonical_set is None:
            null_mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
        else:
            null_mask = ~df[col].isin(canonical_set)
        n_drop = int(null_mask.sum())
        if n_drop:
            drop_reasons[col] = n_drop
        valid_mask &= ~null_mask
    df = df[valid_mask].copy()

    # Coerce numeric columns; drop rows where RMSE failed parsing (those are unusable).
    for col in ("RMSE", "MAE", "MSE", "R2", "Pearson", "Median_AE", "Explained_Variance"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    rmse_nan = int(df["RMSE"].isna().sum())
    if rmse_nan:
        df = df.dropna(subset=["RMSE"]).copy()

    # Drop exact duplicates on the canonical key columns.
    n_before_dedup = len(df)
    df = df.drop_duplicates(
        subset=["dataset_name", "k_type", "split_mode", "Model",
                "Dataset", "embedding_type", "folder", "RMSE"]
    )
    n_dedup = n_before_dedup - len(df)

    n_clean = len(df)
    print(f"Cleaned benchmark: {n_clean} rows ({n_raw - n_clean} dropped: "
          f"{sum(drop_reasons.values())} required-field, {rmse_nan} RMSE-NaN, {n_dedup} duplicate)")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")

    # ---- normalization log ----
    log_lines = ["# Benchmark CSV normalization log", "",
                 f"- Input: `{args.input}` ({n_raw} rows)",
                 f"- Output: `{args.output}` ({n_clean} rows)",
                 f"- Dropped: {n_raw - n_clean} ({sum(drop_reasons.values())} required-field, "
                 f"{rmse_nan} RMSE-NaN, {n_dedup} duplicate)",
                 ""]
    log_lines.append("## Renames applied")
    for col, mapping in sorted(rename_log.items()):
        if not mapping:
            continue
        log_lines.append(f"### `{col}`")
        for orig, new in sorted(mapping.items()):
            log_lines.append(f"- `{orig!r}` → `{new!r}`")
        log_lines.append("")
    if drop_reasons:
        log_lines.append("## Required-field drop counts")
        for col, n in sorted(drop_reasons.items(), key=lambda kv: -kv[1]):
            log_lines.append(f"- `{col}`: {n} rows lacked a canonical value")
        log_lines.append("")
    log_lines.append("## Distinct values after cleanup")
    for col in required:
        vals = sorted(df[col].dropna().unique())
        log_lines.append(f"- `{col}`: {vals}")
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    with open(args.log, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"Wrote {args.log}")


if __name__ == "__main__":
    main()

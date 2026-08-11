#!/usr/bin/env python
"""Ensemble inference for TRACE-Kin — average K checkpoints, report uncertainty.

Mirrors the CatPred pattern (maranasgroup/CatPred, `catpred/train/make_predictions.py`):
enumerate checkpoint paths, run every member over the same data, then emit the
mean prediction plus an uncertainty column and the individual member predictions.

    preds_mean  = mean over members          <- the ensemble prediction
    preds_std   = std  over members          <- EPISTEMIC uncertainty (model disagreement)
    pred_model_i                             <- each member, kept for auditing

Difference from CatPred worth knowing: CatPred's head is probabilistic (it emits a
variance per prediction), so it can decompose uncertainty into *aleatoric* (data
noise) + *epistemic* (ensemble spread). TRACE-Kin's regression head emits a bare
scalar, so only the epistemic term is available here. Adding an MVE head
(2 outputs: mean, log-variance, trained with Gaussian NLL) would recover the
aleatoric half — see the note at the bottom of this docstring.

Efficiency: the protein/ligand graph caches and the DataLoader are built ONCE and
reused across every member, so a K-model ensemble costs K forward passes, not K
full preprocessing runs.

Usage
-----
    python inference/predict_ensemble.py \
        --weights_glob 'results/kcat_v7/save_model_seed*' \
        --datafolder data/kcat_v7 --split test \
        --output results/kcat_v7/ensemble_test.csv \
        --protein_col sequence --ligand_col smiles --label_col log10_value

or list members explicitly with --weights_dirs A B C.

Adding an MVE head later (sketch): make `reg_heads` output 2 values, train with
`0.5*(log s2 + (y-mu)^2/s2)`, then total variance = mean(s2) + var(mu) — the
standard aleatoric + epistemic decomposition CatPred reports.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.loader import DataLoader

from training.dataset import ProteinMoleculeDataset
from training.data_utils import virtual_screening
from training.train_trace_kin import (build_model, find_data_file, load_dataframe,
                                      normalize_columns)
from training.metrics import evaluate_reg

PRED_COL = "predicted_binding_affinity"
LABEL_COL = "regression_label"


def parse_args():
    p = argparse.ArgumentParser(description="Ensemble inference for TRACE-Kin (v1/v4/v5/v5t/v6c/v7).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--weights_dirs", nargs="+",
                     help="Ensemble members: one or more save_model_seed<N>/ directories.")
    src.add_argument("--weights_glob",
                     help="Glob matching member directories, e.g. 'results/run/save_model_seed*'.")

    p.add_argument("--datafolder", required=True,
                   help="Dataset folder holding protein.pt / ligand.pt (and the split file).")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--input_file", default="",
                   help="Score this table instead of a split.")
    p.add_argument("--output", required=True, help="Destination CSV.")

    p.add_argument("--protein_col", default="Protein")
    p.add_argument("--ligand_col", default="Ligand")
    p.add_argument("--label_col", default=LABEL_COL)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def resolve_members(args) -> list[str]:
    dirs = args.weights_dirs or sorted(glob.glob(args.weights_glob))
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        raise FileNotFoundError("no ensemble member directories matched.")
    for d in dirs:
        for f in ("config.json", "degree.pt", "model.pt"):
            if not os.path.exists(os.path.join(d, f)):
                raise FileNotFoundError(f"{d} is missing {f} — not a valid member directory.")
    return dirs


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available()
                          or not str(args.device).startswith("cuda") else "cpu")
    if str(device) != args.device:
        print(f"WARN: CUDA unavailable; using {device}")

    members = resolve_members(args)
    print(f"Ensemble members ({len(members)}):")
    for d in members:
        print(f"  {d}")
    if len(members) == 1:
        print("WARN: single member — this is a plain prediction run, no uncertainty signal.")

    # ---------------- caches, built once and shared by every member ----------
    protein_path = os.path.join(args.datafolder, "protein.pt")
    ligand_path = os.path.join(args.datafolder, "ligand.pt")
    for path in (protein_path, ligand_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing {path}; run preprocessing for this datafolder first.")
    print(f"Loading caches from {args.datafolder} ...")
    protein_dict = torch.load(protein_path)
    ligand_dict = torch.load(ligand_path)
    print(f"  proteins={len(protein_dict)} ligands={len(ligand_dict)}")

    # ---------------- rows to score ----------------
    table_path = args.input_file or find_data_file(args.datafolder, args.split)
    if table_path is None:
        raise FileNotFoundError(f"no {args.split}.[parquet|csv|tsv] in {args.datafolder}")
    print(f"Scoring: {table_path}")

    df = load_dataframe(table_path, columns=None)
    df = normalize_columns(df, args.protein_col, args.ligand_col, args.label_col)
    before = len(df)
    valid = df["Ligand"].isin(ligand_dict) & df["Protein"].isin(protein_dict)
    if (~valid).any():
        print(f"WARN: dropping {int((~valid).sum())} of {before} rows absent from the caches.")
    df = df[valid].reset_index(drop=True)
    if df.empty:
        raise ValueError("no rows left to score after cache filtering.")
    print(f"  {len(df)} rows")

    # DataLoader is deterministic (shuffle=False) so member predictions stay row-aligned.
    ds = ProteinMoleculeDataset(df, ligand_dict, protein_dict, device=device)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        follow_batch=["mol_x", "clique_x", "prot_node_aa"])

    # ---------------- run every member ----------------
    member_preds: list[np.ndarray] = []
    member_rmse: list[float | None] = []
    ref_version = None
    ref_dim = None

    for idx, wdir in enumerate(members):
        with open(os.path.join(wdir, "config.json")) as f:
            model_config = json.load(f)
        version = model_config.get("model_version", "v1")
        dim = model_config["params"]["prot_evo_channels"]

        # Members must be the same architecture over the same features, or the
        # average is meaningless. Fail loudly rather than silently mixing.
        if ref_version is None:
            ref_version, ref_dim = version, dim
        elif (version, dim) != (ref_version, ref_dim):
            raise ValueError(
                f"member {wdir} is {version}/prot_evo_channels={dim} but the first member "
                f"is {ref_version}/{ref_dim}. Ensemble members must share an architecture."
            )

        degree_dict = torch.load(os.path.join(wdir, "degree.pt"), map_location="cpu")

        aa_typical_buffers = None
        if version == "v4":
            from training.aa_typical import load_or_compute_aa_typical, aa_typical_to_buffers
            aa_typical = load_or_compute_aa_typical(
                cache_path=os.path.join(args.datafolder, "aa_typical.pt"),
                protein_dict=protein_dict, force_rebuild=False)
            _, aa_means, aa_stds = aa_typical_to_buffers(aa_typical)
            aa_typical_buffers = {"aa_means": aa_means, "aa_stds": aa_stds}

        model, _ = build_model(model_config, degree_dict["ligand_deg"],
                               degree_dict["protein_deg"], str(device), aa_typical_buffers)
        model.load_state_dict(torch.load(os.path.join(wdir, "model.pt"), map_location=device),
                              strict=True)
        model.eval()

        # virtual_screening MUTATES the frame it is given (adds ID + prediction
        # columns), so hand each member its own copy or later members overwrite
        # earlier predictions in place.
        out = virtual_screening(df.copy(), model, loader,
                                result_path=args.datafolder,
                                save_interpret=False,
                                ligand_dict=ligand_dict, device=device)
        preds = out[PRED_COL].values.astype(float)
        member_preds.append(preds)

        rmse = None
        if LABEL_COL in out:
            y = out[LABEL_COL].values.astype(float)
            m = ~np.isnan(y) & ~np.isnan(preds)
            if m.any():
                rmse = float(evaluate_reg(y[m], preds[m])["rmse"])
        member_rmse.append(rmse)
        print(f"  [{idx}] {os.path.basename(wdir)}: rmse="
              f"{'n/a' if rmse is None else f'{rmse:.4f}'}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------- aggregate ----------------
    P = np.vstack(member_preds)                    # (M, n)
    pred_mean = P.mean(axis=0)
    # ddof=1 is the unbiased estimator across ensemble members; undefined for M=1.
    pred_std = P.std(axis=0, ddof=1) if P.shape[0] > 1 else np.zeros_like(pred_mean)

    result = pd.DataFrame({
        "Protein": df["Protein"].values,
        "Ligand": df["Ligand"].values,
        "prediction": pred_mean,
        "epistemic_std": pred_std,
    })
    if LABEL_COL in df:
        result[LABEL_COL] = df[LABEL_COL].values
    for i in range(P.shape[0]):
        result[f"pred_model_{i}"] = P[i]

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"\nWrote {len(result)} rows -> {args.output}")

    # ---------------- report ----------------
    if LABEL_COL in result:
        y = result[LABEL_COL].values.astype(float)
        m = ~np.isnan(y) & ~np.isnan(pred_mean)
        if m.any():
            ens = evaluate_reg(y[m], pred_mean[m])
            singles = [r for r in member_rmse if r is not None]
            print(f"\nEnsemble ({P.shape[0]} members) on {int(m.sum())} labelled rows:")
            print(f"  rmse={ens['rmse']:.4f}  pearson={ens['pearson']:.4f}  "
                  f"spearman={ens['spearman']:.4f}  mae={ens['mae']:.4f}")
            if singles:
                print(f"  single-model rmse: mean={np.mean(singles):.4f} "
                      f"best={min(singles):.4f} worst={max(singles):.4f}")
                print(f"  ensemble gain vs mean single model: "
                      f"{np.mean(singles) - ens['rmse']:+.4f} rmse")

            # Calibration: does disagreement track error? This is the property
            # CatPred reports — accuracy correlates with lower predicted variance.
            if P.shape[0] > 1:
                from scipy.stats import spearmanr
                abs_err = np.abs(pred_mean[m] - y[m])
                rho, pval = spearmanr(pred_std[m], abs_err)
                print(f"  uncertainty calibration: spearman(epistemic_std, |error|)="
                      f"{rho:.3f} (p={pval:.2g})")
                print("    positive rho => high-disagreement predictions are the wrong ones,")
                print("    i.e. epistemic_std is usable for triaging predictions.")
        else:
            print("No labelled rows — predictions written, metrics skipped.")


if __name__ == "__main__":
    main()

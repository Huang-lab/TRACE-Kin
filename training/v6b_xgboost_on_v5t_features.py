#!/usr/bin/env python
"""TRACE-Kin v6b: XGBoost on v5t deep features (Option B).

Extracts the 1024-d `interaction_fingerprint` from a trained v5t model
and trains XGBoost regression on top. This is a quick parallel experiment
to test whether a gradient-boosted tree on deep features can close the
gap to Random Forest on raw embeddings.

Usage:
    python training/v6b_xgboost_on_v5t_features.py \
        --v5t_checkpoint <path_to_best_model.pt> \
        --datafolder <dataset_folder> \
        --config_path training/config_v5t.json \
        --molformer_path <path_to_molformer> \
        --result_path <output_dir> \
        --seed 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from training.data_utils import DataLoader, compute_pna_degrees
from training.dataset import ProteinMoleculeDataset
from training.ligand_init import ligand_init
from training.molformer_init import compute_molformer_embeddings
from training.protein_init_with_embedding import protein_init_with_embedding
from models.trace_kin import TraceKinV5T


def parse_args():
    parser = argparse.ArgumentParser(description="v6b: XGBoost on v5t features")
    parser.add_argument("--v5t_checkpoint", type=str, required=True,
                        help="Path to trained v5t model checkpoint (best_model.pt)")
    parser.add_argument("--datafolder", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="training/config_v5t.json")
    parser.add_argument("--molformer_path", type=str, default="")
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--protein_col", type=str, default="Sequence")
    parser.add_argument("--feature_col", type=str, default="protein_features")
    parser.add_argument("--ligand_col", type=str, default="Smiles")
    parser.add_argument("--label_col", type=str, default="Label")
    parser.add_argument("--metabolite_feature_col", type=str, default="metabolite_features")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--use_lgb", action="store_true", default=False,
                        help="Use LightGBM instead of XGBoost")
    return parser.parse_args()


def load_data(args, config):
    """Load train/test datasets using the same pipeline as train_trace_kin.py."""
    datafolder = args.datafolder

    train_path = os.path.join(datafolder, "train.parquet")
    test_path = os.path.join(datafolder, "test.parquet")
    if not os.path.exists(train_path):
        train_path = os.path.join(datafolder, "train.csv")
        test_path = os.path.join(datafolder, "test.csv")

    train_df = pd.read_parquet(train_path) if train_path.endswith(".parquet") else pd.read_csv(train_path)
    test_df = pd.read_parquet(test_path) if test_path.endswith(".parquet") else pd.read_csv(test_path)

    return train_df, test_df


@torch.no_grad()
def extract_features(model, dataloader, device):
    """Run forward pass and collect interaction_fingerprint + labels."""
    model.eval()
    all_feats = []
    all_labels = []

    for batch in dataloader:
        batch = batch.to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):
            reg_pred, cls_pred, mcls_pred, *_, attention_dict = model(
                batch.mol_x, batch.mol_x_feat, batch.bond_x, batch.atom_edge_index,
                batch.clique_x, batch.clique_edge_index, batch.atom2clique_index,
                batch.residue_x, batch.prot_node_evo, batch.prot_edge_index, batch.prot_edge_weight,
                mol_batch=batch.mol_x_batch, prot_batch=batch.residue_x_batch,
                clique_batch=batch.clique_x_batch,
                chembert_fp=getattr(batch, 'chembert_fp', None),
                prot_aa_idx=getattr(batch, 'prot_aa_idx', None),
            )

        feat = attention_dict['interaction_fingerprint']
        all_feats.append(feat.float().cpu().numpy())

        if hasattr(batch, 'reg_label'):
            all_labels.append(batch.reg_label.cpu().numpy())
        elif hasattr(batch, 'y'):
            all_labels.append(batch.y.cpu().numpy())

    X = np.concatenate(all_feats, axis=0)
    y = np.concatenate(all_labels, axis=0).ravel()
    return X, y


def train_xgboost(X_train, y_train, X_test, y_test, seed=1):
    """Train XGBoost regressor and return predictions."""
    if not HAS_XGB:
        raise ImportError("xgboost not installed. Run: pip install xgboost")

    params = {
        'objective': 'reg:squarederror',
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': seed,
        'n_jobs': -1,
        'tree_method': 'hist',
    }

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    return train_pred, test_pred, model


def train_lightgbm(X_train, y_train, X_test, y_test, seed=1):
    """Train LightGBM regressor and return predictions."""
    if not HAS_LGB:
        raise ImportError("lightgbm not installed. Run: pip install lightgbm")

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': seed,
        'n_jobs': -1,
        'verbose': -1,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
    )

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    return train_pred, test_pred, model


def evaluate(y_true, y_pred, split_name="test"):
    """Compute and print regression metrics."""
    pearson_r, _ = pearsonr(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"  {split_name.upper()} Results")
    print(f"{'='*50}")
    print(f"  Pearson R:  {pearson_r:.4f}")
    print(f"  RMSE:       {rmse:.4f}")
    print(f"  R²:         {r2:.4f}")
    print(f"  MAE:        {mae:.4f}")
    print(f"{'='*50}\n")

    return {'pearson': pearson_r, 'rmse': rmse, 'r2': r2, 'mae': mae}


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    os.makedirs(args.result_path, exist_ok=True)

    with open(args.config_path) as f:
        config = json.load(f)

    print("=" * 60)
    print("TRACE-Kin v6b: XGBoost on v5t Deep Features")
    print("=" * 60)
    print(f"v5t checkpoint: {args.v5t_checkpoint}")
    print(f"Dataset:        {args.datafolder}")
    print(f"Result:         {args.result_path}")
    print(f"Seed:           {args.seed}")
    print(f"Backend:        {'LightGBM' if args.use_lgb else 'XGBoost'}")
    print("=" * 60)

    # --- Load data using standard TRACE-Kin pipeline ---
    print("\n[1/4] Loading and preprocessing data...")
    train_df, test_df = load_data(args, config)
    print(f"  Train: {len(train_df)} samples, Test: {len(test_df)} samples")

    protein_init_with_embedding(
        args.datafolder, train_df, test_df,
        args.protein_col, args.feature_col, force_rebuild=False)

    ligand_init(args.datafolder, train_df, test_df, args.ligand_col, force_rebuild=False)

    if args.molformer_path:
        compute_molformer_embeddings(
            args.datafolder, train_df, test_df,
            args.ligand_col, args.molformer_path)

    train_dataset = ProteinMoleculeDataset(
        args.datafolder, train_df, label_col=args.label_col,
        metabolite_feature_col=args.metabolite_feature_col,
        regression=True, classification=False, multiclassification=0)
    test_dataset = ProteinMoleculeDataset(
        args.datafolder, test_df, label_col=args.label_col,
        metabolite_feature_col=args.metabolite_feature_col,
        regression=True, classification=False, multiclassification=0)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # --- Build v5t model and load weights ---
    print("\n[2/4] Loading v5t model from checkpoint...")
    mol_deg, prot_deg = compute_pna_degrees(args.datafolder)
    params = config["params"]

    model = TraceKinV5T(
        mol_deg, prot_deg,
        prot_evo_channels=params["prot_evo_channels"],
        d_model=params.get("d_model", 512),
        n_transformer_layers=params.get("n_transformer_layers", 4),
        n_self_attn_heads=params.get("n_self_attn_heads", 8),
        ffn_expand=params.get("ffn_expand", 4),
        graph_pe_rwse_steps=params.get("graph_pe_rwse_steps", 16),
        mol_in_channels=params["mol_in_channels"],
        n_drug_pna_layers=params.get("n_drug_pna_layers", 3),
        n_cross_heads=params.get("n_cross_heads", 8),
        chembert_dim=params.get("chembert_dim", 768),
        dropout=params.get("dropout", 0.1),
        input_dropout=params.get("input_dropout", 0.15),
        regression_head=True,
        classification_head=False,
        multiclassification_head=0,
        device=str(device),
        heads=params.get("heads", 8),
    ).to(device)

    state_dict = torch.load(args.v5t_checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"  Loaded checkpoint: {args.v5t_checkpoint}")

    # --- Extract features ---
    print("\n[3/4] Extracting v5t interaction fingerprints...")
    X_train, y_train = extract_features(model, train_loader, device)
    X_test, y_test = extract_features(model, test_loader, device)
    print(f"  Train features: {X_train.shape}, Test features: {X_test.shape}")

    # Save extracted features for reuse
    np.savez(os.path.join(args.result_path, "v5t_features.npz"),
             X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    # --- Train tree model ---
    print("\n[4/4] Training tree model...")
    if args.use_lgb:
        train_pred, test_pred, tree_model = train_lightgbm(
            X_train, y_train, X_test, y_test, seed=args.seed)
    else:
        train_pred, test_pred, tree_model = train_xgboost(
            X_train, y_train, X_test, y_test, seed=args.seed)

    # --- Evaluate ---
    train_metrics = evaluate(y_train, train_pred, "train")
    test_metrics = evaluate(y_test, test_pred, "test")

    # --- Save results ---
    results_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': test_pred,
    })
    results_df.to_csv(os.path.join(args.result_path, f"v6b_test_prediction_seed{args.seed}.csv"), index=False)

    metrics_summary = {
        'train': train_metrics,
        'test': test_metrics,
        'config': {
            'v5t_checkpoint': args.v5t_checkpoint,
            'datafolder': args.datafolder,
            'seed': args.seed,
            'backend': 'lightgbm' if args.use_lgb else 'xgboost',
            'feature_dim': X_train.shape[1],
        }
    }
    with open(os.path.join(args.result_path, f"v6b_metrics_seed{args.seed}.json"), 'w') as f:
        json.dump(metrics_summary, f, indent=2)

    print("\nDone! Results saved to:", args.result_path)


if __name__ == "__main__":
    main()

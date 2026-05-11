"""Single-pair TRACE-Kin inference wrapper.

Usage::

    from inference import TraceKinPredictor

    predictor = TraceKinPredictor(
        weights_dir="path/to/save_model_seed1",
        embedding="ESM2",                 # for routing protein_features through the right encoder
        embedding_provider=my_esm_fn,     # callable: seq -> np.ndarray of shape (L, 1280)
        device="cuda:0",
    )
    result = predictor.predict("CCO", "MEEPQSD...")  # SMILES, enzyme sequence
    # result -> {"prediction": 6.21, "kinetic_type": "kcat", "model_version": "v2"}

The predictor reads ``config.json`` and ``degree.pt`` from ``weights_dir`` to
choose between :class:`TraceKinV1` and :class:`TraceKinV2`. Per-checkpoint
``version_map.json`` (optional) lets the caller pin which kinetic type each
checkpoint was trained for, so a wrapper service can build kinetic→predictor
maps without loading the wrong file.

This module deliberately avoids defining its own protein/ligand graph
construction code — it reuses :func:`training.protein_init_with_embedding.protein_init_with_embedding`
and :func:`training.ligand_init.ligand_init` so the inference path is
bit-for-bit identical to the training path.
"""
from __future__ import annotations

import json
import os
from typing import Callable

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from models.trace_kin import TraceKinV1, TraceKinV2

from training.dataset import ProteinMoleculeDataset
from training.ligand_init import ligand_init
from training.protein_init_with_embedding import protein_init_with_embedding


class TraceKinPredictor:
    """Load a trained TRACE-Kin checkpoint and predict on a single (SMILES, sequence) pair."""

    def __init__(
        self,
        weights_dir: str,
        embedding: str,
        embedding_provider: Callable[[str], np.ndarray] | None = None,
        kinetic_type: str | None = None,
        device: str = "cpu",
    ):
        self.weights_dir = weights_dir
        self.embedding = embedding
        self.embedding_provider = embedding_provider
        self.kinetic_type = kinetic_type
        self.device = torch.device(device)

        # Load config + degree statistics + checkpoint.
        config_path = os.path.join(weights_dir, "config.json")
        degree_path = os.path.join(weights_dir, "degree.pt")
        ckpt_path = os.path.join(weights_dir, "model.pt")
        for path in (config_path, degree_path, ckpt_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"missing {path} in weights_dir")

        with open(config_path) as f:
            self.config = json.load(f)
        version_map_path = os.path.join(weights_dir, "version_map.json")
        if os.path.exists(version_map_path):
            with open(version_map_path) as f:
                version_map = json.load(f)
            if kinetic_type and kinetic_type in version_map:
                self.config["model_version"] = version_map[kinetic_type]

        degree_dict = torch.load(degree_path, map_location="cpu")
        mol_deg = degree_dict["ligand_deg"]
        prot_deg = degree_dict["protein_deg"]

        self.model = self._build_model(mol_deg, prot_deg)
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.model.to(self.device)

    def _build_model(self, mol_deg, prot_deg):
        params = self.config["params"]
        tasks = self.config["tasks"]
        version = self.config.get("model_version", "v1")
        common = dict(
            mol_in_channels=params["mol_in_channels"],
            prot_in_channels=params["prot_in_channels"],
            prot_evo_channels=params["prot_evo_channels"],
            hidden_channels=params["hidden_channels"],
            pre_layers=params["pre_layers"],
            post_layers=params["post_layers"],
            aggregators=params["aggregators"],
            scalers=params["scalers"],
            total_layer=params["total_layer"],
            K=params["K"],
            heads=params["heads"],
            dropout=params.get("dropout", 0),
            dropout_attn_score=params.get("dropout_attn_score", 0.2),
            regression_head=tasks["regression_task"],
            classification_head=tasks["classification_task"],
            multiclassification_head=tasks["mclassification_task"],
            device=str(self.device),
        )
        if version == "v1":
            return TraceKinV1(
                mol_deg, prot_deg,
                **common,
                use_gated_prot_fusion=params.get("use_gated_prot_fusion", False),
                deep_regression_head=params.get("deep_regression_head", False),
                learnable_aux_loss=params.get("learnable_aux_loss", False),
            )
        return TraceKinV2(
            mol_deg, prot_deg,
            **common,
            aa_residual_weight=params.get("aa_residual_weight", 0.1),
            shortcut_hidden=params.get("shortcut_hidden", 400),
        )

    def predict(self, smiles: str, sequence: str) -> dict:
        """Run a single forward pass and return the scalar prediction."""
        if self.embedding_provider is None:
            raise RuntimeError(
                "embedding_provider is required. Pass a callable that maps a "
                "protein sequence to a (L, prot_evo_channels) numpy array, "
                "produced by the same embedding pipeline used at training time."
            )
        seq_feat = self.embedding_provider(sequence)
        seq_feat = np.asarray(seq_feat, dtype=np.float32)

        # Build protein and ligand graphs using the training-side initializers
        # so featurization is identical to training.
        protein_dict = protein_init_with_embedding([sequence], seq2feat={sequence: seq_feat})
        ligand_dict = ligand_init([smiles])
        if smiles not in ligand_dict:
            raise ValueError(f"RDKit failed to parse SMILES: {smiles!r}")

        # Single-row dataframe driving ProteinMoleculeDataset.
        import pandas as pd
        df = pd.DataFrame([{"Protein": sequence, "Ligand": smiles}])
        ds = ProteinMoleculeDataset(df, ligand_dict, protein_dict, device=self.device)
        loader = DataLoader(ds, batch_size=1, shuffle=False,
                             follow_batch=["mol_x", "clique_x", "prot_node_aa"])

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                reg_pred, *_ = self.model(
                    mol_x=batch.mol_x, mol_x_feat=batch.mol_x_feat, bond_x=batch.mol_edge_attr,
                    atom_edge_index=batch.mol_edge_index, clique_x=batch.clique_x,
                    clique_edge_index=batch.clique_edge_index, atom2clique_index=batch.atom2clique_index,
                    residue_x=batch.prot_node_aa, residue_evo_x=batch.prot_node_evo,
                    residue_edge_index=batch.prot_edge_index,
                    residue_edge_weight=batch.prot_edge_weight,
                    mol_batch=batch.mol_x_batch, prot_batch=batch.prot_node_aa_batch,
                    clique_batch=batch.clique_x_batch,
                    # v3 FP-MLP ChemBERT/MoLFormer embedding; v1 ignores this kwarg.
                    chembert_fp=getattr(batch, 'chembert_fp', None),
                    # v4 per-residue MutaPLM-typical mean/std (None for v1/v3).
                    aa_typical_mean=getattr(batch, 'aa_typical_mean', None),
                    aa_typical_std=getattr(batch, 'aa_typical_std', None),
                )
                value = reg_pred.squeeze().detach().cpu().item()
                return {
                    "prediction": value,
                    "kinetic_type": self.kinetic_type,
                    "model_version": self.config.get("model_version", "v1"),
                    "embedding": self.embedding,
                }

        raise RuntimeError("DataLoader produced no batches.")

"""TRACE-Kin v3: v1 backbone + ChemBERT global graph context.

After the FP-MLP/gate variant underperformed (single MLP can't decompose
ChemBERT's 768-d dense embedding the way RF can on small per-folder train
sets), v3 was simplified: drop the parallel FP-MLP head and the gate; use
ChemBERT as a global graph context that enriches the GNN's pooled
molecular representation before the regression head.

* GNN backbone — v1 verbatim, produces mol_pool (200) and prot_pool (200)
  via the standard PNAConv → MotifPool → MinCut → DrugProteinConv pipeline.
* ChemBERT projection — single Linear(768, hidden_channels) projecting
  the per-molecule ChemBERT (768) embedding into the same space as
  mol_pool. The projection is added to mol_pool as a residual.
* Regression head — fresh MLP([2*hidden, hidden, 1]) over
  cat(mol_pool_enriched, prot_pool). Same shape as v1's, but a separate
  parameter set (since the inputs differ in distribution after ChemBERT
  enrichment).

The chembert projection's weight is zero-initialized so v3 starts at the
v1 baseline (mol_pool_enriched ≈ mol_pool) and only deviates as training
discovers useful ChemBERT signal. The fresh reg_out MLP doesn't inherit
v1's regression head weights — the input distribution changes once the
projection wakes up, so a fresh head is the right choice.

Cross-attention scores from the GNN backbone (residue / atom / clique
attentions) flow through unchanged in attention_dict, so downstream
TRACE-Reason / TRACE-Gen interpretability is unaffected.

No auxiliary losses: the GNN backbone still runs MinCut pooling for the
cluster-shaped DrugProteinConv layer, but the auxiliary loss tensors are
zeroed before being returned, so the trainer's existing loss composition
treats them as no-ops.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .layers import MLP
from .net_v1 import TraceKinV1


class TraceKinV3(nn.Module):
    def __init__(
        self,
        mol_deg,
        prot_deg,
        # v1 backbone args (forwarded verbatim)
        mol_in_channels: int = 43,
        prot_in_channels: int = 33,
        prot_evo_channels: int = 1280,
        hidden_channels: int = 200,
        pre_layers: int = 2,
        post_layers: int = 1,
        aggregators=("mean", "min", "max", "std"),
        scalers=("identity", "amplification", "linear"),
        total_layer: int = 3,
        K=(5, 10, 20),
        t: int = 1,
        heads: int = 5,
        dropout: float = 0.0,
        dropout_attn_score: float = 0.2,
        drop_atom: float = 0.0,
        drop_residue: float = 0.0,
        dropout_cluster_edge: float = 0.0,
        regression_head: bool = True,
        classification_head: bool = False,
        multiclassification_head: int = 0,
        device: str = "cuda:0",
        # v3-specific
        chembert_dim: int = 768,
    ):
        super().__init__()

        # GNN backbone — exactly v1. Its own regression head is computed but
        # discarded in forward(); the v3 head consumes the enriched mol_pool
        # plus prot_pool from the backbone's intermediates.
        self.gnn = TraceKinV1(
            mol_deg=mol_deg,
            prot_deg=prot_deg,
            mol_in_channels=mol_in_channels,
            prot_in_channels=prot_in_channels,
            prot_evo_channels=prot_evo_channels,
            hidden_channels=hidden_channels,
            pre_layers=pre_layers,
            post_layers=post_layers,
            aggregators=list(aggregators),
            scalers=list(scalers),
            total_layer=total_layer,
            K=list(K) if not isinstance(K, int) else K,
            t=t,
            heads=heads,
            dropout=dropout,
            dropout_attn_score=dropout_attn_score,
            drop_atom=drop_atom,
            drop_residue=drop_residue,
            dropout_cluster_edge=dropout_cluster_edge,
            regression_head=regression_head,
            classification_head=classification_head,
            multiclassification_head=multiclassification_head,
            device=device,
        )

        self.hidden_channels = hidden_channels
        self.chembert_dim = chembert_dim

        # Project the per-molecule ChemBERT (768) into the GNN's hidden
        # space (200). Zero-init the weight so v3 starts at v1 baseline
        # (chembert_residual = 0); training learns a non-trivial projection
        # only if the data supports it.
        self.chembert_proj = nn.Linear(chembert_dim, hidden_channels)
        with torch.no_grad():
            self.chembert_proj.weight.zero_()
            self.chembert_proj.bias.zero_()

        # Fresh regression head over (mol_pool_enriched ⊕ prot_pool). Same
        # shape as v1's reg_out (MLP([h*2, h, 1])); separate parameters
        # because the input distribution changes once chembert_proj
        # contributes a non-zero residual.
        self.reg_out = MLP([hidden_channels * 2, hidden_channels, 1])

        self.regression_head = regression_head
        self.classification_head = classification_head
        self.multiclassification_head = multiclassification_head
        self.device = device

        # Mirror v1's introspection attribute so trainer code that checks
        # this works for v3 too.
        self.learnable_aux_loss = False

    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.reg_out.reset_parameters()
        # Re-zero the chembert projection so reset_parameters yields a
        # v3-starts-at-v1 model, matching __init__ semantics.
        with torch.no_grad():
            self.chembert_proj.weight.zero_()
            self.chembert_proj.bias.zero_()

    def forward(
        self,
        # Molecule
        mol_x, mol_x_feat, bond_x, atom_edge_index,
        clique_x, clique_edge_index, atom2clique_index,
        # Protein
        residue_x, residue_evo_x, residue_edge_index, residue_edge_weight,
        # Mol-Protein Interaction batch
        mol_batch=None, prot_batch=None, clique_batch=None,
        save_cluster: bool = False,
        # v3-specific: per-graph ChemBERT molecular embedding (768)
        chembert_fp: Optional[torch.Tensor] = None,
    ):
        # 1. Run the GNN backbone. We use only its mol_pool / prot_pool /
        #    attention_dict; the backbone's own pred_gnn is discarded.
        _, cls_pred, mcls_pred, sp_loss, o_loss, cl_loss, attention_dict = self.gnn(
            mol_x=mol_x, mol_x_feat=mol_x_feat, bond_x=bond_x, atom_edge_index=atom_edge_index,
            clique_x=clique_x, clique_edge_index=clique_edge_index, atom2clique_index=atom2clique_index,
            residue_x=residue_x, residue_evo_x=residue_evo_x,
            residue_edge_index=residue_edge_index, residue_edge_weight=residue_edge_weight,
            mol_batch=mol_batch, prot_batch=prot_batch, clique_batch=clique_batch,
            save_cluster=save_cluster,
        )

        mol_pool = attention_dict["mol_feature"]    # (B, 200)
        prot_pool = attention_dict["prot_feature"]  # (B, 200)

        # Drop MinCut auxiliary losses — they were a major reason v1 lost
        # to RF on catalytic kinetics (PROJECT.md §6 cause #3). Trainer
        # weights them at 1.0; leaving them non-zero would make the
        # optimizer minimize aux-loss instead of regression loss.
        zero_loss = mol_pool.new_zeros(())
        sp_loss = zero_loss
        o_loss = zero_loss
        cl_loss = zero_loss

        # Non-regression tasks: just route the GNN's classification heads
        # through unchanged. ChemBERT enrichment is a regression-only path.
        if not self.regression_head:
            return None, cls_pred, mcls_pred, sp_loss, o_loss, cl_loss, attention_dict

        # 2. ChemBERT global context — required for regression. Raise
        #    loudly if the caller forgot to pass chembert_fp; silent
        #    fallback to bare GNN would use a randomly-initialized reg_out
        #    on un-enriched mol_pool, producing nonsense.
        if chembert_fp is None:
            raise RuntimeError(
                "TraceKinV3 requires chembert_fp kwarg for regression. All "
                "model() call sites in trainer.py, data_utils.virtual_screening, "
                "and inference/ must pass it. v1 silently ignores this kwarg; "
                "v3 does not."
            )
        chembert = chembert_fp.float()                                     # (B, 768) or (B, 1, 768)
        if chembert.dim() == 3:
            # Per-graph cache stores shape (1, 768); some PyG collate
            # paths batch to (B, 1, 768). Squeeze the singleton.
            chembert = chembert.squeeze(1)
        chembert_residual = self.chembert_proj(chembert)                  # (B, 200)
        mol_pool_enriched = mol_pool + chembert_residual                  # (B, 200)

        # 3. Fresh regression head over enriched mol_pool ⊕ prot_pool.
        mol_prot_feat = torch.cat([mol_pool_enriched, prot_pool], dim=-1)  # (B, 400)
        reg_pred = self.reg_out(mol_prot_feat)                             # (B, 1)

        # Expose enriched pool + projection for downstream interpretability.
        attention_dict["mol_feature_enriched"] = mol_pool_enriched
        attention_dict["chembert_residual"] = chembert_residual

        return reg_pred, cls_pred, mcls_pred, sp_loss, o_loss, cl_loss, attention_dict

    def temperature_clamp(self):
        # Trainer calls this every iteration; v1's implementation is a no-op
        # (legacy logit_scale clamping is commented out there). v3 also has
        # no learnable temperature, so this stays a no-op.
        pass

    def configure_optimizers(self, weight_decay, learning_rate, betas, eps, amsgrad):
        """Same decay/no-decay split as v1, but iterates over v3's modules.

        Walks ``self.named_modules()`` so the v1 backbone, chembert_proj,
        and reg_out are all grouped consistently. Biases and Norm/Embedding
        weights skip weight decay; Linear weights get weight decay.
        """
        import torch_geometric
        from torch_geometric.nn.norm import GraphNorm
        from .layers import PosLinear

        decay, no_decay = set(), set()
        whitelist = (torch.nn.Linear, torch_geometric.nn.dense.linear.Linear)
        blacklist = (torch.nn.LayerNorm, torch.nn.Embedding, GraphNorm, PosLinear)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias") or pn.endswith("mean_scale"):
                    no_decay.add(fpn)
                elif "loss_log_var" in pn:
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter = decay & no_decay
        union = decay | no_decay
        assert not inter, f"params in both decay/no_decay: {inter}"
        assert not (param_dict.keys() - union), \
            f"params not assigned: {param_dict.keys() - union}"

        groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, eps=eps, amsgrad=amsgrad)

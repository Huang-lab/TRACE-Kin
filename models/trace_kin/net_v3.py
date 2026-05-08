"""TRACE-Kin v3: dual-head architecture with learned per-sample gating.

Two prediction heads share v1's GNN encoder, combined via a sigmoid gate:

* GNN head:  v1's regression head on top of the standard mol_pool ⊕ prot_pool.
             Wins on Ki tasks where binding-pocket geometry matters.
* RF head:   mean-pooled raw protein embedding ⊕ Morgan FP ⊕ MACCS FP → MLP.
             Receives the same features RF baselines use, so this path can
             match RF where the GNN path can't.
* Gate:      sigmoid network outputs α ∈ [0,1] per sample. Final prediction
             is α · pred_gnn + (1 − α) · pred_rf. Training pushes α high on
             tasks where the GNN wins (Ki) and low on tasks where RF features
             win (catalytic kinetics). v2's mistake was unconditional
             concatenation; v3 makes the trust decision explicit.

Cross-attention scores from the GNN are preserved in the returned attention
dict, so downstream TRACE-Reason / TRACE-Gen interpretability is unaffected.

No auxiliary losses (drops MinCut's ortho_loss / cluster_loss). The GNN
backbone still runs MinCut pooling for its cluster-shaped DrugProteinConv,
but the auxiliary loss tensors are zeroed before being returned, so the
trainer's existing loss composition treats them as no-ops.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_scatter import scatter

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
        # v3-specific args
        morgan_dim: int = 2048,
        maccs_dim: int = 167,
        rf_head_hidden=(512, 128),
        gate_hidden: int = 64,
        gate_init_bias: float = 0.0,
    ):
        super().__init__()

        # GNN backbone — exactly v1, including its regression head producing pred_gnn.
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

        # RF-style head — same input shape as a typical RF baseline:
        # mean-pooled raw embedding ⊕ Morgan FP ⊕ MACCS FP.
        rf_in_dim = prot_evo_channels + morgan_dim + maccs_dim
        rf_layers = []
        prev = rf_in_dim
        for h in rf_head_hidden:
            rf_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        rf_layers += [nn.Linear(prev, 1)]
        self.rf_head = nn.Sequential(*rf_layers)

        # Gate network — sees both GNN summary and RF inputs so it can decide
        # which path to trust per sample. Output bias initialized so α ≈ 0.5
        # by default; training moves it toward the better path per task.
        gate_in_dim = (2 * hidden_channels) + rf_in_dim
        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
        )
        with torch.no_grad():
            self.gate[-1].bias.fill_(gate_init_bias)

        self.regression_head = regression_head
        self.classification_head = classification_head
        self.multiclassification_head = multiclassification_head
        self.device = device

        # Surface for trainer / freezer: v3 has no learnable aux loss like v1
        # could; mirror the attribute so any code that introspects it works.
        self.learnable_aux_loss = False

    def reset_parameters(self):
        self.gnn.reset_parameters()
        for layer in self.rf_head:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for layer in self.gate:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        # Re-initialize the gate's output bias after reset.
        with torch.no_grad():
            self.gate[-1].bias.zero_()

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
        # v3-specific: per-graph ligand fingerprints
        morgan_fp: Optional[torch.Tensor] = None,
        maccs_fp: Optional[torch.Tensor] = None,
    ):
        # 1. Run the GNN backbone. v1 returns a 7-tuple; pred_gnn is reg_pred.
        pred_gnn, cls_pred, mcls_pred, sp_loss, o_loss, cl_loss, attention_dict = self.gnn(
            mol_x=mol_x, mol_x_feat=mol_x_feat, bond_x=bond_x, atom_edge_index=atom_edge_index,
            clique_x=clique_x, clique_edge_index=clique_edge_index, atom2clique_index=atom2clique_index,
            residue_x=residue_x, residue_evo_x=residue_evo_x,
            residue_edge_index=residue_edge_index, residue_edge_weight=residue_edge_weight,
            mol_batch=mol_batch, prot_batch=prot_batch, clique_batch=clique_batch,
            save_cluster=save_cluster,
        )

        # v3 drops the MinCut auxiliary losses — they were a major reason v1 lost
        # to RF on catalytic kinetics (PROJECT.md §6 cause #3). The trainer
        # weights them at 1.0, so leaving them non-zero makes the optimizer
        # minimize aux-loss instead of regression loss. Zero them out.
        zero_loss = pred_gnn.new_zeros(()) if pred_gnn is not None else sp_loss.new_zeros(())
        sp_loss = zero_loss
        o_loss = zero_loss
        cl_loss = zero_loss

        # If we're not in regression mode (rare for kinetic tasks), return the
        # GNN result unchanged — the gate is only meaningful for regression.
        if pred_gnn is None or morgan_fp is None or maccs_fp is None:
            return pred_gnn, cls_pred, mcls_pred, sp_loss, o_loss, cl_loss, attention_dict

        # 2. RF-style head input: mean-pool the raw 1280-dim embedding per
        #    sample, then concatenate ligand fingerprints.
        evo_mean = scatter(residue_evo_x, prot_batch, dim=0, reduce="mean")  # (B, 1280)
        morgan = morgan_fp.float()                                           # (B, 2048)
        maccs = maccs_fp.float()                                             # (B, 167)
        rf_features = torch.cat([evo_mean, morgan, maccs], dim=-1)           # (B, 3495)
        pred_rf = self.rf_head(rf_features)                                  # (B, 1)

        # 3. Gate input: GNN summary (mol_pool + prot_pool) plus RF features.
        mol_pool = attention_dict["mol_feature"]                             # (B, 200)
        prot_pool = attention_dict["prot_feature"]                           # (B, 200)
        gate_features = torch.cat([mol_pool, prot_pool, rf_features], dim=-1)
        alpha = torch.sigmoid(self.gate(gate_features))                      # (B, 1)

        # 4. Combine.
        pred_combined = alpha * pred_gnn + (1.0 - alpha) * pred_rf

        # Expose gate + heads for analysis / interpretability.
        attention_dict["gate_alpha"] = alpha
        attention_dict["pred_gnn"] = pred_gnn
        attention_dict["pred_rf"] = pred_rf

        return pred_combined, cls_pred, mcls_pred, sp_loss, o_loss, cl_loss, attention_dict

    def temperature_clamp(self):
        # Trainer calls this every iteration; v1's implementation is a no-op
        # (the legacy logit_scale clamping is commented out there). v3 has no
        # learnable temperature either, so this stays a no-op.
        pass

    def configure_optimizers(self, weight_decay, learning_rate, betas, eps, amsgrad):
        """Same decay/no-decay split as v1, but iterates over v3's modules.

        Walks ``self.named_modules()`` so the v1 backbone, the rf_head, and
        the gate are all grouped consistently. Biases and Norm/Embedding
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

"""TRACE-Kin v4: MAP-GNN — Mutation-Aware Pocket GNN.

Three architectural changes from v3-MoLFormer:

1. **Per-residue novelty score**. For each residue, compute
   ``||MutaPLM[r] - aa_typical_mean[aa[r]]|| / aa_typical_std[aa[r]]``,
   where aa_typical statistics are precomputed once over the training
   corpus. The score is high when a residue's MutaPLM embedding is
   unusual for its amino acid type — exactly what a single point
   mutation produces, without needing explicit (WT, mutant) labels.

2. **Pocket attention**. A small MLP over [residue_evo, novelty,
   residue_degree] produces a per-residue logit. Softmax over the
   protein gives ``pocket_weight[r]`` summing to 1 per protein. Soft
   pocket — every residue contributes, but residues at the active site
   (which the model learns to focus on) dominate.

3. **Pocket-weighted protein pool**. v1's ``residue_attn_lin`` pool is
   replaced by ``sum_r pocket_weight[r] * residue_h[r]``, where
   residue_h is the post-GNN per-residue feature exposed by v1's
   attention_dict (added to v1 in a non-invasive manner — see
   net_v1.py 'residue_x_post_gnn' key).

What's preserved from v3-MoLFormer:
- v1 GNN backbone (PNAConv ×3 over contact map; MotifPool ×3 on drug)
- MoLFormer/ChemBERT context residual added to mol_pool
- Multi-seed ensemble; per-folder eval; no auxiliary GNN losses

The chembert_proj weight is zero-initialized as in v3, so the model
starts at v1+novelty-injection baseline; the MoLFormer signal wakes
up only as training discovers it. The pocket attention starts as
softmax over uniform logits (Linear weights at default init), so the
initial pool is approximately uniform across residues — gradually
sharpens as training assigns higher logits to functionally-important
residues.

Per-residue mean/std for novelty are passed in via per-residue
features (aa_typical_mean, aa_typical_std), batched alongside
prot_node_evo by PyG. See training/dataset.py and aa_typical.py.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.utils import degree, softmax
from torch_scatter import scatter

from .layers import MLP
from .net_v1 import TraceKinV1


class TraceKinV4(nn.Module):
    def __init__(
        self,
        mol_deg,
        prot_deg,
        # v1 backbone args (forwarded verbatim)
        mol_in_channels: int = 43,
        prot_in_channels: int = 33,
        prot_evo_channels: int = 1024,        # MutaPLM dim
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
        # v4-specific
        chembert_dim: int = 768,
        pocket_hidden: int = 64,
    ):
        super().__init__()

        # GNN backbone — exactly v1. Its own pred_gnn / prot_pool are
        # discarded; v4 replaces the protein pool with a pocket-attention
        # pool over the post-GNN residue features (exposed by v1 in
        # attention_dict["residue_x_post_gnn"]).
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
        self.prot_evo_channels = prot_evo_channels
        self.chembert_dim = chembert_dim

        # Pocket attention MLP. Inputs per residue:
        #   - residue_evo (prot_evo_channels = MutaPLM dim, e.g. 1024)
        #   - novelty score (1)
        #   - residue degree (1)
        # Output: scalar logit; softmax-pooled per protein.
        self.pocket_attn = nn.Sequential(
            nn.Linear(prot_evo_channels + 2, pocket_hidden),
            nn.ReLU(),
            nn.Linear(pocket_hidden, 1),
        )

        # Molecular context residual (preserved from v3). Zero-init so v4
        # starts at v1+novelty baseline; chembert signal wakes up gradually.
        self.chembert_proj = nn.Linear(chembert_dim, hidden_channels)
        with torch.no_grad():
            self.chembert_proj.weight.zero_()
            self.chembert_proj.bias.zero_()

        # Fresh regression head over (mol_pool_enriched ⊕ pocket_pool).
        # Same shape as v1's reg_out so parameter counts stay comparable.
        self.reg_out = MLP([hidden_channels * 2, hidden_channels, 1])

        self.regression_head = regression_head
        self.classification_head = classification_head
        self.multiclassification_head = multiclassification_head
        self.device = device

        # Mirror v1's introspection attribute so trainer code that checks
        # this works for v4 too.
        self.learnable_aux_loss = False

    def reset_parameters(self):
        self.gnn.reset_parameters()
        for layer in self.pocket_attn:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        self.reg_out.reset_parameters()
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
        # v3-style ChemBERT/MoLFormer molecular embedding
        chembert_fp: Optional[torch.Tensor] = None,
        # v4-specific: per-residue mean/std for novelty (computed offline
        # via training.aa_typical, batched alongside prot_node_evo)
        aa_typical_mean: Optional[torch.Tensor] = None,
        aa_typical_std: Optional[torch.Tensor] = None,
    ):
        # 1. Run the GNN backbone. Use only residue_x_post_gnn / mol_feature /
        #    attention_dict; discard the backbone's own pred and prot_feature.
        _, cls_pred, mcls_pred, sp_loss, o_loss, cl_loss, attention_dict = self.gnn(
            mol_x=mol_x, mol_x_feat=mol_x_feat, bond_x=bond_x, atom_edge_index=atom_edge_index,
            clique_x=clique_x, clique_edge_index=clique_edge_index, atom2clique_index=atom2clique_index,
            residue_x=residue_x, residue_evo_x=residue_evo_x,
            residue_edge_index=residue_edge_index, residue_edge_weight=residue_edge_weight,
            mol_batch=mol_batch, prot_batch=prot_batch, clique_batch=clique_batch,
            save_cluster=save_cluster,
        )

        mol_pool = attention_dict["mol_feature"]                  # (B, hidden)
        residue_h = attention_dict["residue_x_post_gnn"]          # (N_res, hidden)

        # Drop MinCut auxiliary losses (same rationale as v3). Trainer
        # weights them at 1.0; leaving them non-zero would make the
        # optimizer minimize aux-loss instead of regression loss.
        zero_loss = mol_pool.new_zeros(())
        sp_loss = zero_loss
        o_loss = zero_loss
        cl_loss = zero_loss

        # Non-regression tasks: just route the GNN's classification heads
        # through unchanged. Pocket attention is a regression-only path.
        if not self.regression_head:
            return None, cls_pred, mcls_pred, sp_loss, o_loss, cl_loss, attention_dict

        # 2. v4-specific kwargs are required for regression. Raise loudly
        #    if any caller forgot to pass them — silent fallback would
        #    use a randomly-initialized reg_out on a uniform-pooled
        #    protein representation, producing nonsense.
        if chembert_fp is None:
            raise RuntimeError(
                "TraceKinV4 requires chembert_fp kwarg for regression. "
                "All model() call sites in trainer.py, "
                "data_utils.virtual_screening, and inference/ must pass it."
            )
        if aa_typical_mean is None or aa_typical_std is None:
            raise RuntimeError(
                "TraceKinV4 requires aa_typical_mean and aa_typical_std "
                "kwargs (per-residue MutaPLM-typical mean/std for the "
                "residue's amino acid type, used to compute the novelty "
                "score). Build via training.aa_typical and attach them "
                "to MultiGraphData in training/dataset.py."
            )

        # 3. Compute novelty per residue:
        #      novelty[r] = ||residue_evo[r] - aa_typical_mean[r]|| / ||aa_typical_std[r]||
        # The std-normalization makes the score scale-invariant to which
        # MutaPLM hidden dim happens to vary most across the corpus.
        diff = (residue_evo_x.float() - aa_typical_mean.float())
        norm_factor = aa_typical_std.float().norm(dim=-1).clamp(min=1e-3)
        novelty = diff.norm(dim=-1) / norm_factor                 # (N_res,)

        # 4. Pocket attention. Inputs per residue: evo embedding, novelty,
        #    graph degree (as a buriedness proxy). Logits → softmax over
        #    each protein's residues.
        deg = degree(residue_edge_index[0],
                     num_nodes=residue_evo_x.size(0)).float()      # (N_res,)
        pocket_in = torch.cat([
            residue_evo_x.float(),
            novelty.unsqueeze(-1),
            deg.unsqueeze(-1),
        ], dim=-1)                                                  # (N_res, evo+2)
        pocket_logits = self.pocket_attn(pocket_in).squeeze(-1)    # (N_res,)
        pocket_weight = softmax(pocket_logits, prot_batch)         # (N_res,)

        # 5. Pocket-weighted protein pool over post-GNN residue features.
        prot_pool = scatter(
            residue_h * pocket_weight.unsqueeze(-1),
            prot_batch, dim=0, reduce="sum",
        )                                                           # (B, hidden)

        # 6. MoLFormer/ChemBERT context residual on mol_pool (preserved
        #    from v3). Squeeze (B, 1, D) → (B, D) when PyG batches the
        #    per-graph (1, D) cache to (B, 1, D).
        chembert = chembert_fp.float()
        if chembert.dim() == 3:
            chembert = chembert.squeeze(1)
        chembert_residual = self.chembert_proj(chembert)            # (B, hidden)
        mol_pool_enriched = mol_pool + chembert_residual            # (B, hidden)

        # 7. Fresh regression head over enriched mol_pool ⊕ pocket_pool.
        mol_prot_feat = torch.cat([mol_pool_enriched, prot_pool], dim=-1)  # (B, 2*hidden)
        reg_pred = self.reg_out(mol_prot_feat)                              # (B, 1)

        # Expose v4 internals for downstream interpretability and
        # mechanism-validity diagnostics (Phase A in PROJECT.md plan).
        attention_dict["pocket_weight"] = pocket_weight
        attention_dict["novelty_score"] = novelty
        attention_dict["mol_feature_enriched"] = mol_pool_enriched
        attention_dict["prot_feature_pocket"] = prot_pool

        return reg_pred, cls_pred, mcls_pred, sp_loss, o_loss, cl_loss, attention_dict

    def temperature_clamp(self):
        # No learnable temperature; mirror v1/v3.
        pass

    def configure_optimizers(self, weight_decay, learning_rate, betas, eps, amsgrad):
        """Same decay/no-decay grouping as v1/v3, walking v4's modules.

        Biases and Norm/Embedding weights skip weight decay; Linear weights
        get weight decay.
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

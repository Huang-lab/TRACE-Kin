"""TRACE-Kin v6c: Hierarchical Pocket Graph + Multi-Modal Cross-Attention Fusion.

Inspired by GraphKcat (2025) and ERBA (2026), adapted for TRACE-Kin's
MutaPLM per-residue embeddings and contact-map graphs.

Key innovations over v5t:
  - GATv2 on contact-map graph replaces Transformer self-attention
    (graph-native, no padding, interpretable edge attention)
  - Dynamic pocket identification from protein-to-ligand cross-attention
    (no external docking tools required)
  - MMCAF: Multi-Modal Cross-Attention Fusion at the pocket level
  - Hierarchical pooling: gated fusion of pocket-level + global-level

Self-contained file (no cross-imports from v5/v5t) to avoid rollback coupling.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.utils import degree, to_dense_batch
from torch_geometric.utils import softmax as pyg_softmax
from torch_scatter import scatter

from .layers import MLP


# ---------------------------------------------------------------------------
# Shared building blocks (self-contained)
# ---------------------------------------------------------------------------

def _rbf(D: torch.Tensor, D_min: float = 0., D_max: float = 1.,
         D_count: int = 16, device: str = 'cpu') -> torch.Tensor:
    """Radial basis function embedding of scalar distances."""
    D = torch.where(D < D_max, D, torch.tensor(D_max, dtype=torch.float, device=device))
    D_mu = torch.linspace(D_min, D_max, D_count, device=device).view(1, -1)
    D_sigma = (D_max - D_min) / D_count
    return torch.exp(-((D.unsqueeze(-1) - D_mu) / D_sigma) ** 2)


class GraphStructuralEncoding(nn.Module):
    """Random-walk structural encoding from the protein contact-map."""

    def __init__(self, d_model: int, rwse_steps: int = 16):
        super().__init__()
        self.rwse_steps = rwse_steps
        self.proj = nn.Linear(rwse_steps, d_model)

    def forward(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        row, col = edge_index
        deg = degree(row, num_nodes=num_nodes).clamp(min=1)
        deg_inv = 1.0 / deg

        rw_landing = torch.zeros(num_nodes, self.rwse_steps, device=edge_index.device)
        rw_landing[:, 0] = 1.0

        cur = torch.ones(num_nodes, device=edge_index.device)
        for k in range(1, self.rwse_steps):
            msg = cur[col] * deg_inv[col]
            cur = scatter(msg, row, dim=0, dim_size=num_nodes, reduce='sum')
            rw_landing[:, k] = cur

        return self.proj(rw_landing)


class DrugPNAEncoder(nn.Module):
    """Ligand GNN encoder using PNA (matches v1's Drug_PNAConv wrapper)."""

    def __init__(self, mol_in_channels: int, d_model: int, mol_deg: torch.Tensor,
                 n_layers: int = 3, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        from .pna import PNAConv

        self.atom_type_encoder = nn.Embedding(20, d_model)
        self.atom_feat_encoder = MLP([mol_in_channels, d_model * 2, d_model], out_norm=True)
        self.bond_encoder = nn.Embedding(5, d_model)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(PNAConv(
                in_channels=d_model, out_channels=d_model,
                edge_dim=d_model,
                aggregators=['mean', 'min', 'max', 'std'],
                scalers=['identity', 'amplification', 'linear'],
                deg=mol_deg, pre_layers=2, post_layers=1, towers=heads,
            ))
            self.norms.append(GraphNorm(d_model))

        self.dropout = nn.Dropout(dropout)
        self.attn_pool = nn.Linear(d_model, 1)

    def forward(self, mol_x, mol_x_feat, bond_x, atom_edge_index, mol_batch):
        atom_h = self.atom_type_encoder(mol_x.squeeze()) + self.atom_feat_encoder(mol_x_feat)
        encoded_bonds = self.bond_encoder(bond_x.squeeze())
        for conv, norm in zip(self.convs, self.norms):
            atom_h = conv(atom_h, atom_edge_index, encoded_bonds)
            atom_h = norm(atom_h, mol_batch)
            atom_h = self.dropout(F.relu(atom_h))

        attn = pyg_softmax(self.attn_pool(atom_h).squeeze(-1), mol_batch)
        mol_pool = global_add_pool(atom_h * attn.unsqueeze(-1), mol_batch)
        return atom_h, mol_pool


class MultiHeadCrossAttention(nn.Module):
    """Cross-attention producing interpretable enzyme-ligand attention maps."""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, query_mask=None, key_mask=None):
        B, L_q, _ = query.shape
        _, L_k, _ = key.shape

        Q = self.W_Q(query).view(B, L_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(B, L_k, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(B, L_k, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if key_mask is not None:
            scores = scores.masked_fill(~key_mask[:, None, None, :], float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        out = self.W_O(out)
        return self.norm(query + out), attn


# ---------------------------------------------------------------------------
# v6c-specific: GATv2 protein encoder
# ---------------------------------------------------------------------------

class ProteinGATv2Encoder(nn.Module):
    """GATv2 encoder operating on the residue contact-map graph.

    Uses RBF-encoded edge weights as edge features and returns both
    per-residue embeddings and per-edge attention coefficients for
    interpretability.
    """

    def __init__(self, d_model: int, n_layers: int = 3, n_heads: int = 8,
                 rbf_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.n_layers = n_layers
        self.rbf_dim = rbf_dim

        self.edge_proj = nn.Linear(rbf_dim, d_model)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(GATv2Conv(
                in_channels=d_model,
                out_channels=d_model // n_heads,
                heads=n_heads,
                edge_dim=d_model,
                dropout=dropout,
                add_self_loops=True,
                concat=True,
            ))
            self.norms.append(nn.LayerNorm(d_model))
            self.ffns.append(nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.Dropout(dropout),
            ))

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> tuple[torch.Tensor, list]:
        edge_attr = _rbf(edge_weight, D_max=1.0, D_count=self.rbf_dim,
                         device=x.device)
        edge_feat = self.edge_proj(edge_attr)

        edge_attns = []
        for conv, norm, ffn in zip(self.convs, self.norms, self.ffns):
            h, alpha = conv(x, edge_index, edge_attr=edge_feat,
                            return_attention_weights=True)
            x = norm(x + h)
            x = x + ffn(x)
            edge_attns.append(alpha)

        x = self.final_norm(x)
        return x, edge_attns


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class TraceKinV6C(nn.Module):
    """TRACE-Kin v6c: Hierarchical Pocket Graph + Multi-Modal Fusion.

    Protein: MutaPLM per-residue -> projection -> RWSE graph PE ->
    GATv2 on contact-map graph -> dynamic pocket identification ->
    MMCAF at pocket level -> hierarchical pooling.
    Ligand: PNA GNN + MoLFormer context.
    """

    def __init__(
        self,
        mol_deg: torch.Tensor,
        prot_deg: torch.Tensor,
        prot_evo_channels: int = 4096,
        d_model: int = 512,
        n_gat_layers: int = 3,
        n_gat_heads: int = 8,
        gat_rbf_dim: int = 16,
        graph_pe_rwse_steps: int = 16,
        pocket_k: int = 64,
        mol_in_channels: int = 43,
        n_drug_pna_layers: int = 3,
        n_cross_heads: int = 8,
        chembert_dim: int = 768,
        dropout: float = 0.1,
        regression_head: bool = True,
        classification_head: bool = False,
        multiclassification_head: int = 0,
        device: str = "cuda:0",
        # v1-compat (unused, accepted for trainer compatibility)
        prot_in_channels: int = 33,
        hidden_channels: int = 200,
        pre_layers: int = 2,
        post_layers: int = 1,
        aggregators=None,
        scalers=None,
        total_layer: int = 3,
        K=None,
        t: int = 1,
        heads: int = 8,
        dropout_attn_score: float = 0.2,
        drop_atom: float = 0.0,
        drop_residue: float = 0.0,
        dropout_cluster_edge: float = 0.0,
        # v5-compat (unused)
        n_mamba_layers: int = 4,
        mamba_expand: int = 2,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        # v5t-compat (unused)
        n_transformer_layers: int = 4,
        n_self_attn_heads: int = 8,
        ffn_expand: int = 4,
        input_dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.prot_evo_channels = prot_evo_channels
        self.pocket_k = pocket_k
        self.regression_head = regression_head
        self.classification_head = classification_head
        self.multiclassification_head = multiclassification_head
        self.device = device
        self.learnable_aux_loss = False

        # --- Protein encoder ---
        self.prot_proj = nn.Linear(prot_evo_channels, d_model)
        self.prot_proj_norm = nn.LayerNorm(d_model)
        self.graph_pe = GraphStructuralEncoding(d_model, rwse_steps=graph_pe_rwse_steps)

        self.prot_gat = ProteinGATv2Encoder(
            d_model, n_layers=n_gat_layers, n_heads=n_gat_heads,
            rbf_dim=gat_rbf_dim, dropout=dropout)

        # --- Ligand encoder ---
        self.drug_encoder = DrugPNAEncoder(
            mol_in_channels, d_model, mol_deg,
            n_layers=n_drug_pna_layers, heads=heads, dropout=dropout)
        self.mol_context_proj = nn.Linear(chembert_dim, d_model)

        # --- Pocket identification cross-attention ---
        self.pocket_cross_attn = MultiHeadCrossAttention(d_model, n_cross_heads, dropout)

        # --- MMCAF: pocket-level bidirectional cross-attention ---
        self.mmcaf_prot = MultiHeadCrossAttention(d_model, n_cross_heads, dropout)
        self.mmcaf_lig = MultiHeadCrossAttention(d_model, n_cross_heads, dropout)

        # --- Hierarchical pooling ---
        self.pocket_pool_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.ReLU(), nn.Linear(d_model // 4, 1))
        self.global_pool_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.ReLU(), nn.Linear(d_model // 4, 1))
        self.mol_pool_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.ReLU(), nn.Linear(d_model // 4, 1))

        self.pocket_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Linear(d_model, 1), nn.Sigmoid())

        # --- Task heads ---
        if regression_head:
            self.reg_out = MLP([d_model * 2, d_model, 1])
        if classification_head:
            self.cls_out = MLP([d_model * 2, d_model, 2])
        if multiclassification_head:
            self.mcls_out = MLP([d_model * 2, d_model, multiclassification_head])

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def _pool_dense(self, h_dense, mask, pool_mlp):
        """Attention-weighted pooling over a dense (B, L, D) tensor with mask."""
        logits = pool_mlp(h_dense).squeeze(-1)
        logits = logits.masked_fill(~mask, float('-inf'))
        w = F.softmax(logits, dim=-1)
        pooled = (h_dense * w.unsqueeze(-1)).sum(1)
        return pooled, w

    def forward(
        self,
        mol_x, mol_x_feat, bond_x, atom_edge_index,
        clique_x, clique_edge_index, atom2clique_index,
        residue_x, residue_evo_x, residue_edge_index, residue_edge_weight,
        mol_batch=None, prot_batch=None, clique_batch=None,
        save_cluster: bool = False,
        chembert_fp: Optional[torch.Tensor] = None,
        prot_aa_idx: Optional[torch.Tensor] = None,
    ):
        reg_pred = None
        cls_pred = None
        mcls_pred = None
        zero_loss = residue_evo_x.new_zeros(())

        # === Protein: project + graph PE + GATv2 on contact map ===
        prot_h = self.prot_proj_norm(self.prot_proj(residue_evo_x.float()))
        graph_pe = self.graph_pe(residue_edge_index, num_nodes=prot_h.size(0))
        prot_h = prot_h + graph_pe

        prot_h, gat_edge_attns = self.prot_gat(
            prot_h, residue_edge_index, residue_edge_weight)

        # === Ligand: PNA + MoLFormer ===
        atom_h, _ = self.drug_encoder(mol_x, mol_x_feat, bond_x, atom_edge_index, mol_batch)

        if chembert_fp is not None:
            ctx = self.mol_context_proj(chembert_fp.float())
            if ctx.dim() == 3:
                ctx = ctx.squeeze(1)
            atom_h = atom_h + ctx[mol_batch]

        # === Dense batching for cross-attention ===
        prot_h_dense, prot_mask = to_dense_batch(prot_h, prot_batch)
        atom_h_dense, atom_mask = to_dense_batch(atom_h, mol_batch)

        B, L_prot, D = prot_h_dense.shape
        _, L_lig, _ = atom_h_dense.shape

        # === Dynamic pocket identification ===
        _, pocket_attn = self.pocket_cross_attn(
            prot_h_dense, atom_h_dense, atom_h_dense,
            query_mask=prot_mask, key_mask=atom_mask)
        # pocket_attn: (B, H, L_prot, L_lig)
        # Per-residue pocket score: mean over heads and ligand atoms
        pocket_score = pocket_attn.mean(dim=(1, 3))  # (B, L_prot)
        pocket_score = pocket_score.masked_fill(~prot_mask, 0.0)

        # Select top-K residues per sample as pocket
        K = min(self.pocket_k, L_prot)
        _, pocket_idx = pocket_score.topk(K, dim=-1)  # (B, K)
        pocket_mask = torch.zeros(B, L_prot, device=prot_h.device, dtype=torch.bool)
        pocket_mask.scatter_(1, pocket_idx, True)
        pocket_mask = pocket_mask & prot_mask

        # Extract pocket residue features
        pocket_features = prot_h_dense * pocket_mask.unsqueeze(-1).float()
        # Create a dense (B, K, D) tensor of pocket residues
        pocket_dense = torch.gather(
            prot_h_dense, 1,
            pocket_idx.unsqueeze(-1).expand(-1, -1, D))
        pocket_dense_mask = torch.gather(prot_mask, 1, pocket_idx)

        # === MMCAF: pocket-level bidirectional cross-attention ===
        pocket_attended, mmcaf_prot_attn = self.mmcaf_prot(
            pocket_dense, atom_h_dense, atom_h_dense,
            query_mask=pocket_dense_mask, key_mask=atom_mask)

        lig_attended, mmcaf_lig_attn = self.mmcaf_lig(
            atom_h_dense, pocket_dense, pocket_dense,
            query_mask=atom_mask, key_mask=pocket_dense_mask)

        # === Hierarchical pooling ===
        # Pocket-level pool (from MMCAF-refined pocket residues)
        pocket_pool, pocket_w = self._pool_dense(
            pocket_attended, pocket_dense_mask, self.pocket_pool_mlp)

        # Global-level pool (from GATv2-encoded full protein)
        global_pool, global_w = self._pool_dense(
            prot_h_dense, prot_mask, self.global_pool_mlp)

        # Gated fusion: learned combination of pocket + global
        gate = self.pocket_gate(torch.cat([pocket_pool, global_pool], dim=-1))
        prot_pool = gate * pocket_pool + (1.0 - gate) * global_pool

        # Ligand pool (from MMCAF-refined ligand)
        mol_pool, lig_w = self._pool_dense(
            lig_attended, atom_mask, self.mol_pool_mlp)

        # === Prediction ===
        feat = torch.cat([mol_pool, prot_pool], dim=-1)

        if self.regression_head:
            reg_pred = self.reg_out(feat)
        if self.classification_head:
            cls_pred = self.cls_out(feat)
        if self.multiclassification_head:
            mcls_pred = self.mcls_out(feat)

        # === Attention dict for interpretability ===
        # Expand pocket weights back to full protein length for v1-compatible storage
        full_prot_w = torch.zeros(B, L_prot, device=prot_h.device)
        full_prot_w.scatter_(1, pocket_idx, pocket_w)
        # Blend with global weights where pocket is absent
        full_prot_w = torch.where(pocket_mask, full_prot_w, global_w)

        prot_w_flat = full_prot_w[prot_mask]
        lig_w_flat = lig_w[atom_mask]

        attention_dict = {
            'residue_final_score': prot_w_flat,
            'atom_final_score': lig_w_flat,
            'residue_layer_scores': prot_w_flat,
            'clique_layer_scores': lig_w_flat,
            'drug_clique_index': mol_batch,
            'cluster_s': {},
            'pocket_score': pocket_score,
            'pocket_mask': pocket_mask,
            'pocket_gate': gate,
            'mmcaf_prot_attn': mmcaf_prot_attn,
            'mmcaf_lig_attn': mmcaf_lig_attn,
            'gat_edge_attns': gat_edge_attns,
            'mol_feature': mol_pool,
            'prot_feature': prot_pool,
            'interaction_fingerprint': feat,
            'drug_atom_index': mol_batch,
            'protein_residue_index': prot_batch,
        }

        return reg_pred, cls_pred, mcls_pred, zero_loss, zero_loss, zero_loss, attention_dict

    def temperature_clamp(self):
        pass

    def configure_optimizers(self, weight_decay, learning_rate, betas, eps, amsgrad):
        decay, no_decay = set(), set()
        whitelist = (nn.Linear, nn.Conv1d)
        blacklist = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist):
                    no_decay.add(fpn)
                else:
                    decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter = decay & no_decay
        if inter:
            no_decay -= inter

        unassigned = param_dict.keys() - (decay | no_decay)
        decay |= unassigned

        groups = [
            {"params": [param_dict[pn] for pn in sorted(decay) if pn in param_dict],
             "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay) if pn in param_dict],
             "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, eps=eps, amsgrad=amsgrad)

"""TRACE-Kin v5t: Vanilla Transformer + Cross-Attention.

Alternative to v5 (Graph-Mamba) for head-to-head comparison.
Replaces BiMamba + GraphGatedAggregation with standard Transformer
self-attention blocks using Rotary Positional Encoding (RoPE).

Same shared components as v5 (copied, not imported, for isolation):
  - GraphStructuralEncoding (RWSE from contact map)
  - DrugPNAEncoder (PNA GNN for ligand)
  - MultiHeadCrossAttention (enzyme-ligand interaction)
  - Substrate-conditioned pooling
  - MoLFormer context injection

Key difference: protein encoder uses O(L^2) self-attention vs v5's O(L)
Mamba scan. Should be more expressive on shorter sequences but slower
on long enzymes (>1000 residues).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import degree, to_dense_batch
from torch_geometric.utils import softmax as pyg_softmax
from torch_scatter import scatter

from .layers import MLP


# ---------------------------------------------------------------------------
# Shared building blocks (self-contained copies from v5)
# ---------------------------------------------------------------------------

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


class DrugPNAEncoder(nn.Module):
    """Ligand GNN encoder using PNA (matches v1's Drug_PNAConv wrapper)."""

    def __init__(self, mol_in_channels: int, d_model: int, mol_deg: torch.Tensor,
                 n_layers: int = 3, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        from .pna import PNAConv
        from torch_geometric.nn.norm import GraphNorm

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


# ---------------------------------------------------------------------------
# v5t-specific: Transformer protein encoder
# ---------------------------------------------------------------------------

class RotaryPositionalEncoding(nn.Module):
    """Rotary Positional Encoding (RoPE) for self-attention.

    Applies rotation to Q and K based on position, giving relative
    position awareness without learned embeddings. Decays naturally
    with distance.
    """

    def __init__(self, d_k: int, max_len: int = 4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer("inv_freq", inv_freq)
        self.max_len = max_len

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (cos, sin) each of shape (1, 1, seq_len, d_k)."""
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary embeddings to Q and K. All shapes (B, H, L, d_k)."""
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with RoPE self-attention + FFN.

    Pre-norm (LayerNorm before attention/FFN) is more stable for
    training deep models without warmup.
    """

    def __init__(self, d_model: int, n_heads: int = 8, ffn_expand: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_expand, d_model),
            nn.Dropout(dropout),
        )

        self.rope = RotaryPositionalEncoding(self.d_k)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pre-norm Transformer with RoPE + Flash Attention. (B, L, d_model) -> (B, L, d_model)."""
        B, L, _ = x.shape

        # Self-attention with RoPE
        h = self.norm1(x)
        Q = self.W_Q(h).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(h).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(h).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        cos, sin = self.rope(L, x.device)
        Q, K = apply_rope(Q, K, cos, sin)

        # Flash Attention: O(1) memory, never materializes L×L matrix.
        # Convert padding mask to additive bias for SDPA efficiency.
        attn_mask = None
        if mask is not None:
            # (B, 1, 1, L) -> broadcast to (B, H, L, L) inside SDPA
            attn_mask = torch.zeros(B, 1, 1, L, device=x.device, dtype=Q.dtype)
            attn_mask.masked_fill_(~mask[:, None, None, :], float('-inf'))

        dropout_p = self.attn_drop.p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=attn_mask, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.W_O(out)
        x = x + out

        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class TraceKinV5T(nn.Module):
    """TRACE-Kin v5t: Vanilla Transformer + Cross-Attention.

    Protein: MutaPLM per-residue -> projection -> RWSE graph PE ->
    Transformer self-attention x4 with RoPE -> cross-attention.
    Ligand: PNA GNN + MoLFormer context.
    Interaction: Multi-head cross-attention (interpretable).
    """

    def __init__(
        self,
        mol_deg: torch.Tensor,
        prot_deg: torch.Tensor,
        prot_evo_channels: int = 4096,
        d_model: int = 512,
        n_transformer_layers: int = 4,
        n_self_attn_heads: int = 8,
        ffn_expand: int = 4,
        graph_pe_rwse_steps: int = 16,
        mol_in_channels: int = 43,
        n_drug_pna_layers: int = 3,
        n_cross_heads: int = 8,
        chembert_dim: int = 768,
        dropout: float = 0.1,
        regression_head: bool = True,
        classification_head: bool = False,
        multiclassification_head: int = 0,
        device: str = "cuda:0",
        # v1-compat (unused)
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
    ):
        super().__init__()
        self.d_model = d_model
        self.prot_evo_channels = prot_evo_channels
        self.regression_head = regression_head
        self.classification_head = classification_head
        self.multiclassification_head = multiclassification_head
        self.device = device
        self.learnable_aux_loss = False
        self.gradient_checkpointing = True

        # Protein encoder
        self.prot_proj = nn.Linear(prot_evo_channels, d_model)
        self.prot_proj_norm = nn.LayerNorm(d_model)
        self.graph_pe = GraphStructuralEncoding(d_model, rwse_steps=graph_pe_rwse_steps)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads=n_self_attn_heads,
                             ffn_expand=ffn_expand, dropout=dropout)
            for _ in range(n_transformer_layers)
        ])

        # Ligand encoder
        self.drug_encoder = DrugPNAEncoder(
            mol_in_channels, d_model, mol_deg,
            n_layers=n_drug_pna_layers, heads=heads, dropout=dropout)
        self.mol_context_proj = nn.Linear(chembert_dim, d_model)

        # Cross-attention
        self.prot_cross_attn = MultiHeadCrossAttention(d_model, n_cross_heads, dropout)
        self.lig_cross_attn = MultiHeadCrossAttention(d_model, n_cross_heads, dropout)

        # Pooling
        self.prot_pool_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.ReLU(), nn.Linear(d_model // 4, 1))
        self.mol_pool_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.ReLU(), nn.Linear(d_model // 4, 1))

        # Task heads
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

        # === Protein: project + graph PE + Transformer self-attention ===
        prot_h = self.prot_proj_norm(self.prot_proj(residue_evo_x.float()))
        graph_pe = self.graph_pe(residue_edge_index, num_nodes=prot_h.size(0))
        prot_h = prot_h + graph_pe

        prot_h_dense, prot_mask = to_dense_batch(prot_h, prot_batch)

        for transformer_block in self.transformer_blocks:
            if self.gradient_checkpointing and self.training:
                prot_h_dense = grad_checkpoint(
                    transformer_block, prot_h_dense, prot_mask, use_reentrant=False)
            else:
                prot_h_dense = transformer_block(prot_h_dense, mask=prot_mask)

        # === Ligand: PNA + MoLFormer ===
        atom_h, _ = self.drug_encoder(mol_x, mol_x_feat, bond_x, atom_edge_index, mol_batch)

        if chembert_fp is not None:
            ctx = self.mol_context_proj(chembert_fp.float())
            if ctx.dim() == 3:
                ctx = ctx.squeeze(1)
            atom_h = atom_h + ctx[mol_batch]

        atom_h_dense, atom_mask = to_dense_batch(atom_h, mol_batch)

        # === Cross-attention ===
        prot_attended, prot_attn = self.prot_cross_attn(
            prot_h_dense, atom_h_dense, atom_h_dense,
            query_mask=prot_mask, key_mask=atom_mask)

        lig_attended, lig_attn = self.lig_cross_attn(
            atom_h_dense, prot_h_dense, prot_h_dense,
            query_mask=atom_mask, key_mask=prot_mask)

        # === Substrate-conditioned pooling ===
        prot_logits = self.prot_pool_mlp(prot_attended).squeeze(-1)
        prot_logits = prot_logits.masked_fill(~prot_mask, float('-inf'))
        prot_w = F.softmax(prot_logits, dim=-1)
        prot_pool = (prot_attended * prot_w.unsqueeze(-1)).sum(1)

        lig_logits = self.mol_pool_mlp(lig_attended).squeeze(-1)
        lig_logits = lig_logits.masked_fill(~atom_mask, float('-inf'))
        lig_w = F.softmax(lig_logits, dim=-1)
        mol_pool = (lig_attended * lig_w.unsqueeze(-1)).sum(1)

        # === Prediction ===
        feat = torch.cat([mol_pool, prot_pool], dim=-1)

        if self.regression_head:
            reg_pred = self.reg_out(feat)
        if self.classification_head:
            cls_pred = self.cls_out(feat)
        if self.multiclassification_head:
            mcls_pred = self.mcls_out(feat)

        prot_w_flat = prot_w[prot_mask]
        lig_w_flat = lig_w[atom_mask]

        attention_dict = {
            'residue_final_score': prot_w_flat,
            'atom_final_score': lig_w_flat,
            'residue_layer_scores': prot_w_flat,
            'clique_layer_scores': lig_w_flat,
            'drug_clique_index': mol_batch,
            'cluster_s': {},
            'prot_cross_attn_weights': prot_attn,
            'lig_cross_attn_weights': lig_attn,
            'prot_pool_weights': prot_w,
            'lig_pool_weights': lig_w,
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

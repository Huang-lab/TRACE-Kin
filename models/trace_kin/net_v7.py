"""
TRACE-Kin v7: Structure-Guided Embedding Distillation (SGED).

Novel architecture that decouples structural reasoning from embedding
aggregation through Pocket-Conditioned Embedding Readout (PCER).

Key contributions:
  1. PCER: GATv2 structure stream (512-d) produces per-residue pocket scores;
     embedding stream (4096-d) uses those scores as soft readout weights over
     uncompressed PLM features. Streams share NO parameters, coupled only
     through attention weights.
  2. Multi-Scale PCER: readout at three biologically meaningful scales —
     catalytic (top-16), pocket (top-64), global (all residues) — with
     learned scale gating.
  3. Mutation-Aware Gating: per-residue gate learned from raw MutaPLM
     embeddings modulates PCER attention, exploiting MutaPLM's mutation-
     specific training objective for kinetics prediction.

Self-contained file (no cross-imports from v5/v5t/v6c).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch_geometric.nn import GATv2Conv, global_add_pool
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.utils import degree, to_dense_batch
from torch_geometric.utils import softmax as pyg_softmax


# NOTE: torch_scatter and torch_sparse are deliberately NOT imported.
# v7 needed torch_scatter for a single sum-scatter (replaced by index_add_ below)
# and never used torch_sparse -- it arrived only because `from .layers import MLP`
# pulls in layers.py, which imports SparseTensor at module level.



class MLP(nn.Module):
    def __init__(self, dims, out_norm=False, in_norm=False, bias=True):
        super().__init__()
        self.FC_layers = nn.ModuleList(
                [nn.Linear(dims[i - 1], dims[i], bias=bias) for i in range(1, len(dims))])
        self.hidden_layers = len(dims) - 2
        self.out_norm, self.in_norm = out_norm, in_norm
        if out_norm:
            self.out_ln = nn.LayerNorm(dims[-1])
        if in_norm:
            self.in_ln = nn.LayerNorm(dims[0])

    def reset_parameters(self):
        for layer in self.FC_layers:
            layer.reset_parameters()
        if self.out_norm:
            self.out_ln.reset_parameters()
        if self.in_norm:
            self.in_ln.reset_parameters()
            
    def forward(self, x):
        y = self.in_ln(x) if self.in_norm else x
        for i in range(self.hidden_layers):
            y = F.relu(self.FC_layers[i](y))
        y = self.FC_layers[-1](y)
        
        return self.out_ln(y) if self.out_norm else y
# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

def _rbf(D: torch.Tensor, D_min: float = 0., D_max: float = 1.,
         D_count: int = 16) -> torch.Tensor:
    """Radial basis function embedding of scalar distances."""
    D = D.float().clamp(max=D_max)
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device).view(1, -1)
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
            cur = torch.zeros(num_nodes, device=edge_index.device).index_add_(0, row, msg)
            rw_landing[:, k] = cur

        return self.proj(rw_landing)


class DrugPNAEncoder(nn.Module):
    """Ligand GNN encoder using PNA."""

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
# Structure Stream: GPS Graph Transformer protein encoder
# ---------------------------------------------------------------------------

class GPSLayer(nn.Module):
    """GPS layer: local GATv2 message-passing + global self-attention + FFN.

    Combines graph-topology-aware local aggregation (respects contact-map edges)
    with global self-attention (captures long-range allosteric interactions).
    """

    def __init__(self, d_model: int, n_local_heads: int = 8,
                 n_global_heads: int = 8, edge_dim: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_global_heads = n_global_heads
        self.d_k = d_model // n_global_heads

        # --- Local: GATv2 message-passing on contact-map graph ---
        self.local_conv = GATv2Conv(
            in_channels=d_model,
            out_channels=d_model // n_local_heads,
            heads=n_local_heads,
            edge_dim=edge_dim,
            dropout=dropout,
            add_self_loops=True,
            concat=True,
        )

        # --- Global: Multi-head self-attention (Flash Attention) ---
        self.global_Q = nn.Linear(d_model, d_model)
        self.global_K = nn.Linear(d_model, d_model)
        self.global_V = nn.Linear(d_model, d_model)
        self.global_O = nn.Linear(d_model, d_model)
        self.global_dropout = nn.Dropout(dropout)

        # --- Fusion + FFN ---
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_feat: torch.Tensor, batch: torch.Tensor
                ) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            x: (N, d_model) node features (flat, all graphs concatenated)
            edge_index: (2, E) edges
            edge_feat: (E, d_model) edge features
            batch: (N,) graph membership

        Returns:
            x: (N, d_model) updated node features
            edge_attn: GATv2 edge attention weights for interpretability
        """
        residual = x

        # --- Local path: GATv2 ---
        local_out, edge_attn = self.local_conv(
            x, edge_index, edge_attr=edge_feat,
            return_attention_weights=True)

        # --- Global path: self-attention over dense batch ---
        x_dense, mask = to_dense_batch(x, batch)  # (B, L, D)
        B, L, D = x_dense.shape

        Q = self.global_Q(x_dense).view(B, L, self.n_global_heads, self.d_k).transpose(1, 2)
        K = self.global_K(x_dense).view(B, L, self.n_global_heads, self.d_k).transpose(1, 2)
        V = self.global_V(x_dense).view(B, L, self.n_global_heads, self.d_k).transpose(1, 2)

        # Key padding only — keep shape (B, 1, 1, L_k) so it broadcasts to (B, H, L_q, L_k).
        # Never materialize (B, H, L, L): that forces huge allocations when L is the
        # batch max sequence length (multi-GB, OOM on H100).
        # SDPA bool mask: True = position to ignore (here: padded keys).
        key_pad = ~mask  # (B, L) True = padded residue
        key_pad = key_pad.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)

        dropout_p = self.global_dropout.p if self.training else 0.0
        chunk_q = 512
        if L <= chunk_q:
            global_out_dense = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=key_pad, dropout_p=dropout_p)
        else:
            parts = []
            for t0 in range(0, L, chunk_q):
                t1 = min(t0 + chunk_q, L)
                parts.append(F.scaled_dot_product_attention(
                    Q[:, :, t0:t1, :], K, V, attn_mask=key_pad, dropout_p=dropout_p))
            global_out_dense = torch.cat(parts, dim=2)

        global_out_dense = global_out_dense.transpose(1, 2).contiguous().view(B, L, D)
        global_out_dense = self.global_O(global_out_dense)

        # Convert back to flat: extract valid positions using mask
        global_out = global_out_dense[mask]  # (N, D)

        # --- Fuse local + global with residual ---
        x = self.norm1(residual + local_out + global_out)

        # --- FFN with residual ---
        x = self.norm2(x + self.ffn(x))

        return x, edge_attn


class ProteinGPSEncoder(nn.Module):
    """GPS encoder: GATv2 (local) + Self-Attention (global) per layer.

    Combines contact-map graph inductive bias with global residue-residue
    attention for long-range allosteric reasoning.
    """

    def __init__(self, d_model: int, n_layers: int = 3, n_local_heads: int = 8,
                 n_global_heads: int = 8, rbf_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.rbf_dim = rbf_dim
        self.edge_proj = nn.Linear(rbf_dim, d_model)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(GPSLayer(
                d_model=d_model,
                n_local_heads=n_local_heads,
                n_global_heads=n_global_heads,
                edge_dim=d_model,
                dropout=dropout,
            ))

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor, batch: torch.Tensor
                ) -> tuple[torch.Tensor, list]:
        edge_attr = _rbf(edge_weight, D_max=1.0, D_count=self.rbf_dim)
        edge_feat = self.edge_proj(edge_attr)

        edge_attns = []
        for layer in self.layers:
            x, edge_attn = grad_checkpoint(
                layer, x, edge_index, edge_feat, batch,
                use_reentrant=False)
            edge_attns.append(edge_attn)

        x = self.final_norm(x)
        return x, edge_attns


# ---------------------------------------------------------------------------
# Embedding Stream: Multi-Scale PCER + Mutation-Aware Gate
# ---------------------------------------------------------------------------

class MutationAwareGate(nn.Module):
    """Per-residue gate learned from raw PLM embeddings.

    Exploits MutaPLM's mutation-specific training: residues where mutations
    have large functional impact get higher gate values, focusing the PCER
    readout on mutation-sensitive positions.
    """

    def __init__(self, prot_evo_channels: int):
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Linear(prot_evo_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, raw_emb: torch.Tensor) -> torch.Tensor:
        """raw_emb: (N, 4096) flat per-residue. Returns (N, 1) gate in [0,1]."""
        return torch.sigmoid(self.gate_proj(raw_emb))


class MultiScalePCER(nn.Module):
    """Pocket-Conditioned Embedding Readout at three biological scales.

    Uses detached pocket scores from the structure stream as soft readout
    weights over uncompressed PLM embeddings, with per-scale compression
    and learned scale gating.
    """

    def __init__(self, prot_evo_channels: int, d_model: int,
                 k_catalytic: int = 16, k_pocket: int = 64):
        super().__init__()
        self.k_catalytic = k_catalytic
        self.k_pocket = k_pocket

        self.compress_catalytic = nn.Sequential(
            nn.Linear(prot_evo_channels, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.compress_pocket = nn.Sequential(
            nn.Linear(prot_evo_channels, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.compress_global = nn.Sequential(
            nn.Linear(prot_evo_channels, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

        self.scale_gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 3),
        )

    def forward(self, raw_emb_dense: torch.Tensor, pocket_score: torch.Tensor,
                mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            raw_emb_dense: (B, L, 4096) — uncompressed PLM embeddings (dense batch)
            pocket_score: (B, L) — detached per-residue pocket relevance scores
            mask: (B, L) — padding mask

        Returns:
            pcer_pool: (B, d_model) — fused multi-scale readout
            scale_weights: (B, 3) — interpretable scale importance
        """
        B, L, E = raw_emb_dense.shape

        # Mask-aware softmax for each scale
        score = pocket_score.masked_fill(~mask, float('-inf'))

        # --- Catalytic scale: top-K1 residues ---
        K1 = min(self.k_catalytic, L)
        _, cat_idx = pocket_score.masked_fill(~mask, -1e9).topk(K1, dim=-1)
        cat_mask = torch.zeros(B, L, device=mask.device, dtype=torch.bool)
        cat_mask.scatter_(1, cat_idx, True)
        cat_mask = cat_mask & mask
        cat_logits = score.masked_fill(~cat_mask, float('-inf'))
        cat_w = F.softmax(cat_logits, dim=-1)
        cat_pool = (raw_emb_dense * cat_w.unsqueeze(-1)).sum(1)  # (B, 4096)
        feat_cat = self.compress_catalytic(cat_pool)  # (B, d_model)

        # --- Pocket scale: top-K2 residues ---
        K2 = min(self.k_pocket, L)
        _, pkt_idx = pocket_score.masked_fill(~mask, -1e9).topk(K2, dim=-1)
        pkt_mask = torch.zeros(B, L, device=mask.device, dtype=torch.bool)
        pkt_mask.scatter_(1, pkt_idx, True)
        pkt_mask = pkt_mask & mask
        pkt_logits = score.masked_fill(~pkt_mask, float('-inf'))
        pkt_w = F.softmax(pkt_logits, dim=-1)
        pkt_pool = (raw_emb_dense * pkt_w.unsqueeze(-1)).sum(1)
        feat_pkt = self.compress_pocket(pkt_pool)

        # --- Global scale: all residues ---
        global_w = F.softmax(score, dim=-1)
        global_pool = (raw_emb_dense * global_w.unsqueeze(-1)).sum(1)
        feat_global = self.compress_global(global_pool)

        # --- Learned scale gating ---
        scale_input = torch.cat([feat_cat, feat_pkt, feat_global], dim=-1)
        scale_weights = F.softmax(self.scale_gate(scale_input), dim=-1)  # (B, 3)

        pcer_pool = (scale_weights[:, 0:1] * feat_cat +
                     scale_weights[:, 1:2] * feat_pkt +
                     scale_weights[:, 2:3] * feat_global)

        return pcer_pool, scale_weights


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class TraceKinV7(nn.Module):
    """TRACE-Kin v7: Structure-Guided Embedding Distillation (SGED).

    Dual-stream architecture:
      - Structure Stream (512-d): GPS Graph Transformer (GATv2 local + global
        self-attention per layer) on contact map -> pocket identification via
        cross-attention with ligand -> structural pooling + MMCAF
      - Embedding Stream (4096-d): raw MutaPLM -> mutation-aware gate ->
        multi-scale PCER using detached pocket scores from structure stream
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
        n_global_heads: int = 8,
        gat_rbf_dim: int = 16,
        graph_pe_rwse_steps: int = 16,
        pocket_k: int = 64,
        pcer_k_catalytic: int = 16,
        pcer_k_pocket: int = 64,
        use_mutation_gate: bool = True,
        use_graph_transformer: bool = True,
        n_pred_heads: int = 3,
        emb_recon_weight: float = 0.1,
        mol_in_channels: int = 43,
        n_drug_pna_layers: int = 3,
        # Ligand positional/structural encoding: "none" (v7 baseline) | "input"
        # (RWSE added once) | "lspe" (learnable p, own PNA path). See
        # models/trace_kin/ligand_encoder.py.
        mol_pe_mode: str = "none",
        mol_pe_steps: int = 8,
        mol_pe_dim: int = 16,
        n_cross_heads: int = 8,
        chembert_dim: int = 768,
        dropout: float = 0.1,
        regression_head: bool = True,
        classification_head: bool = False,
        multiclassification_head: int = 0,
        device: str = "cuda:0",
        # Compat params (unused, accepted for trainer compatibility)
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
        n_mamba_layers: int = 4,
        mamba_expand: int = 2,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        n_transformer_layers: int = 4,
        n_self_attn_heads: int = 8,
        ffn_expand: int = 4,
        input_dropout: float = 0.0,
        n_gat_layers_compat: int = 3,
        gat_rbf_dim_compat: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.prot_evo_channels = prot_evo_channels
        self.pocket_k = pocket_k
        self.use_mutation_gate = use_mutation_gate
        self.regression_head = regression_head
        self.classification_head = classification_head
        self.multiclassification_head = multiclassification_head
        self.device = device
        self.learnable_aux_loss = False
        self._n_pred_heads = n_pred_heads
        self.emb_recon_weight = emb_recon_weight

        # =====================================================================
        # STRUCTURE STREAM (d=512)
        # =====================================================================
        self.prot_proj = nn.Linear(prot_evo_channels, d_model)
        self.prot_proj_norm = nn.LayerNorm(d_model)
        self.graph_pe = GraphStructuralEncoding(d_model, rwse_steps=graph_pe_rwse_steps)

        self.prot_gps = ProteinGPSEncoder(
            d_model, n_layers=n_gat_layers, n_local_heads=n_gat_heads,
            n_global_heads=n_global_heads, rbf_dim=gat_rbf_dim, dropout=dropout)

        # Pocket identification cross-attention (structure stream)
        self.pocket_cross_attn = MultiHeadCrossAttention(d_model, n_cross_heads, dropout)

        # MMCAF at pocket level (structure stream)
        self.mmcaf_prot = MultiHeadCrossAttention(d_model, n_cross_heads, dropout)
        self.mmcaf_lig = MultiHeadCrossAttention(d_model, n_cross_heads, dropout)

        # Structure stream pooling
        self.struct_pocket_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.ReLU(), nn.Linear(d_model // 4, 1))
        self.struct_global_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.ReLU(), nn.Linear(d_model // 4, 1))
        self.struct_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Linear(d_model, 1), nn.Sigmoid())

        # =====================================================================
        # EMBEDDING STREAM (d=4096, never compressed during aggregation)
        # =====================================================================
        self.emb_norm = nn.LayerNorm(prot_evo_channels)

        if use_mutation_gate:
            self.mut_gate = MutationAwareGate(prot_evo_channels)

        self.pcer = MultiScalePCER(
            prot_evo_channels, d_model,
            k_catalytic=pcer_k_catalytic, k_pocket=pcer_k_pocket)

        # Auxiliary: embedding reconstruction decoder (forces PCER to preserve info)
        self.emb_decoder = nn.Linear(d_model, prot_evo_channels)

        # =====================================================================
        # LIGAND ENCODER
        # =====================================================================
        self.mol_pe_mode = mol_pe_mode
        from .ligand_encoder import LigandEncoder
        self.drug_encoder = LigandEncoder(
                mol_in_channels, d_model, mol_deg,
                n_layers=n_drug_pna_layers, heads=heads, dropout=dropout,
                pe_mode=mol_pe_mode, pe_steps=mol_pe_steps, pe_dim=mol_pe_dim)
            
        self.mol_context_proj = nn.Linear(chembert_dim, d_model)

        self.mol_pool_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.ReLU(), nn.Linear(d_model // 4, 1))

        # =====================================================================
        # PREDICTION HEAD (ensemble of n_pred_heads for variance reduction)
        # =====================================================================
        # feat = struct_pool (d_model) + pcer_pool (d_model) + lig_pool (d_model)
        feat_dim = d_model * 3
        if regression_head:
            self.reg_heads = nn.ModuleList([
                MLP([feat_dim, d_model, d_model // 2, 1])
                for _ in range(self._n_pred_heads)
            ])
        if classification_head:
            self.cls_out = MLP([feat_dim, d_model, 2])
        if multiclassification_head:
            self.mcls_out = MLP([feat_dim, d_model, multiclassification_head])

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
        """Attention-weighted pooling over a dense (B, L, D) tensor."""
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

        raw_evo = residue_evo_x.float()

        # =================================================================
        # STRUCTURE STREAM
        # =================================================================
        prot_h = self.prot_proj_norm(self.prot_proj(raw_evo))
        graph_pe = self.graph_pe(residue_edge_index, num_nodes=prot_h.size(0))
        prot_h = prot_h + graph_pe

        prot_h, gat_edge_attns = self.prot_gps(
            prot_h, residue_edge_index, residue_edge_weight, prot_batch)

        # =================================================================
        # LIGAND ENCODER
        # =================================================================

        # DrugPNAEncoder returns 2 values, LigandEncoder returns 3 (the extra one
        # is the learnable positional tensor p, used only in lspe mode).
        _enc = self.drug_encoder(mol_x, mol_x_feat, bond_x, atom_edge_index, mol_batch)
        atom_h, mol_pe = (_enc[0], _enc[2]) if len(_enc) == 3 else (_enc[0], None)
       
        if chembert_fp is not None:
            ctx = self.mol_context_proj(chembert_fp.float())
            if ctx.dim() == 3:
                ctx = ctx.squeeze(1)
            atom_h = atom_h + ctx[mol_batch]

        # =================================================================
        # DENSE BATCHING
        # =================================================================
        prot_h_dense, prot_mask = to_dense_batch(prot_h, prot_batch)
        atom_h_dense, atom_mask = to_dense_batch(atom_h, mol_batch)

        B, L_prot, D = prot_h_dense.shape

        # =================================================================
        # POCKET IDENTIFICATION (Structure Stream)
        # =================================================================
        _, pocket_attn = self.pocket_cross_attn(
            prot_h_dense, atom_h_dense, atom_h_dense,
            query_mask=prot_mask, key_mask=atom_mask)
        # Per-residue pocket score: mean over heads and ligand atoms
        pocket_score = pocket_attn.mean(dim=(1, 3))  # (B, L_prot)
        pocket_score = pocket_score.masked_fill(~prot_mask, 0.0)

        # =================================================================
        # STRUCTURE STREAM: MMCAF + hierarchical pooling
        # =================================================================
        K = min(self.pocket_k, L_prot)
        _, pocket_idx = pocket_score.topk(K, dim=-1)
        pocket_dense = torch.gather(
            prot_h_dense, 1,
            pocket_idx.unsqueeze(-1).expand(-1, -1, D))
        pocket_dense_mask = torch.gather(prot_mask, 1, pocket_idx)

        pocket_attended, mmcaf_prot_attn = self.mmcaf_prot(
            pocket_dense, atom_h_dense, atom_h_dense,
            query_mask=pocket_dense_mask, key_mask=atom_mask)

        lig_attended, mmcaf_lig_attn = self.mmcaf_lig(
            atom_h_dense, pocket_dense, pocket_dense,
            query_mask=atom_mask, key_mask=pocket_dense_mask)

        # Structure stream pooling
        struct_pocket_pool, struct_pocket_w = self._pool_dense(
            pocket_attended, pocket_dense_mask, self.struct_pocket_pool)
        struct_global_pool, struct_global_w = self._pool_dense(
            prot_h_dense, prot_mask, self.struct_global_pool)

        gate = self.struct_gate(torch.cat([struct_pocket_pool, struct_global_pool], dim=-1))
        struct_pool = gate * struct_pocket_pool + (1.0 - gate) * struct_global_pool

        # Ligand pool
        mol_pool, lig_w = self._pool_dense(
            lig_attended, atom_mask, self.mol_pool_mlp)

        # =================================================================
        # EMBEDDING STREAM: Multi-Scale PCER
        # =================================================================
        raw_normed = self.emb_norm(raw_evo)
        raw_dense, _ = to_dense_batch(raw_normed, prot_batch)

        # Detach pocket score: structure stream trains via struct_pool path,
        # embedding stream trains via PCER compress layers only.
        pcer_score = pocket_score.detach()

        # Mutation-aware gating: modulate pocket scores by per-residue
        # mutation sensitivity learned from raw PLM embeddings
        if self.use_mutation_gate:
            mut_gate_flat = self.mut_gate(raw_normed)  # (N, 1)
            mut_gate_dense, _ = to_dense_batch(mut_gate_flat.squeeze(-1), prot_batch)
            pcer_score = pcer_score * mut_gate_dense

        pcer_pool, scale_weights = self.pcer(raw_dense, pcer_score, prot_mask)

        # Auxiliary embedding reconstruction loss: PCER pool -> reconstruct
        # the mean-pooled raw embedding (forces information preservation)
        emb_recon_target = (raw_dense * prot_mask.unsqueeze(-1).float()).sum(1) / prot_mask.sum(1, keepdim=True).float()
        emb_recon_pred = self.emb_decoder(pcer_pool)
        emb_recon_loss = self.emb_recon_weight * F.mse_loss(emb_recon_pred, emb_recon_target.detach())

        # =================================================================
        # PREDICTION
        # =================================================================
        feat = torch.cat([mol_pool, struct_pool, pcer_pool], dim=-1)

        if self.regression_head:
            reg_pred = torch.stack([h(feat) for h in self.reg_heads]).mean(0)
        if self.classification_head:
            cls_pred = self.cls_out(feat)
        if self.multiclassification_head:
            mcls_pred = self.mcls_out(feat)

        # =================================================================
        # INTERPRETABILITY OUTPUTS
        # =================================================================
        pocket_mask = torch.zeros(B, L_prot, device=prot_h.device, dtype=torch.bool)
        pocket_mask.scatter_(1, pocket_idx, True)
        pocket_mask = pocket_mask & prot_mask

        full_prot_w = torch.zeros(B, L_prot, device=prot_h.device)
        full_prot_w.scatter_(1, pocket_idx, struct_pocket_w)
        full_prot_w = torch.where(pocket_mask, full_prot_w, struct_global_w)
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
            'struct_gate': gate,
            'scale_weights': scale_weights,
            'pcer_score': pcer_score,
            'mmcaf_prot_attn': mmcaf_prot_attn,
            'mmcaf_lig_attn': mmcaf_lig_attn,
            'gat_edge_attns': gat_edge_attns,
            'mol_feature': mol_pool,
            'prot_feature': struct_pool,
            'interaction_fingerprint': feat,
            'drug_atom_index': mol_batch,
            'protein_residue_index': prot_batch,
        }

        return reg_pred, cls_pred, mcls_pred, emb_recon_loss, zero_loss, zero_loss, attention_dict

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

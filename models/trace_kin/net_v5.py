"""TRACE-Kin v5: Bidirectional Graph-Mamba + Cross-Attention.

Architecture for mutation-aware enzyme kinetic parameter prediction.

1. Bidirectional Mamba processes per-residue MutaPLM embeddings — O(L) for
   long enzymes, captures long-range dependencies beyond GNN receptive field.
2. Graph-structure injection via gated neighbor aggregation from the protein
   contact-map graph after each Mamba block.
3. Cross-attention between enzyme residues and ligand atoms produces
   interpretable attention maps for site-of-metabolism identification.
4. Substrate-conditioned pooling: learned attention weights depend on
   cross-attention output — the model learns which residues matter for
   each specific substrate (unlike RF's blind mean-pool).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import degree, to_dense_batch
from torch_geometric.utils import softmax as pyg_softmax
from torch_scatter import scatter

from .layers import MLP


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class GraphStructuralEncoding(nn.Module):
    """Random-walk structural encoding from the protein contact-map.

    Computes k-step random-walk landing probabilities per node and projects
    to d_model. Gives structural (3D-aware) positional information without
    requiring explicit 3D coordinates.
    """

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


class BiMambaBlock(nn.Module):
    """Bidirectional selective state-space block for protein sequences.

    Processes residues in both forward and backward directions using the
    S6 selective scan mechanism, then combines via learned projection.
    """

    def __init__(self, d_model: int, expand: int = 2, d_state: int = 16,
                 d_conv: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state

        # Forward SSM
        self.fwd_in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.fwd_conv = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                  padding=d_conv - 1, groups=self.d_inner)
        self.fwd_x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.fwd_dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.fwd_out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Backward SSM
        self.bwd_in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.bwd_conv = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                  padding=d_conv - 1, groups=self.d_inner)
        self.bwd_x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.bwd_dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.bwd_out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Fixed A matrix (log-spaced)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.d_inner, -1)
        self.register_buffer("A_log", torch.log(A))

        self.combine = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _ssm_scan(self, x, in_proj, conv, x_proj, dt_proj, out_proj):
        """Selective scan in one direction. (B, L, d_model) -> (B, L, d_model)"""
        B, L, _ = x.shape

        xz = in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        x_conv = x_inner.transpose(1, 2)
        x_conv = conv(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        bc = x_proj(x_conv)
        B_in, C_in = bc.chunk(2, dim=-1)

        dt = F.softplus(dt_proj(x_conv))
        A = -torch.exp(self.A_log)

        d_inner = x_conv.shape[-1]
        d_state = B_in.shape[-1]
        h = torch.zeros(B, d_inner, d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1)
            B_t = B_in[:, t, :].unsqueeze(1)
            C_t = C_in[:, t, :]
            x_t = x_conv[:, t, :].unsqueeze(-1)

            dA = torch.exp(A.unsqueeze(0) * dt_t)
            dB = dt_t * B_t
            h = dA * h + dB * x_t
            y_t = (h * C_t.unsqueeze(1)).sum(-1)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        y = y * F.silu(z)
        return out_proj(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Bidirectional scan with residual. (B, L, d_model) -> (B, L, d_model)"""
        residual = x
        fwd_out = self._ssm_scan(x, self.fwd_in_proj, self.fwd_conv,
                                  self.fwd_x_proj, self.fwd_dt_proj, self.fwd_out_proj)
        bwd_out = self._ssm_scan(x.flip(1), self.bwd_in_proj, self.bwd_conv,
                                  self.bwd_x_proj, self.bwd_dt_proj, self.bwd_out_proj).flip(1)
        combined = self.combine(torch.cat([fwd_out, bwd_out], dim=-1))
        return self.norm(residual + self.dropout(combined))


class GraphGatedAggregation(nn.Module):
    """Graph-gated neighbor aggregation from contact-map edges.

    Softer than MinCut clustering: no hard assignments, no auxiliary losses.
    Residues exchange information through learnable attention + gating.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.attn_proj = nn.Linear(d_model * 2, 1)
        self.gate_proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        N = h.size(0)

        h_i = h[row]
        h_j = h[col]
        alpha = self.attn_proj(torch.cat([h_i, h_j], dim=-1)).squeeze(-1)
        alpha = pyg_softmax(alpha, row, num_nodes=N)

        agg = scatter(alpha.unsqueeze(-1) * h_j, row, dim=0,
                      dim_size=N, reduce='sum')
        gate = torch.sigmoid(self.gate_proj(torch.cat([h, agg], dim=-1)))
        return self.norm(h + gate * agg)


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
        """Returns (output, attn_weights)."""
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
    """Ligand GNN encoder using PNA (same validated architecture as v1 drug branch)."""

    def __init__(self, mol_in_channels: int, d_model: int, mol_deg: torch.Tensor,
                 n_layers: int = 3, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        from .pna import PNAConv
        from torch_geometric.nn.norm import GraphNorm

        self.atom_type_encoder = nn.Embedding(20, d_model)
        self.atom_feat_encoder = MLP([mol_in_channels, d_model * 2, d_model], out_norm=True)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(PNAConv(
                in_channels=d_model, out_channels=d_model,
                aggregators=['mean', 'min', 'max', 'std'],
                scalers=['identity', 'amplification', 'linear'],
                deg=mol_deg, pre_layers=2, post_layers=1, towers=heads,
            ))
            self.norms.append(GraphNorm(d_model))

        self.dropout = nn.Dropout(dropout)
        self.attn_pool = nn.Linear(d_model, 1)

    def forward(self, mol_x, mol_x_feat, bond_x, atom_edge_index, mol_batch):
        atom_h = self.atom_type_encoder(mol_x.squeeze()) + self.atom_feat_encoder(mol_x_feat)
        for conv, norm in zip(self.convs, self.norms):
            atom_h = conv(atom_h, bond_x, atom_edge_index)
            atom_h = norm(atom_h, mol_batch)
            atom_h = self.dropout(F.relu(atom_h))

        attn = pyg_softmax(self.attn_pool(atom_h).squeeze(-1), mol_batch)
        mol_pool = global_add_pool(atom_h * attn.unsqueeze(-1), mol_batch)
        return atom_h, mol_pool


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class TraceKinV5(nn.Module):
    """TRACE-Kin v5: Bidirectional Graph-Mamba + Cross-Attention.

    Protein: MutaPLM per-residue → projection → BiMamba with graph-gated
    aggregation → substrate-conditioned via cross-attention.
    Ligand: PNA GNN + MoLFormer context.
    Interaction: Multi-head cross-attention (interpretable).
    """

    def __init__(
        self,
        mol_deg: torch.Tensor,
        prot_deg: torch.Tensor,
        prot_evo_channels: int = 4096,
        d_model: int = 512,
        n_mamba_layers: int = 4,
        mamba_expand: int = 2,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
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
        # Accepted for config compatibility with v1 (unused)
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
    ):
        super().__init__()
        self.d_model = d_model
        self.prot_evo_channels = prot_evo_channels
        self.regression_head = regression_head
        self.classification_head = classification_head
        self.multiclassification_head = multiclassification_head
        self.device = device
        self.learnable_aux_loss = False

        # Protein encoder
        self.prot_proj = nn.Linear(prot_evo_channels, d_model)
        self.prot_proj_norm = nn.LayerNorm(d_model)
        self.graph_pe = GraphStructuralEncoding(d_model, rwse_steps=graph_pe_rwse_steps)

        self.mamba_blocks = nn.ModuleList([
            BiMambaBlock(d_model, expand=mamba_expand, d_state=mamba_d_state,
                         d_conv=mamba_d_conv, dropout=dropout)
            for _ in range(n_mamba_layers)
        ])
        self.graph_gates = nn.ModuleList([
            GraphGatedAggregation(d_model)
            for _ in range(n_mamba_layers)
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
        """Forward pass. Returns same tuple as v1/v4 for trainer compatibility."""
        reg_pred = None
        cls_pred = None
        mcls_pred = None
        zero_loss = residue_evo_x.new_zeros(())

        # === Protein: project + graph PE + BiMamba + graph gate ===
        prot_h = self.prot_proj_norm(self.prot_proj(residue_evo_x.float()))
        graph_pe = self.graph_pe(residue_edge_index, num_nodes=prot_h.size(0))
        prot_h = prot_h + graph_pe

        prot_h_dense, prot_mask = to_dense_batch(prot_h, prot_batch)

        for mamba_block, graph_gate in zip(self.mamba_blocks, self.graph_gates):
            prot_h_dense = mamba_block(prot_h_dense)
            prot_h_flat = prot_h_dense[prot_mask]
            prot_h_flat = graph_gate(prot_h_flat, residue_edge_index, prot_batch)
            prot_h_dense, _ = to_dense_batch(prot_h_flat, prot_batch)

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

        attention_dict = {
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
                elif "A_log" in pn:
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

"""GRIT ligand encoder — Graph Inductive bias Transformer, no message passing.

Ma, Lin, Lim, Romero-Soriano, Dokania, Coates, Torr & Lim,
"Graph Inductive Biases in Transformers without Message Passing", ICML 2023.

Drop-in alternative to LigandEncoder: same constructor arguments where they
overlap and the SAME return contract `(atom_h, mol_pool, p)`, so net_v7 needs
only a dispatch line.

WHY RRWP IS FREE HERE
---------------------
GRIT's positional encoding is RRWP: the FULL k-step transition matrix
(D^-1 A)^k, used as a *pair* feature. `ligand_encoder.atom_rwse` already builds
exactly that matrix and then throws away everything but `.diagonal()`:

    M = P @ P
    out = [M.diagonal()]          # <- RWSE keeps this
    for _ in range(steps - 1):
        M = M @ P                 # <- M IS RRWP at step k
        out.append(M.diagonal())

So RWSE = diag(RRWP). Keeping M costs no extra matmuls; only memory for the
pair list, which is small at molecular scale (~18 atoms -> ~324 pairs).

WHAT MAKES IT "WITHOUT MESSAGE PASSING"
---------------------------------------
Attention runs over ALL pairs within a molecule, not bonded neighbours, and
edge representations are carried and updated across layers. So this is not a
`conv` and cannot sit behind a make_conv() registry -- it replaces the layer
stack outright. Bond types are still used: they are added into the initial pair
embedding at the positions that correspond to real bonds.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import softmax as pyg_softmax


# --------------------------------------------------------------------------- #
# RRWP                                                                          #
# --------------------------------------------------------------------------- #
def atom_rrwp(edge_index: torch.Tensor, num_nodes: int, steps: int
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Relative random-walk probabilities.

    Returns
    -------
    node_pe    (N, steps)   diag((D^-1 A)^k) for k = 2..steps+1  -- identical to
                            atom_rwse, so the two arms stay comparable.
    pair_attr  (P, steps)   [(D^-1 A)^k]_ij for the retained pairs.
    pair_index (2, P)       the retained pairs, dst-major ordering not assumed.

    Pairs are those with nonzero RRWP at any k, plus self-loops. A batch is
    block diagonal and a random walk cannot leave its molecule, so the nonzero
    pattern is automatically within-graph -- no `batch` vector needed. Pairs
    further apart than `steps` hops carry no signal and are dropped, which is
    what keeps P small.
    """
    A = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
    A[edge_index[0], edge_index[1]] = 1.0
    deg = A.sum(dim=1).clamp(min=1)
    P = A / deg.unsqueeze(1)

    # Reference add_full_rrwp: [I, A, A^2, ..., A^(K-1)] with add_identity=True.
    # The identity marks the self-pair and A^1 marks a direct bond -- the two
    # most informative PAIR channels. They are useless on the diagonal (always
    # 1 and 0), which is why atom_rwse skips them, but dropping them from the
    # pair representation loses exactly the signal GRIT attends over.
    eye = torch.eye(num_nodes, device=A.device)
    mats = [eye, P]
    M = P
    for _ in range(steps - 2):
        M = M @ P
        mats.append(M)

    stacked = torch.stack(mats, dim=-1)          # (N, N, steps)
    # Node PE keeps the k >= 2 slice so it stays identical to atom_rwse and the
    # PNA and GRIT arms share the same positional signal.
    node_pe = torch.stack([m.diagonal() for m in mats[2:]], dim=-1)   # (N, steps-2)

    occupied = stacked.abs().sum(-1) > 0
    occupied = occupied | torch.eye(num_nodes, dtype=torch.bool, device=A.device)
    pair_index = occupied.nonzero(as_tuple=False).t().contiguous()  # (2, P)
    pair_attr = stacked[pair_index[0], pair_index[1]]               # (P, steps)
    return node_pe, pair_attr, pair_index


def _signed_sqrt(x: torch.Tensor) -> torch.Tensor:
    """rho in GRIT Eq. 5, exactly as grit_layer.py:94:
        sqrt(relu(x)) - sqrt(relu(-x))
    Note the gradient is infinite at x=0; the reference accepts this."""
    return torch.sqrt(torch.relu(x)) - torch.sqrt(torch.relu(-x))


# --------------------------------------------------------------------------- #
# GRIT attention layer                                                          #
# --------------------------------------------------------------------------- #
class GRITLayer(nn.Module):
    """One GRIT block. Updates BOTH node and pair representations.

        e_hat_ij = rho( (x_i W_Q + x_j W_K) * (W_Ew e_ij) + W_Eb e_ij )
        alpha_ij = softmax_j( w_A . e_hat_ij )
        x_i      = sum_j alpha_ij ( x_j W_V + W_Ev e_hat_ij )
        e_ij     = e_hat_ij

    followed by GRIT's degree scaler, BatchNorm, and an FFN, each residual.
    """

    def __init__(self, d_model: int, e_dim: int, heads: int = 8,
                 dropout: float = 0.0, attn_dropout: float = 0.2,
                 ffn_mult: int = 2,
                 clamp: Optional[float] = 5.0, edge_enhance: bool = True,
                 act: Optional[str] = "relu") -> None:
        super().__init__()
        self.clamp = clamp
        self.edge_enhance = edge_enhance
        # grit_layer.py:75-78 and :97 -- an activation applied to e_hat AFTER
        # E_b, before both the attention contraction and the VeRow path.
        # cfg.attn.get("act", "relu") and the shipped zinc config both use relu.
        self.act = {"relu": nn.ReLU(), "gelu": nn.GELU(),
                    "elu": nn.ELU(), None: nn.Identity()}[act]
        assert d_model % heads == 0, "heads must divide d_model"
        self.h, self.dh = heads, d_model // heads

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.WEw = nn.Linear(e_dim, d_model)      # multiplicative pair term
        self.WEb = nn.Linear(e_dim, d_model)      # additive pair term
        self.wA = nn.Parameter(torch.randn(heads, self.dh) * (self.dh ** -0.5))
        if edge_enhance:
            # grit_layer.py:81 -- VeRow is (out_dim, num_heads, out_dim) and is
            # contracted per head: "nhd,dhc->nhc". A flat Linear over H*D would
            # mix heads, which the reference deliberately does not do.
            self.VeRow = nn.Parameter(torch.zeros(self.dh, heads, self.dh))
            nn.init.xavier_normal_(self.VeRow)

        self.out_x = nn.Linear(d_model, d_model)
        self.out_e = nn.Linear(d_model, e_dim)

        # GRIT degree scaler: x <- t1*x + t2*x*log(1+deg)
        self.theta1 = nn.Parameter(torch.ones(d_model))
        self.theta2 = nn.Parameter(torch.zeros(d_model))

        self.bn_x1 = nn.BatchNorm1d(d_model)
        self.bn_x2 = nn.BatchNorm1d(d_model)
        self.bn_e = nn.BatchNorm1d(e_dim)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model * ffn_mult, d_model))
        # Reference zinc-GRIT-RRWP.yaml uses TWO rates: dropout: 0.0 on the
        # residual/FFN path and attn_dropout: 0.2 on the attention weights.
        self.drop = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor, e: torch.Tensor,
                pair_index: torch.Tensor, log_deg: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x           (N, d_model)
        e           (P, e_dim)
        pair_index  (2, P)   row 0 = source j, row 1 = destination i
        log_deg     (N, 1)   log(1 + degree), precomputed once
        """
        N, H, DH = x.size(0), self.h, self.dh
        src, dst = pair_index[0], pair_index[1]

        q = self.WQ(x).view(N, H, DH)
        k = self.WK(x).view(N, H, DH)
        v = self.WV(x).view(N, H, DH)
        ew = self.WEw(e).view(-1, H, DH)
        eb = self.WEb(e).view(-1, H, DH)

        # Reference order (grit_layer.py:93-97): multiply by E_w, THEN signed
        # sqrt, THEN add E_b. Folding E_b inside the sqrt changes the function.
        e_hat = self.act(_signed_sqrt((q[dst] + k[src]) * ew) + eb)   # (P, H, DH)

        score = (e_hat * self.wA.unsqueeze(0)).sum(-1)          # (P, H)
        if self.clamp is not None:                              # grit_layer.py:106
            score = score.clamp(-self.clamp, self.clamp)
        alpha = pyg_softmax(score, dst, num_nodes=N)            # per destination
        alpha = self.attn_drop(alpha)

        # grit_layer.py:115-122. The value path is weighted by attention ALONE;
        # the edge contribution enters as a separate per-head aggregation
        # projected by VeRow. Following that order also runs the projection on
        # (N, H, DH) rather than (P, H, DH), and P >> N.
        agg = x.new_zeros(N, H, DH)
        agg.index_add_(0, dst, v[src] * alpha.unsqueeze(-1))

        if self.edge_enhance:
            rowV = x.new_zeros(N, H, DH)
            rowV.index_add_(0, dst, e_hat * alpha.unsqueeze(-1))
            rowV = torch.einsum("nhd,dhc->nhc", rowV, self.VeRow)
            agg = agg + rowV

        agg = agg.reshape(N, H * DH)

        # degree scaler, then node residual
        agg = self.theta1 * agg + self.theta2 * agg * log_deg
        x = self.bn_x1(x + self.drop(self.out_x(agg)))
        x = self.bn_x2(x + self.drop(self.ffn(x)))

        # pair residual
        e = self.bn_e(e + self.drop(self.out_e(e_hat.reshape(-1, H * DH))))
        return x, e


# --------------------------------------------------------------------------- #
# Encoder                                                                       #
# --------------------------------------------------------------------------- #
class _MLP(nn.Module):
    """Local copy, mirroring ligand_encoder._MLP so the two arms share
    featurisation exactly."""

    def __init__(self, dims, out_norm=False):
        super().__init__()
        self.fc = nn.ModuleList([nn.Linear(dims[i - 1], dims[i]) for i in range(1, len(dims))])
        self.hidden = len(dims) - 2
        self.ln = nn.LayerNorm(dims[-1]) if out_norm else None

    def forward(self, x):
        for i in range(self.hidden):
            x = F.relu(self.fc[i](x))
        x = self.fc[-1](x)
        return self.ln(x) if self.ln is not None else x


class GRITLigandEncoder(nn.Module):
    """GRIT stack at its own operating point, projected to net_v7's width.

    Returns (atom_h, mol_pool, None) -- the third slot keeps net_v7's 3-tuple
    unpacking valid. GRIT carries position in the PAIR representation, updated
    every layer, so there is no separate positional stream to fold back and
    `pe_fold_norm` does not apply.

    WIDTH. The reference runs DEEP AND NARROW: 10 layers at dim_hidden 64,
    8 heads (head dim 8), ksteps 21 -- see zinc-GRIT-RRWP.yaml. net_v7 needs
    atom_h at d_model for pocket attention and MMCAF, so the stack runs at
    `grit_dim` and a final Linear lifts grit_dim -> d_model. That keeps GRIT at
    the operating point its benchmark numbers come from instead of forcing it
    to 512 wide and 3 deep.
    """

    def __init__(self, mol_in_channels: int, d_model: int, mol_deg=None,
                 n_layers: int = 10, heads: int = 8,
                 grit_dim: int = 64, e_dim: Optional[int] = None,
                 pe_steps: int = 21, dropout: float = 0.0,
                 attn_dropout: float = 0.2, pe_raw_norm: str = "none",
                 clamp: Optional[float] = 5.0, edge_enhance: bool = True,
                 act: Optional[str] = "relu", **_ignored) -> None:
        super().__init__()
        if pe_steps < 3:
            raise ValueError("pe_steps must be >= 3 (identity, k=1, and k>=2)")
        e_dim = grit_dim if e_dim is None else e_dim     # reference: edge dim == dim_hidden
        self.pe_steps = pe_steps

        self.atom_type_encoder = nn.Embedding(20, grit_dim)
        self.atom_feat_encoder = _MLP([mol_in_channels, grit_dim * 2, grit_dim],
                                      out_norm=True)
        self.bond_encoder = nn.Embedding(5, e_dim)

        if pe_raw_norm == "batch":
            self.pe_norm = nn.BatchNorm1d(pe_steps - 2, eps=1e-3)
            self.pair_norm = nn.BatchNorm1d(pe_steps, eps=1e-3)
        elif pe_raw_norm == "layer":
            self.pe_norm = nn.LayerNorm(pe_steps - 2)
            self.pair_norm = nn.LayerNorm(pe_steps)
        elif pe_raw_norm == "none":
            self.pe_norm = self.pair_norm = None
        else:
            raise ValueError(f"pe_raw_norm must be none|batch|layer, got {pe_raw_norm!r}")

        self.node_pe_in = nn.Linear(pe_steps - 2, grit_dim)   # RRWP diagonal, k>=2
        self.pair_pe_in = nn.Linear(pe_steps, e_dim)          # full RRWP incl. I and k=1

        self.layers = nn.ModuleList([
            GRITLayer(grit_dim, e_dim, heads=heads, dropout=dropout,
                      attn_dropout=attn_dropout, clamp=clamp,
                      edge_enhance=edge_enhance, act=act)
            for _ in range(n_layers)])

        # Lift to the width net_v7 expects for pocket attention / MMCAF.
        self.out_proj = nn.Linear(grit_dim, d_model)
        self.attn_pool = nn.Linear(d_model, 1)

    def forward(self, mol_x, mol_x_feat, bond_x, atom_edge_index, mol_batch):
        N = mol_x_feat.size(0)
        atom_h = self.atom_type_encoder(mol_x.squeeze(-1)) + self.atom_feat_encoder(mol_x_feat)

        node_pe, pair_attr, pair_index = atom_rrwp(atom_edge_index, N, self.pe_steps)
        if self.pe_norm is not None:
            node_pe = self.pe_norm(node_pe)
            pair_attr = self.pair_norm(pair_attr)

        atom_h = atom_h + self.node_pe_in(node_pe)
        e = self.pair_pe_in(pair_attr)

        # Fold real bond types into the pair embedding at the matching pairs.
        # Pair ids are encoded as src*N + dst so the two lists match with one
        # searchsorted rather than an N x N table.
        pair_key = pair_index[0] * N + pair_index[1]
        bond_key = atom_edge_index[0] * N + atom_edge_index[1]
        order = torch.argsort(pair_key)
        pos = torch.searchsorted(pair_key[order], bond_key).clamp(max=pair_key.numel() - 1)
        slot = order[pos]
        valid = pair_key[slot] == bond_key            # every bond is a 1-hop pair
        e = e.index_add(0, slot[valid], self.bond_encoder(bond_x.squeeze(-1))[valid])

        deg = torch.zeros(N, device=atom_h.device).index_add_(
            0, atom_edge_index[1],
            torch.ones(atom_edge_index.size(1), device=atom_h.device))
        log_deg = torch.log1p(deg).unsqueeze(-1)

        for layer in self.layers:
            atom_h, e = layer(atom_h, e, pair_index, log_deg)

        atom_h = self.out_proj(atom_h)                # grit_dim -> d_model

        # net_v7 discards this pooled vector (it re-pools lig_attended after
        # MMCAF); kept only so the return contract matches LigandEncoder.
        attn = pyg_softmax(self.attn_pool(atom_h).squeeze(-1), mol_batch)
        mol_pool = global_add_pool(atom_h * attn.unsqueeze(-1), mol_batch)
        return atom_h, mol_pool, None

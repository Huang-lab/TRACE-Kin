"""Ligand encoder with switchable positional/structural encoding.

One class, three modes, selected by config so ablations need no code edits:

    pe_mode="none"    atom_type embedding + 43-d feature MLP           (current v7)
    pe_mode="input"   ... + RWSE projected and added once at the input
    pe_mode="lspe"    ... + a learnable positional tensor `p` carried through
                      every layer with its own PNA aggregation

**Why RWSE and not Laplacian PE.** Laplacian eigenvectors are defined only up to
sign, needing sign-flip augmentation, and are unstable across near-degenerate
eigenvalues. RWSE has no such ambiguity, and its return probabilities directly
encode ring size and membership -- the chemistry that 1-WL message passing
provably cannot distinguish (two fused 6-rings vs a bridged bicyclic, etc.).

**LSPE** follows Dwivedi et al., "Graph Neural Networks with Learnable Structural
and Positional Representations" (ICLR 2022), reference implementation
vijaydwivedi75/gnn-lspe. The distinguishing property is that positional
information *co-evolves* with content rather than being a frozen input feature:

    h_l+1 = h_l + relu( PNA_h( [h_l || p_l] ) )        content path, ReLU
    p_l+1 = p_l + tanh( PNA_p(  p_l ) )                positional path, tanh

`tanh` on the positional path is deliberate -- it keeps `p` bounded so it cannot
blow up across layers. `[h || p]` (rather than `h + p`) lets the message function
see content and position jointly instead of a pre-summed mixture.

Deviation from the reference: LSPE's DGL implementation uses two mailboxes in a
single message-passing pass. PyG's `PNAConv` exposes one aggregation path, so we
run two `PNAConv` instances sharing the same edge features. This preserves the
separate-aggregation-with-separate-nonlinearity structure; it does not reproduce
the fused kernel exactly.

Not carried over from the reference: the auxiliary Laplacian-eigenvector loss on
`p`. It requires LapPE targets and is reported as a small effect; add it here if
the LSPE arm shows promise.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.utils import softmax as pyg_softmax

from .pna import PNAConv


# ---------------------------------------------------------------------------
# Structural encoding
# ---------------------------------------------------------------------------

def atom_rwse(edge_index: torch.Tensor, num_nodes: int, steps: int) -> torch.Tensor:
    """RWSE: diag((D^-1 A)^k) for k = 2..steps+1.  -> (num_nodes, steps)

    Entry (i, k) is the probability that a k-step random walk starting at atom i
    returns to atom i. For an atom in a ring of size r this is ~0 at odd k and
    peaks at k = r, which is what makes it a ring descriptor.

    **The walk starts at k=2, not at the canonical k=1.** A molecular graph has
    no self-loops, so diag(P) is identically zero and the k=1 channel carried no
    information; dropping it buys one more step of reach for the same `steps`.
    The columns are therefore shifted by one relative to the usual RWSE
    definition and to any number recorded before `4f5c231`: with steps=8 this
    covers ring sizes 2..9 where it used to cover 1..8. It is not a free change
    of convention -- it changes the input features of `pe_mode="input"` as much
    as of `pe_mode="lspe"`, so a checkpoint or a metric from either arm predates
    it.

    A batch of molecules is block-diagonal, so one dense (N, N) covers the whole
    batch and walks cannot cross between molecules. Molecules are small: at
    ~1500 atoms per batch this is a 9 MB matrix and `steps` matmuls.
    """
    A = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
    A[edge_index[0], edge_index[1]] = 1.0
    deg = A.sum(dim=1).clamp(min=1)
    P = A / deg.unsqueeze(1)                      # row-stochastic D^-1 A
    M = P @ P
    out = [M.diagonal()]
    for _ in range(steps - 1):
        M = M @ P
        out.append(M.diagonal())
    return torch.stack(out, dim=-1)               # (N, steps)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class LigandEncoder(nn.Module):
    """PNA ligand encoder with pe_mode in {"none", "input", "lspe"}.

    Returns (atom_h, mol_pool, p):
      atom_h   (N, d_model)  per-atom representations
      mol_pool (B, d_model)  attention-pooled molecule vector
      p        (N, pe_dim) in lspe mode, else None
    """

    def __init__(self, mol_in_channels: int, d_model: int, mol_deg,
                 n_layers: int = 3, heads: int = 8, dropout: float = 0.1,
                 pe_mode: str = "none", pe_steps: int = 16, pe_dim: int = 16,
                 pe_fold_norm: bool = True, pe_raw_norm: str = "none"):
        super().__init__()
        if pe_mode not in ("none", "input", "lspe"):
            raise ValueError(f"pe_mode must be none|input|lspe, got {pe_mode!r}")
        self.pe_mode = pe_mode
        self.pe_steps = pe_steps
        self.pe_fold_norm = pe_fold_norm
        self.pe_dim = pe_dim
        self.n_layers = n_layers
        
        if pe_mode == "none" or pe_raw_norm == "none":
            self.pe_raw_norm = None
        elif pe_raw_norm == "batch":
            self.pe_raw_norm = nn.BatchNorm1d(pe_steps, eps=1e-3)
        elif pe_raw_norm == "layer":
            self.pe_raw_norm = nn.LayerNorm(pe_steps)
        else: 
            raise ValueError(f"pe_raw_norm must be none|batch|layer, got {pe_raw_norm!r}")

        # --- atom / bond featurization (unchanged from the v7 baseline) ---
        self.atom_type_encoder = nn.Embedding(20, d_model)
        self.atom_feat_encoder = _MLP([mol_in_channels, d_model * 2, d_model], out_norm=True)
        self.bond_encoder = nn.Embedding(5, d_model)

        # --- positional encoding entry points ---
        if pe_mode == "input":
            self.pe_in = nn.Linear(pe_steps, d_model)
        elif pe_mode == "lspe":
            self.pe_in = nn.Linear(pe_steps, pe_dim)

        # The fold-back pair is used only on the lspe path, where `p` exists.
        # Built unconditionally they are dead weights in the other two modes --
        # no gradient reaches them, so they stay at initialisation -- and, worse,
        # they change the state_dict of a `pe_mode="none"` run, so a baseline
        # checkpoint from before the lspe work no longer loads into a baseline
        # model. The arm the fold-back was not supposed to touch was the one it
        # broke.
        if pe_mode == "lspe":
            self.p_out = nn.Linear(pe_dim, pe_steps)
            self.Whp = nn.Linear(d_model + pe_steps, d_model)

        # --- message passing ---
        # In lspe mode the content conv consumes [h || p], so its input widens.
        h_in = d_model + pe_dim if pe_mode == "lspe" else d_model

        self.convs_h = nn.ModuleList()
        self.norms_h = nn.ModuleList()
        self.convs_p = nn.ModuleList() if pe_mode == "lspe" else None

        for _ in range(n_layers):
            self.convs_h.append(PNAConv(
                in_channels=h_in, out_channels=d_model, edge_dim=d_model,
                aggregators=['mean', 'min', 'max', 'std'],
                scalers=['identity', 'amplification', 'linear'],
                deg=mol_deg, pre_layers=2, post_layers=1, towers=heads,
            ))
            self.norms_h.append(GraphNorm(d_model))
            if pe_mode == "lspe":
                # towers=1 on the positional path: pe_dim is small and need not
                # be divisible by `heads`.
                self.convs_p.append(PNAConv(
                    in_channels=pe_dim, out_channels=pe_dim, edge_dim=d_model,
                    aggregators=['mean', 'min', 'max', 'std'],
                    scalers=['identity', 'amplification', 'linear'],
                    deg=mol_deg, pre_layers=2, post_layers=1, towers=1,
                ))

        self.dropout = nn.Dropout(dropout)
        self.attn_pool = nn.Linear(d_model, 1)

    def forward(self, mol_x, mol_x_feat, bond_x, atom_edge_index, mol_batch):
        atom_h = self.atom_type_encoder(mol_x.squeeze()) + self.atom_feat_encoder(mol_x_feat)
        encoded_bonds = self.bond_encoder(bond_x.squeeze())

        p = None
        if self.pe_mode != "none":
            pe = atom_rwse(atom_edge_index, atom_h.size(0), self.pe_steps)
            if self.pe_raw_norm is not None:
                pe = self.pe_raw_norm(pe)
            if self.pe_mode == "input":
                atom_h = atom_h + self.pe_in(pe)          # frozen, added once
            else:
                p = self.pe_in(pe)                        # seed the learnable p

        for i in range(self.n_layers):
            h_in = atom_h if p is None else torch.cat([atom_h, p], dim=-1)
            h_new = self.convs_h[i](h_in, atom_edge_index, encoded_bonds)
            h_new = self.norms_h[i](h_new, mol_batch)
            atom_h = atom_h + self.dropout(F.relu(h_new))          # content: ReLU

            if p is not None:
                p_new = self.convs_p[i](p, atom_edge_index, encoded_bonds)
                p = p + self.dropout(torch.tanh(p_new))            # position: tanh
        
        if p is not None:
            p_c = self.p_out(self.dropout(p))
            
            if self.pe_fold_norm:
                p_c = p_c - global_mean_pool(p_c, mol_batch)[mol_batch]
                nrm = (global_add_pool(p_c ** 2, mol_batch) + 1e-6).sqrt()

                p_c = p_c / nrm[mol_batch]

            atom_h = self.Whp(self.dropout(torch.cat([atom_h, p_c], dim = -1)))

        attn = pyg_softmax(self.attn_pool(atom_h).squeeze(-1), mol_batch)
        mol_pool = global_add_pool(atom_h * attn.unsqueeze(-1), mol_batch)
        return atom_h, mol_pool, p


class _MLP(nn.Module):
    """Local copy of the shared MLP so this module stays self-contained."""

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

    def forward(self, x):
        y = self.in_ln(x) if self.in_norm else x
        for i in range(self.hidden_layers):
            y = F.relu(self.FC_layers[i](y))
        y = self.FC_layers[-1](y)
        return self.out_ln(y) if self.out_norm else y


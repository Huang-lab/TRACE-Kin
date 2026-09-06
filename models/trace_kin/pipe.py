"""PIPE: Positional Encoding meets Persistent Homology — PNA variant, no C++.

Reimplementation of Aalto-QuML/PIPE (Verma, Souza & Garg, ICML 2025), which is
SPE (Huang et al., ICLR 2024) with its phi network replaced by a GNN that emits
persistent-homology descriptors at every layer (RePHINE; Immonen, Souza & Garg,
NeurIPS 2023).

Two deliberate departures from the reference, both documented at their site:

  1. phi uses PNAConv, not GIN. See PhiPNA for how the (N, S, M) tensor is fed
     to a 2-D conv without breaking equivariance in the slot axis.
  2. No C++ extension. Graph persistent homology is only H0 + H1, so it reduces
     to "sort the edges, run union-find". The combinatorics carries no gradient
     -- it only decides WHICH indices pair -- so it runs in plain Python and the
     diagram VALUES are gathers that autograd handles for free.

ARCHITECTURE (matches the reference dataflow)

    Lambda (B,K), V (N,K)          Laplacian eigenpairs, PRECOMPUTED
      |  psi_1..psi_M              M MLPs on eigenvalues        -> Z (B,K,M)
      |  W = V diag(Z_m) V^T       per graph                    -> W (N,N,M)
      |  phi = PhiPNA              L layers; after each, RePHINE
      +--> node_pe   (N, out_dims)
      +--> topo_emb  (B, ph_out)

    The node PE is injected ONCE at the base model's input (`pe_aggregate:add`
    in the reference) -- PIPE is NOT an LSPE-style co-evolving dual stream. The
    topological embedding bypasses the GNN entirely and is concatenated at the
    prediction head (reference model.py:189).

INTEGRATION WITH TRACE-Kin v7

    node_pe   -> `pe_mode="input"` hook: atom_h = atom_h + W_pe(node_pe)
    topo_emb  -> fourth block in `feat`, feat_dim d_model*3 -> d_model*3 + ph_out

    This is ligand-only. phi materialises a dense (N, S, hidden) tensor, so it
    is quadratic in nodes: ~0.8 MB per 40-atom ligand but ~128 MB per 500-residue
    protein at hidden=128. Do not put it on the contact-map graph.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pna import PNAConv


# --------------------------------------------------------------------------- #
# 0. Preprocessing: Laplacian eigenpairs                                        #
# --------------------------------------------------------------------------- #
def laplacian_eigenpairs(edge_index: torch.Tensor, num_nodes: int, k: int
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Smallest-k eigenpairs of the symmetric normalised Laplacian.

    Returns (Lambda (k,), V (num_nodes, k)), zero-padded when num_nodes < k.
    Deterministic given the graph, so compute this ONCE per molecule at cache
    build time and store it -- not in the training loop.
    """
    A = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
    A[edge_index[0], edge_index[1]] = 1.0
    A = torch.maximum(A, A.t())                       # force symmetry
    deg = A.sum(1).clamp(min=1)
    dinv = deg.pow(-0.5)
    L = torch.eye(num_nodes, device=A.device) - dinv.unsqueeze(1) * A * dinv.unsqueeze(0)
    evals, evecs = torch.linalg.eigh(L)               # ascending
    kk = min(k, num_nodes)
    Lam = torch.zeros(k, device=A.device)
    Vv = torch.zeros(num_nodes, k, device=A.device)
    Lam[:kk] = evals[:kk]
    Vv[:, :kk] = evecs[:, :kk]
    return Lam, Vv


# --------------------------------------------------------------------------- #
# 1. Persistent homology core (no gradient; indices only)                       #
# --------------------------------------------------------------------------- #
class _UnionFind:
    """Iterative find with path compression. The reference recurses
    (unionfind.hh); that overflows the Python stack on long chains."""

    __slots__ = ("parent",)

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, u: int) -> int:
        r = u
        while self.parent[r] != r:
            r = self.parent[r]
        while self.parent[u] != r:
            self.parent[u], u = r, self.parent[u]
        return r

    def merge(self, u: int, v: int) -> None:
        ru, rv = self.find(u), self.find(v)
        if ru != rv:
            self.parent[ru] = rv


def pair_graph(filt_v, filt_e, edges, n_vertices, diagram_type: str = "standard"):
    """Sort + union-find. Returns index tables, no values.

    diagram_type:
      "standard" -- elder rule on vertex values only. This is what PIPE's
                    GIN_PH actually constructs (gin.py:84), paired with an edge
                    filtration derived as max of the endpoints.
      "rephine"  -- adds RePHINE's tie-break: when two roots have EQUAL vertex
                    values, the one whose first incident edge (in filtration
                    order) is smaller counts as older
                    (rephine_mt_cpu.cpp:96-104). Only meaningful with an
                    independently-learned edge filtration.

    Returns (death_edge, first_edge, cycle_edges); -1 means "none".
    """
    order = sorted(range(len(edges)), key=lambda i: filt_e[i])   # stable
    death = [-1] * n_vertices
    first = [-1] * n_vertices
    cycles: List[int] = []
    uf = _UnionFind(n_vertices)

    for e in order:
        n1, n2 = edges[e]
        if first[n1] == -1:
            first[n1] = e
        if first[n2] == -1:
            first[n2] = e

        younger, older = uf.find(n1), uf.find(n2)
        if younger == older:
            cycles.append(e)                       # H1 birth
            continue

        flip = filt_v[younger] < filt_v[older]
        if diagram_type == "rephine" and filt_v[younger] == filt_v[older]:
            fy, fo = first[younger], first[older]
            if fy != -1 and fo != -1 and filt_e[fy] < filt_e[fo]:
                flip = True
        if flip:
            younger, older = older, younger
            n1, n2 = n2, n1

        death[younger] = e
        uf.merge(n1, n2)

    return death, first, cycles


# --------------------------------------------------------------------------- #
# 2. Diagram encoders (DeepSets over ragged sets)                               #
# --------------------------------------------------------------------------- #
def _mlp(dims: List[int], out_act: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(1, len(dims)):
        layers.append(nn.Linear(dims[i - 1], dims[i]))
        if i < len(dims) - 1 or out_act:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class DeepSet(nn.Module):
    """phi_set then sum-pool then rho. Permutation invariant over the set axis,
    which is required: a persistence diagram is a multiset, not a sequence."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.phi = _mlp([in_dim, hidden, hidden], out_act=True)
        self.rho = _mlp([hidden, hidden, out_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (S, in_dim) -> (out_dim,). S may be 0."""
        if x.numel() == 0:
            return x.new_zeros(self.rho[-1].out_features)
        return self.rho(self.phi(x).sum(dim=0))


# --------------------------------------------------------------------------- #
# 3. RePHINE layer: learnable filtrations -> PH -> graph descriptor             #
# --------------------------------------------------------------------------- #
class RephineLayer(nn.Module):
    """Node features -> F learnable filtrations -> H0/H1 diagrams -> vector.

    Mirrors RePHINE/layers/rephine_layer.py. `sig_filtrations=True` in the
    reference, i.e. filtration values are squashed to (0, 1) -- that keeps the
    sort stable early in training when the MLP is near its initialisation.
    """

    def __init__(self, n_features: int, n_filtrations: int = 8,
                 filtration_hidden: int = 16, out_dim: int = 64,
                 diagram_type: str = "standard", dim1: bool = True,
                 sig_filtrations: bool = True) -> None:
        super().__init__()
        self.n_filtrations = n_filtrations
        self.diagram_type = diagram_type
        self.dim1 = dim1
        self.sig = sig_filtrations

        self.filtrations = _mlp([n_features, filtration_hidden, n_filtrations])
        if diagram_type == "rephine":
            # independent edge filtration -- the RePHINE contribution
            self.edge_filtrations = _mlp([n_features, filtration_hidden, n_filtrations])

        self.set0 = DeepSet(3, out_dim, out_dim)          # (birth, death, essential)
        self.set1 = DeepSet(2, out_dim, out_dim) if dim1 else None
        fused = out_dim * (2 if dim1 else 1) * n_filtrations
        self.out = _mlp([fused, out_dim, out_dim])

    @torch.no_grad()
    def _pairs(self, fv, fe, edges, n_v):
        return pair_graph(fv.detach().cpu().tolist(), fe.detach().cpu().tolist(),
                          edges.detach().cpu().tolist(), n_v, self.diagram_type)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor, edge_batch: torch.Tensor) -> torch.Tensor:
        """
        x           (N, n_features)
        edge_index  (E, 2)  each undirected edge ONCE, global node ids
        batch       (N,)    graph id per node
        edge_batch  (E,)    graph id per edge
        returns     (B, out_dim)
        """
        fv_all = self.filtrations(x)                                  # (N, F)
        if self.diagram_type == "rephine":
            fe_all = self.edge_filtrations(x[edge_index[:, 0]] + x[edge_index[:, 1]])
        else:
            # PIPE's actual setting: edge value = max of its endpoints
            fe_all = torch.maximum(fv_all[edge_index[:, 0]], fv_all[edge_index[:, 1]])
        if self.sig:
            fv_all, fe_all = torch.sigmoid(fv_all), torch.sigmoid(fe_all)

        n_graphs = int(batch.max()) + 1
        outs = []
        for g in range(n_graphs):
            vsel = (batch == g).nonzero(as_tuple=True)[0]
            esel = (edge_batch == g).nonzero(as_tuple=True)[0]
            n_v = vsel.numel()
            remap = torch.full((x.size(0),), -1, dtype=torch.long, device=x.device)
            remap[vsel] = torch.arange(n_v, device=x.device)
            local = remap[edge_index[esel]]

            per_filt = []
            for f in range(self.n_filtrations):
                fv, fe = fv_all[vsel, f], fe_all[esel, f]
                death, _first, cycles = self._pairs(fv, fe, local, n_v)
                dt = torch.as_tensor(death, device=x.device)
                ess = dt == -1
                # gathers: this is where the gradient flows back into the MLPs
                d_val = torch.where(ess, fv.new_ones(()), fe[dt.clamp(min=0)]) \
                    if fe.numel() else fv.new_ones(n_v)
                dgm0 = torch.stack([fv, d_val, ess.to(fv.dtype)], dim=-1)
                per_filt.append(self.set0(dgm0))
                if self.dim1:
                    if cycles:
                        c = torch.as_tensor(cycles, device=x.device)
                        dgm1 = torch.stack([fe[c], torch.ones_like(fe[c])], dim=-1)
                    else:
                        dgm1 = fv.new_zeros((0, 2))
                    per_filt.append(self.set1(dgm1))
            outs.append(torch.cat(per_filt, dim=-1))
        return self.out(torch.stack(outs, dim=0))


# --------------------------------------------------------------------------- #
# 4. phi: PNA over the (N, S, M) tensor, with PH after every layer              #
# --------------------------------------------------------------------------- #
class PhiPNA(nn.Module):
    """SPE's phi, with PNAConv in place of GIN.

    THE SHAPE PROBLEM. SPE hands phi a tensor W of shape (N, S, M): for each
    node, an S-vector per channel, where S is the padded node axis of
    V diag(Z) V^T. The reference's GIN keeps that middle axis as a passive batch
    dimension and runs its MLP on the last axis, which is what makes phi
    equivariant to permutations of BOTH axes.

    PNAConv only accepts 2-D node features, so it cannot take (N, S, M)
    directly. FLATTENING to (N, S*M) would be wrong -- it lets the MLP treat
    slot s=3 differently from s=7, destroying equivariance in the slot axis.

    Instead we fold S into the NODE axis: (N*S, M), with edge_index replicated S
    times at offsets s*N. Every slot then sees the identical graph and the
    identical weights, so equivariance is preserved exactly. Cost is S copies of
    the message passing -- acceptable for ligands (S <= ~64), which is the other
    reason this is ligand-only.
    """

    def __init__(self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int,
                 deg: torch.Tensor, ph_out: int = 64, n_filtrations: int = 8,
                 filtration_hidden: int = 16, diagram_type: str = "standard",
                 dim1: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.ph_layers = nn.ModuleList()
        d = in_dims
        for i in range(n_layers):
            out = out_dims if i == n_layers - 1 else hidden_dims
            self.convs.append(PNAConv(
                in_channels=d, out_channels=out, edge_dim=None,
                aggregators=['mean', 'min', 'max', 'std'],
                scalers=['identity', 'amplification', 'linear'],
                deg=deg, pre_layers=2, post_layers=1, towers=1))
            if i != n_layers - 1:                       # reference: no PH on the last layer
                self.ph_layers.append(RephineLayer(
                    n_features=out, n_filtrations=n_filtrations,
                    filtration_hidden=filtration_hidden, out_dim=ph_out,
                    diagram_type=diagram_type, dim1=dim1))
            d = out
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _replicate(edge_index: torch.Tensor, n_nodes: int, S: int) -> torch.Tensor:
        """(2, E) -> (2, E*S) with slot s offset by s*n_nodes."""
        off = (torch.arange(S, device=edge_index.device) * n_nodes).view(1, S, 1)
        return (edge_index.unsqueeze(1) + off).reshape(2, -1)

    def forward(self, W, mp_edge_index, mask, ph_edge_index, batch, edge_batch):
        """
        W              (N, S, M)  from SPE
        mp_edge_index  (2, E)     DIRECTED, both orientations -- for message passing
        mask           (N, S)     True where the slot is a real node
        ph_edge_index  (E', 2)    UNDIRECTED, each edge once -- for persistent homology
        """
        N, S, _ = W.shape
        rep = self._replicate(mp_edge_index, N, S)
        X = W
        ph_vecs = []
        for i, conv in enumerate(self.convs):
            flat = X.permute(1, 0, 2).reshape(N * S, -1)      # (S*N, M) slot-major
            flat = conv(flat, rep)
            X = flat.view(S, N, -1).permute(1, 0, 2)          # back to (N, S, D)
            X = self.dropout(F.relu(X))
            if i != len(self.convs) - 1:
                # reference gin.py:118 -- masked reduction over the slot axis
                x_ph = (X * mask.unsqueeze(-1)).sum(dim=1)    # (N, D)
                ph_vecs.append(self.ph_layers[i](x_ph, ph_edge_index, batch, edge_batch))
        node_pe = (X * mask.unsqueeze(-1)).sum(dim=1)         # (N, out_dims)
        topo = torch.stack(ph_vecs).mean(dim=0) if ph_vecs else None
        return node_pe, topo


# --------------------------------------------------------------------------- #
# 5. PIPE                                                                       #
# --------------------------------------------------------------------------- #
class PIPE(nn.Module):
    """Returns (node_pe (N, pe_dims), topo_emb (B, ph_out))."""

    def __init__(self, pe_dims: int = 37, n_psis: int = 16, psi_hidden: int = 16,
                 n_psi_layers: int = 3, n_phi_layers: int = 8, phi_hidden: int = 128,
                 deg: Optional[torch.Tensor] = None, ph_out: int = 64,
                 n_filtrations: int = 8, filtration_hidden: int = 16,
                 diagram_type: str = "standard", dim1: bool = True,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if deg is None:
            raise ValueError("PNA needs the training-set degree histogram `deg`.")
        self.pe_dims = pe_dims
        self.psi_list = nn.ModuleList([
            _mlp([1] + [psi_hidden] * (n_psi_layers - 1) + [1])
            for _ in range(n_psis)
        ])
        self.phi = PhiPNA(n_phi_layers, n_psis, phi_hidden, pe_dims, deg,
                          ph_out=ph_out, n_filtrations=n_filtrations,
                          filtration_hidden=filtration_hidden,
                          diagram_type=diagram_type, dim1=dim1, dropout=dropout)

    def forward(self, Lambda, V, mp_edge_index, ph_edge_index, batch, edge_batch):
        """
        Lambda  (B, K)  eigenvalues, precomputed
        V       (N, K)  eigenvectors, precomputed
        """
        Z = torch.stack([psi(Lambda.unsqueeze(-1)).squeeze(-1)
                         for psi in self.psi_list], dim=2)        # (B, K, M)

        n_graphs = int(batch.max()) + 1
        S = int(torch.bincount(batch, minlength=n_graphs).max())
        N, M = V.size(0), Z.size(2)
        W = V.new_zeros(N, S, M)
        mask = torch.zeros(N, S, dtype=torch.bool, device=V.device)

        for g in range(n_graphs):
            idx = (batch == g).nonzero(as_tuple=True)[0]
            Vi = V[idx]                                            # (n_i, K)
            Zi = Z[g].permute(1, 0).diag_embed()                   # (M, K, K)
            Wi = Vi.unsqueeze(0).matmul(Zi).matmul(Vi.t().unsqueeze(0))   # (M, n_i, n_i)
            n_i = idx.numel()
            W[idx[:, None], torch.arange(n_i, device=V.device)[None, :], :] = \
                Wi.permute(1, 2, 0)
            mask[idx[:, None], torch.arange(n_i, device=V.device)[None, :]] = True

        return self.phi(W, mp_edge_index, mask.to(V.dtype), ph_edge_index,
                        batch, edge_batch)


def undirected_edges(edge_index: torch.Tensor) -> torch.Tensor:
    """PyG (2, E) bidirectional -> (E/2, 2), each undirected edge ONCE.

    TRACE-Kin's `atom_edge_index` carries both orientations (bond_feature writes
    both triangles). Persistent homology must see each bond once or every cycle
    is counted twice.
    """
    src, dst = edge_index[0], edge_index[1]
    keep = src < dst
    return torch.stack([src[keep], dst[keep]], dim=1)

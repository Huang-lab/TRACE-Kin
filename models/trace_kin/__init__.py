"""TRACE-Kin model package.

Three architecture variants are exposed:

- :class:`TraceKinV1` — the original PSICHIC architecture that produced
  the 1,847-config benchmark in ``trace_doc/kinetic_regress_benchmark.csv``.
  Frozen baseline; also serves as the GNN backbone of v4.
- :class:`TraceKinV4` — MAP-GNN (Mutation-Aware Pocket GNN). v1 GNN
  backbone + per-residue novelty scoring + pocket-attention pool.
- :class:`TraceKinV5` — Bidirectional Graph-Mamba + Cross-Attention.
  Novel architecture: Mamba SSM for efficient long-range protein sequence
  modeling, graph-structure injection from contact maps, cross-attention
  for interpretable enzyme-ligand interaction, substrate-conditioned
  pooling. Designed for mutation-aware kinetics with MutaPLM embeddings.
"""

from .net_v1 import TraceKinV1
from .net_v4 import TraceKinV4
from .net_v5 import TraceKinV5

__all__ = ["TraceKinV1", "TraceKinV4", "TraceKinV5"]

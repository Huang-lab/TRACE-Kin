"""TRACE-Kin model package.

One architecture is exported: :class:`TraceKinV7`.

``TraceKinV1``, ``V4``, ``V5``, ``V5T`` and ``V6C`` were deleted in dfb7ab3a
("got rid of unused older versions"). Their configs (``training/config_v*.json``)
and any checkpoints still exist, but the code to build them does not; recover it
from that commit's parent if a prior number has to be reproduced. Their
descriptions are kept below so a checkpoint or a benchmark row can still be
matched to what produced it.

Removed variants:

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
- :class:`TraceKinV5T` — Vanilla Transformer + Cross-Attention.
  Alternative to v5: standard self-attention with RoPE replaces BiMamba.
  Same cross-attention and pooling as v5 for fair comparison.
- :class:`TraceKinV6C` — Hierarchical Pocket Graph + Multi-Modal Fusion.
  GATv2 on contact-map graph, dynamic pocket identification from
  protein-to-ligand cross-attention, MMCAF at pocket level, hierarchical
  pooling with gated pocket/global fusion.
Exported:

- :class:`TraceKinV7` - Structure-Guided Embedding Distillation (SGED).
  Dual-stream: GATv2 structure stream for pocket identification +
  Multi-Scale PCER over raw 4096-d PLM embeddings with Mutation-Aware
  Gating. Decouples structural reasoning from embedding aggregation.
"""

from .net_v7 import TraceKinV7

__all__ = ["TraceKinV7"]

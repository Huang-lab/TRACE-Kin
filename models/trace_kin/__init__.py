"""TRACE-Kin model package.

Two architecture variants are exposed:

- :class:`TraceKinV1` — the original PSICHIC architecture that produced
  the 1,847-config benchmark in ``trace_doc/kinetic_regress_benchmark.csv``.
  Frozen baseline; also serves as the GNN backbone of v4.
- :class:`TraceKinV4` — MAP-GNN (Mutation-Aware Pocket GNN). v1 GNN
  backbone + per-residue novelty scoring
  (``||MutaPLM[r] − aa_typical_mean[aa[r]]|| / std``) + pocket-attention
  pool over post-GNN residue features + ChemBERT/MoLFormer (768-d)
  global molecular context residual added to mol_pool. Targets
  mutation-sensitive prediction without explicit WT/mutant labels.

v3 (FP-MLP and ChemBERT-context-residual variants) was removed after
empirical results showed it regressed v1 on a meaningful subset of
cells without closing the RF gap on any. v4 supersedes it. See
PROJECT.md for the design rationale.
"""

from .net_v1 import TraceKinV1
from .net_v4 import TraceKinV4

__all__ = ["TraceKinV1", "TraceKinV4"]

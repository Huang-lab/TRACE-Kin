"""TRACE-Kin model package.

Three architecture variants are exposed:

- :class:`TraceKinV1` — the original PSICHIC architecture that produced the
  1,847-config benchmark in ``trace_doc/kinetic_regress_benchmark.csv``.
  Frozen baseline; also serves as the GNN backbone of v3 and v4.
- :class:`TraceKinV3` — v1 GNN + ChemBERT/MoLFormer (768-d) global graph
  context residual added to mol_pool, with a fresh regression head. Targets
  closing the RF gap on data-rich cells via richer molecular features.
- :class:`TraceKinV4` — MAP-GNN (Mutation-Aware Pocket GNN). Adds per-residue
  novelty scoring (||MutaPLM[r] − aa_typical_mean[aa[r]]|| / std) and a
  pocket-attention pool over post-GNN residue features, on top of v3's
  MoLFormer context. Targets mutation-sensitive prediction without explicit
  WT/mutant labels.

The full design rationale lives in ``PROJECT.md``.
"""

from .net_v1 import TraceKinV1
from .net_v3 import TraceKinV3
from .net_v4 import TraceKinV4

__all__ = ["TraceKinV1", "TraceKinV3", "TraceKinV4"]

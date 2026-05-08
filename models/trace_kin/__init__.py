"""TRACE-Kin model package.

Two architecture variants are exposed:

- :class:`TraceKinV1` — the original PSICHIC architecture that produced the
  1,847-config benchmark in ``trace_doc/kinetic_regress_benchmark.csv``. Frozen
  baseline used for A/B comparison and the GNN backbone of v3.
- :class:`TraceKinV3` — dual-head architecture: v1's GNN runs alongside an
  RF-style head (mean-pooled raw protein embedding ⊕ Morgan + MACCS ligand
  fingerprints), combined via a learned per-sample sigmoid gate. Targets the
  RF gap on catalytic kinetics while preserving v1's Ki advantage.

The full design rationale lives in ``PROJECT.md``.
"""

from .net_v1 import TraceKinV1
from .net_v3 import TraceKinV3

__all__ = ["TraceKinV1", "TraceKinV3"]

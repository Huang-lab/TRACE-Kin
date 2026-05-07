"""TRACE-Kin model package.

Two architecture variants are exposed:

- :class:`TraceKinV1` — the original PSICHIC architecture that produced the
  1,847-config benchmark in ``trace_doc/kinetic_regress_benchmark.csv``. Frozen
  baseline used for A/B comparison.
- :class:`TraceKinV2` — the redesigned architecture with an embedding shortcut
  branch, attention-based residue pooling, and a fixed-weight AA residual
  fusion. Targets the RF gap on catalytic kinetics (kcat/Km/Kd/kcat_Km).

The full root-cause analysis behind the v2 design lives in ``PROJECT.md``.
"""

from .net_v1 import TraceKinV1

# net_v2 may be absent until Phase 3 lands the redesigned architecture; tolerate
# its absence so v1-only consumers (e.g. the inference API loading legacy
# checkpoints) can import this package without pulling in v2.
try:
    from .net_v2 import TraceKinV2  # noqa: F401
    __all__ = ["TraceKinV1", "TraceKinV2"]
except ImportError:
    __all__ = ["TraceKinV1"]

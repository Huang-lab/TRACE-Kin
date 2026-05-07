"""TRACE-Kin inference API.

Two public entry points:

* :class:`TraceKinPredictor` — load a trained TRACE-Kin checkpoint (v1 or v2)
  and predict a kinetic value from a (SMILES, enzyme sequence, embedding)
  triple.
* :func:`dual_embedding_confidence` — combine two predictor outputs (e.g. ESM2
  and MutaPLM branches) into a [0, 1] agreement score that operationalizes
  the "embedding disagreement as uncertainty" claim from
  ``trace_doc/TRACE_research_design.md``.
"""

from .confidence import dual_embedding_confidence

# TraceKinPredictor pulls in torch; load lazily so tooling that only needs the
# confidence metric (which is pure Python) does not require a full ML stack.
try:
    from .trace_kin_predictor import TraceKinPredictor  # noqa: F401
    __all__ = ["TraceKinPredictor", "dual_embedding_confidence"]
except ImportError:
    __all__ = ["dual_embedding_confidence"]

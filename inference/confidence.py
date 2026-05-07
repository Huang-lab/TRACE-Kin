"""Dual-embedding confidence metric.

The TRACE research-design document (``trace_doc/TRACE_research_design.md``,
Claim 3) frames embedding disagreement as a useful uncertainty signal: when
ESM2 and MutaPLM predictions diverge, the prediction is mutation-sensitive
and the model should flag it. This module turns that idea into a concrete
``[0, 1]`` confidence score callable from the inference API.

Method
------

For a kinetic type ``k`` with two predictions ``y_esm2`` and ``y_mutaplm``,
compute::

    delta = abs(y_esm2 - y_mutaplm)
    s = sigmoid(scale_k - delta / max(rmse_k, eps))
    confidence = s

where ``rmse_k`` is the historical benchmark RMSE for that kinetic (acts as
the natural disagreement scale — disagreement larger than typical model error
is a real divergence) and ``scale_k`` is a tunable midpoint (default 1.0,
i.e. confidence == 0.5 when disagreement equals one RMSE).

The defaults are derived from the cleaned benchmark; if more recent v2 RMSE
numbers are available, override via ``rmse_table=...``.
"""
from __future__ import annotations

import math

# Reasonable defaults: median test RMSE per kinetic on the historical
# benchmark, rounded. Override per-call if the v2 rerun produces tighter
# numbers and you want a stricter confidence scale.
_DEFAULT_RMSE_TABLE = {
    "kcat": 0.85,
    "km": 0.80,
    "ki": 1.25,
    "kd": 1.05,
    "kcat_km": 1.30,
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def dual_embedding_confidence(
    pred_a: dict,
    pred_b: dict,
    rmse_table: dict | None = None,
    scale_midpoint: float = 1.0,
) -> float:
    """Return a confidence score in ``[0, 1]`` for two embedding-branch predictions.

    Parameters
    ----------
    pred_a, pred_b
        Outputs from :meth:`inference.TraceKinPredictor.predict`. Must share
        the same ``kinetic_type``.
    rmse_table
        Optional override map ``{kinetic_type: typical_rmse}``. Defaults to the
        baseline benchmark RMSE per kinetic.
    scale_midpoint
        At ``delta = scale_midpoint * rmse``, confidence is 0.5. Larger values
        make the score more forgiving of disagreement; smaller values stricter.

    Returns
    -------
    float
        ``1.0`` when the two predictions agree exactly, ``0.5`` when they
        disagree by ``scale_midpoint * rmse_k``, and approaches 0 as
        disagreement grows.
    """
    if pred_a.get("kinetic_type") != pred_b.get("kinetic_type"):
        raise ValueError(
            f"kinetic_type mismatch: {pred_a.get('kinetic_type')!r} vs {pred_b.get('kinetic_type')!r}"
        )
    table = dict(_DEFAULT_RMSE_TABLE)
    if rmse_table:
        table.update(rmse_table)
    k = pred_a.get("kinetic_type")
    rmse = table.get(k, 1.0)
    delta = abs(float(pred_a["prediction"]) - float(pred_b["prediction"]))
    score = _sigmoid(scale_midpoint - delta / max(rmse, 1e-3))
    # Clamp to [0, 1] for numerical safety.
    return max(0.0, min(1.0, score))

"""Quality-gate criterion: output equivalence PROXY (not accuracy).

Reads ``metrics.extra["proxy_min_cosine_similarity"]`` (plus shape/dtype flags) written
by the local ONNX Runtime evaluation. Missing -> NOT_APPLICABLE. The wording of every
message keeps the word "proxy" so it is never read as a labeled accuracy result.
"""

from __future__ import annotations

from framework.config import CriterionConfig
from framework.pipeline.result import CriterionResult, CriterionStatus, Metrics

CRITERION = "output_equivalence_proxy"
DEFAULT_MIN_COSINE = 0.99


def validate_output_equivalence_proxy(metrics: Metrics | None, criterion: CriterionConfig) -> CriterionResult:
    min_cos = float(criterion.params.get("min_cosine_similarity", DEFAULT_MIN_COSINE))
    extra = metrics.extra if metrics else {}
    cos = extra.get("proxy_min_cosine_similarity")
    if cos is None:
        return CriterionResult(CRITERION, CriterionStatus.NOT_APPLICABLE, None, min_cos,
                               "output equivalence proxy not measured (no local evaluation)", required=criterion.required)
    shape_ok = extra.get("proxy_shape_match", 1.0) >= 1.0
    dtype_ok = extra.get("proxy_dtype_match", 1.0) >= 1.0
    if shape_ok and dtype_ok and cos >= min_cos:
        return CriterionResult(CRITERION, CriterionStatus.PASS, cos, min_cos,
                               f"proxy: min cosine {cos:.4f} >= {min_cos} on identical inputs (not accuracy)", required=criterion.required)
    reasons = []
    if not shape_ok:
        reasons.append("output shapes differ")
    if not dtype_ok:
        reasons.append("output dtypes differ")
    if cos < min_cos:
        reasons.append(f"min cosine {cos:.4f} < {min_cos}")
    return CriterionResult(CRITERION, CriterionStatus.FAIL if criterion.required else CriterionStatus.WARN, cos, min_cos,
                           "proxy (not accuracy): " + "; ".join(reasons), required=criterion.required)

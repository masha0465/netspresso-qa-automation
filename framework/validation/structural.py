"""Quality-gate criterion: artifact structural validity.

Reads the result that :mod:`framework.evaluation.artifact_structure` stored under
``artifact.metadata["structural_validation"]``. Missing -> NOT_APPLICABLE.
"""

from __future__ import annotations

from framework.config import CriterionConfig
from framework.pipeline.result import Artifact, CriterionResult, CriterionStatus

CRITERION = "structural_validity"


def validate_structural_validity(artifact: Artifact | None, criterion: CriterionConfig) -> CriterionResult:
    threshold = "required" if criterion.required else "optional"
    sv = (artifact.metadata or {}).get("structural_validation") if artifact else None
    status = sv.get("status") if isinstance(sv, dict) else None
    if status is None or status == "NOT_APPLICABLE":
        return CriterionResult(CRITERION, CriterionStatus.NOT_APPLICABLE, status, threshold,
                               (sv or {}).get("message", "structural validation not performed") if isinstance(sv, dict) else "structural validation not performed",
                               required=criterion.required)
    if status == "PASS":
        return CriterionResult(CRITERION, CriterionStatus.PASS, "PASS", threshold, sv.get("message", "structurally valid"), required=criterion.required)
    return CriterionResult(CRITERION, CriterionStatus.FAIL if criterion.required else CriterionStatus.WARN, status, threshold,
                           "structural validation failed: " + str(sv.get("message", "")), required=criterion.required)

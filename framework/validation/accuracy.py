"""Accuracy validation: baseline vs optimized/current.

Semantics: ``drop_pp = (baseline.accuracy - current.accuracy) * 100`` in
absolute percentage points. Improvements (negative drop) always pass.
"""

from __future__ import annotations

from framework.config import CriterionConfig
from framework.pipeline.result import CriterionResult, CriterionStatus, Metrics

CRITERION = "accuracy"


def accuracy_drop_pp(baseline: Metrics | None, current: Metrics | None) -> float | None:
    if baseline is None or current is None or baseline.accuracy is None or current.accuracy is None:
        return None
    return round((baseline.accuracy - current.accuracy) * 100.0, 4)


def validate_accuracy(baseline: Metrics | None, current: Metrics | None, criterion: CriterionConfig) -> CriterionResult:
    drop = accuracy_drop_pp(baseline, current)
    threshold = criterion.threshold
    if drop is None or threshold is None:
        return CriterionResult(
            name=CRITERION,
            status=CriterionStatus.NOT_APPLICABLE,
            observed=drop,
            threshold=threshold,
            message="accuracy not available for baseline and/or current metrics",
            required=criterion.required,
        )
    if drop <= threshold:
        return CriterionResult(
            name=CRITERION,
            status=CriterionStatus.PASS,
            observed=drop,
            threshold=threshold,
            message=f"accuracy drop {drop:.2f} pp <= {threshold:.2f} pp",
            required=criterion.required,
        )
    return CriterionResult(
        name=CRITERION,
        status=CriterionStatus.FAIL if criterion.required else CriterionStatus.WARN,
        observed=drop,
        threshold=threshold,
        message=f"accuracy drop {drop:.2f} pp exceeds threshold {threshold:.2f} pp",
        required=criterion.required,
    )

"""Performance / memory / model-size validation.

Semantics: relative increase ``(current - baseline) / baseline * 100``.
Decreases (improvements) always pass. A zero or missing baseline makes the
criterion NOT_APPLICABLE rather than silently passing.
"""

from __future__ import annotations

from framework.config import CriterionConfig
from framework.pipeline.result import CriterionResult, CriterionStatus, Metrics


def relative_increase_percent(baseline: float | None, current: float | None) -> float | None:
    if baseline is None or current is None or baseline <= 0:
        return None
    return round((current - baseline) / baseline * 100.0, 3)


def _validate_increase(
    name: str, unit: str, baseline_value: float | None, current_value: float | None, criterion: CriterionConfig
) -> CriterionResult:
    increase = relative_increase_percent(baseline_value, current_value)
    threshold = criterion.threshold
    if increase is None or threshold is None:
        return CriterionResult(
            name=name,
            status=CriterionStatus.NOT_APPLICABLE,
            observed=increase,
            threshold=threshold,
            message=f"{name} not available or baseline is zero",
            required=criterion.required,
        )
    detail = f"{baseline_value:.3f} -> {current_value:.3f} {unit} ({increase:+.2f}%)"
    if increase <= threshold:
        return CriterionResult(
            name=name,
            status=CriterionStatus.PASS,
            observed=increase,
            threshold=threshold,
            message=f"{name} {detail} within +{threshold:.1f}%",
            required=criterion.required,
        )
    return CriterionResult(
        name=name,
        status=CriterionStatus.FAIL if criterion.required else CriterionStatus.WARN,
        observed=increase,
        threshold=threshold,
        message=f"{name} {detail} exceeds +{threshold:.1f}%",
        required=criterion.required,
    )


def validate_latency(baseline: Metrics | None, current: Metrics | None, criterion: CriterionConfig) -> CriterionResult:
    return _validate_increase(
        "latency", "ms", baseline.latency_ms if baseline else None, current.latency_ms if current else None, criterion
    )


def validate_memory(baseline: Metrics | None, current: Metrics | None, criterion: CriterionConfig) -> CriterionResult:
    return _validate_increase(
        "memory", "MB", baseline.memory_mb if baseline else None, current.memory_mb if current else None, criterion
    )


def validate_model_size(baseline: Metrics | None, current: Metrics | None, criterion: CriterionConfig) -> CriterionResult:
    return _validate_increase(
        "model_size",
        "MB",
        baseline.model_size_mb if baseline else None,
        current.model_size_mb if current else None,
        criterion,
    )

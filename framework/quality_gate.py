"""Configurable Quality Gate.

The gate evaluates every configured criterion and returns *all* individual
results plus the overall verdict. A single failing **required** criterion fails
the gate. Failing optional criteria are reported as WARN. A required criterion
that cannot be evaluated (NOT_APPLICABLE) also fails the gate: a release
decision must not rest on missing data.
"""

from __future__ import annotations

from collections.abc import Callable

from framework.config import CriterionConfig, QualityGateConfig
from framework.pipeline.result import (
    CriterionResult,
    CriterionStatus,
    ExecutionResult,
    ExecutionStatus,
    QualityGateResult,
    Status,
)
from framework.validation.accuracy import validate_accuracy
from framework.validation.artifact import validate_artifact
from framework.validation.performance import validate_latency, validate_memory, validate_model_size
from framework.validation.reproducibility import validate_reproducibility

Validator = Callable[[ExecutionResult, CriterionConfig], CriterionResult]

_VALIDATORS: dict[str, Validator] = {
    "accuracy": lambda ex, c: validate_accuracy(ex.baseline_metrics, ex.metrics, c),
    "latency": lambda ex, c: validate_latency(ex.baseline_metrics, ex.metrics, c),
    "memory": lambda ex, c: validate_memory(ex.baseline_metrics, ex.metrics, c),
    "model_size": lambda ex, c: validate_model_size(ex.baseline_metrics, ex.metrics, c),
    "artifact": lambda ex, c: validate_artifact(ex.artifact, c),
    "reproducibility": lambda ex, c: validate_reproducibility(ex.reproducibility, c),
}

ORDER = ("accuracy", "latency", "memory", "model_size", "artifact", "reproducibility")
LABELS = {
    "accuracy": "Accuracy",
    "latency": "Latency",
    "memory": "Memory",
    "model_size": "Model Size",
    "artifact": "Artifact Integrity",
    "reproducibility": "Reproducibility",
}


class QualityGate:
    def __init__(self, config: QualityGateConfig) -> None:
        self._config = config
        unknown = set(config.criteria) - set(_VALIDATORS)
        if unknown:
            raise ValueError(f"Quality gate has no validator for criteria: {sorted(unknown)}")

    def evaluate(self, execution: ExecutionResult) -> QualityGateResult:
        if execution.execution_status != ExecutionStatus.COMPLETED:
            reason = f"execution status is {execution.execution_status.value}; gate not evaluated"
            return QualityGateResult(overall=Status.FAIL, criteria=[], reasons=[reason])

        criteria: list[CriterionResult] = []
        reasons: list[str] = []
        for name in ORDER:
            criterion = self._config.get(name)
            if criterion is None:
                continue
            result = _VALIDATORS[name](execution, criterion)
            criteria.append(result)
            if result.status == CriterionStatus.FAIL or (result.status == CriterionStatus.NOT_APPLICABLE and criterion.required):
                reasons.append(f"{LABELS[name]}: {result.message}")

        overall = Status.FAIL if reasons else Status.PASS
        return QualityGateResult(overall=overall, criteria=criteria, reasons=reasons)


def render_text(result: QualityGateResult, width: int = 32) -> str:
    """Plain-text table in the style used by the project documentation."""
    lines = ["QUALITY GATE", "-" * width]
    for c in result.criteria:
        label = LABELS.get(c.name, c.name)
        lines.append(f"{label:<20}{c.status.value}")
    lines.append("-" * width)
    lines.append(f"{'Overall':<20}{result.overall.value}")
    if result.reasons:
        lines.append("")
        lines.append("Reasons:")
        lines.extend(f"  - {r}" for r in result.reasons)
    return "\n".join(lines)

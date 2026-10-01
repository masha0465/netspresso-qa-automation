"""Reproducibility assessment and validation.

Levels (never claimed without evidence):
* BITWISE_REPRODUCIBLE       – all repeated runs produced identical artifact checksums
* FUNCTIONALLY_REPRODUCIBLE  – checksums differ, but metrics of every run stay
                               within ``tolerance`` of the first run
* NOT_REPRODUCIBLE           – metrics diverge beyond tolerance
* NOT_VERIFIED               – fewer than two runs available
"""

from __future__ import annotations

from framework.config import CriterionConfig
from framework.pipeline.result import (
    CriterionResult,
    CriterionStatus,
    Metrics,
    Reproducibility,
    ReproducibilityLevel,
)

CRITERION = "reproducibility"
ACCEPTED = {ReproducibilityLevel.BITWISE_REPRODUCIBLE, ReproducibilityLevel.FUNCTIONALLY_REPRODUCIBLE}


def _within(a: float | None, b: float | None, tolerance: float) -> bool:
    if a is None or b is None:
        return a is None and b is None
    scale = max(abs(a), 1e-12)
    return abs(a - b) / scale <= tolerance


def assess_reproducibility(
    checksums: list[str], metrics_runs: list[Metrics] | None = None, tolerance: float = 0.01
) -> Reproducibility:
    """Derive a :class:`Reproducibility` from repeated-run evidence."""
    runs = len(checksums)
    if runs < 2:
        return Reproducibility(level=ReproducibilityLevel.NOT_VERIFIED, runs=runs, checksums=list(checksums),
                               notes="fewer than two runs")
    if len(set(checksums)) == 1:
        return Reproducibility(level=ReproducibilityLevel.BITWISE_REPRODUCIBLE, runs=runs, checksums=list(checksums),
                               notes="identical checksums")
    if metrics_runs and len(metrics_runs) == runs:
        first = metrics_runs[0]
        functional = all(
            _within(first.accuracy, m.accuracy, tolerance)
            and _within(first.latency_ms, m.latency_ms, tolerance)
            and _within(first.memory_mb, m.memory_mb, tolerance)
            for m in metrics_runs[1:]
        )
        if functional:
            return Reproducibility(level=ReproducibilityLevel.FUNCTIONALLY_REPRODUCIBLE, runs=runs,
                                   checksums=list(checksums), notes=f"metrics within {tolerance:.2%} tolerance")
        return Reproducibility(level=ReproducibilityLevel.NOT_REPRODUCIBLE, runs=runs, checksums=list(checksums),
                               notes=f"metrics diverge beyond {tolerance:.2%} tolerance")
    return Reproducibility(level=ReproducibilityLevel.NOT_REPRODUCIBLE, runs=runs, checksums=list(checksums),
                           notes="checksums differ and no metric evidence for functional equivalence")


def validate_reproducibility(repro: Reproducibility | None, criterion: CriterionConfig) -> CriterionResult:
    level = repro.level if repro else ReproducibilityLevel.NOT_VERIFIED
    runs = repro.runs if repro else 0
    threshold = "required" if criterion.required else "optional"
    if level in ACCEPTED:
        return CriterionResult(name=CRITERION, status=CriterionStatus.PASS, observed=level.value, threshold=threshold,
                               message=f"{level.value} over {runs} run(s)", required=criterion.required)
    if level == ReproducibilityLevel.NOT_VERIFIED:
        status = CriterionStatus.FAIL if criterion.required else CriterionStatus.NOT_APPLICABLE
        return CriterionResult(name=CRITERION, status=status, observed=level.value, threshold=threshold,
                               message="reproducibility not verified (fewer than two runs)", required=criterion.required)
    return CriterionResult(
        name=CRITERION,
        status=CriterionStatus.FAIL if criterion.required else CriterionStatus.WARN,
        observed=level.value,
        threshold=threshold,
        message=f"NOT_REPRODUCIBLE over {runs} run(s): {repro.notes if repro else ''}".rstrip(": "),
        required=criterion.required,
    )

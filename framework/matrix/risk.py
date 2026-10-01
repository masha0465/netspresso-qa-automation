"""Risk-based scoring and selection.

The score is an additive heuristic over configurable factor weights
(``configs/risk.yaml``). Each contributing factor is recorded so the framework
can *explain* why a case was selected. No statistical model is implied.
"""

from __future__ import annotations

from framework.config import FrameworkConfig, RiskConfig
from framework.matrix.generator import TestCase
from framework.pipeline.result import Configuration, RiskAssessment, RiskLevel


class RiskScorer:
    def __init__(self, config: FrameworkConfig) -> None:
        self._config = config
        self._risk: RiskConfig = config.risk

    def assess(self, configuration: Configuration) -> RiskAssessment:
        weights = self._risk.weights
        factors: list[str] = []
        score = 0

        sources = (
            ("model", self._config.model(configuration.model).risk),
            ("device", self._config.device(configuration.device).risk),
            ("runtime", self._config.runtime(configuration.runtime).risk),
            ("backend", self._config.backend(configuration.backend).risk),
            ("optimization", self._config.optimization(configuration.optimization).risk),
        )
        for dimension, flags in sources:
            for flag, enabled in sorted(flags.items()):
                weight = weights.get(flag, 0)
                if enabled and weight:
                    score += weight
                    factors.append(f"{dimension}.{flag} (+{weight})")

        if self._config.optimization(configuration.optimization).lossy and weights.get("lossy_precision"):
            w = weights["lossy_precision"]
            score += w
            factors.append(f"optimization.lossy_precision (+{w})")

        return RiskAssessment(score=score, level=self.level_for(score), factors=factors)

    def level_for(self, score: int) -> RiskLevel:
        if score >= self._risk.high_min_score:
            return RiskLevel.HIGH
        if score >= self._risk.medium_min_score:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def annotate(self, cases: list[TestCase]) -> None:
        for case in cases:
            case.risk = self.assess(case.configuration)


def select_by_risk(cases: list[TestCase], include_levels: tuple[RiskLevel, ...]) -> list[TestCase]:
    """Select feasible cases whose risk level is in ``include_levels``.

    Requires :meth:`RiskScorer.annotate` to have run first.
    """
    selected: list[TestCase] = []
    for case in cases:
        if not case.feasible or case.risk is None:
            continue
        if case.risk.level in include_levels:
            case.selected = True
            top = ", ".join(case.risk.factors[:3]) or "no factors"
            case.selection_reason = f"risk {case.risk.level.value} (score {case.risk.score}): {top}"
            selected.append(case)
    return selected


def risk_distribution(cases: list[TestCase]) -> dict[str, int]:
    dist = {level.value: 0 for level in RiskLevel}
    for case in cases:
        if case.risk is not None:
            dist[case.risk.level.value] += 1
    return dist

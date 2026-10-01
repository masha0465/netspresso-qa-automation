"""Configuration-driven Model x Device x Runtime x Backend x Optimization matrix."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from framework.config import FrameworkConfig
from framework.pipeline.result import Configuration, RiskAssessment, SupportState


@dataclass
class TestCase:
    configuration: Configuration
    support: SupportState
    support_reason: str | None = None
    risk: RiskAssessment | None = None
    selected: bool = False
    selection_reason: str | None = None

    @property
    def case_id(self) -> str:
        return self.configuration.case_id

    @property
    def feasible(self) -> bool:
        """A case can be executed unless it is explicitly UNSUPPORTED."""
        return self.support != SupportState.UNSUPPORTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            **self.configuration.to_dict(),
            "support": self.support.value,
            "support_reason": self.support_reason,
            "risk": self.risk.to_dict() if self.risk else None,
            "selected": self.selected,
            "selection_reason": self.selection_reason,
        }


@dataclass
class MatrixStats:
    total_combinations: int
    feasible_combinations: int
    unsupported_combinations: int
    unknown_support_combinations: int
    selected_combinations: int
    dimension_sizes: dict[str, int] = field(default_factory=dict)

    @property
    def reduction_percent(self) -> float:
        if self.feasible_combinations == 0:
            return 0.0
        return round(100.0 * (1 - self.selected_combinations / self.feasible_combinations), 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_combinations": self.total_combinations,
            "feasible_combinations": self.feasible_combinations,
            "unsupported_combinations": self.unsupported_combinations,
            "unknown_support_combinations": self.unknown_support_combinations,
            "selected_combinations": self.selected_combinations,
            "reduction_percent_vs_feasible": self.reduction_percent,
            "dimension_sizes": dict(self.dimension_sizes),
        }


def determine_support(config: FrameworkConfig, configuration: Configuration) -> tuple[SupportState, str | None]:
    """Static support decision.

    Only *explicit* knowledge yields UNSUPPORTED. Absence of evidence yields
    UNKNOWN (reported as NOT_TESTED until executed). SUPPORTED is never assumed here.
    """
    backend = config.backend(configuration.backend)
    if configuration.runtime not in backend.applicable_runtimes:
        return (
            SupportState.UNSUPPORTED,
            f"backend '{backend.name}' is not defined for runtime '{configuration.runtime}' (structural constraint)",
        )
    device = config.device(configuration.device)
    for rule in device.known_unsupported:
        if rule.matches(configuration.runtime, configuration.backend, configuration.optimization):
            return SupportState.UNSUPPORTED, f"device '{device.name}': {rule.reason}"
    return SupportState.UNKNOWN, None


class MatrixGenerator:
    """Builds the full Cartesian product in a stable, configuration-file order."""

    DIMENSIONS = ("model", "device", "runtime", "backend", "optimization")

    def __init__(self, config: FrameworkConfig) -> None:
        self._config = config

    @property
    def dimension_values(self) -> dict[str, list[str]]:
        c = self._config
        return {
            "model": [m.name for m in c.models],
            "device": [d.name for d in c.devices],
            "runtime": [r.name for r in c.runtimes],
            "backend": [b.name for b in c.backends],
            "optimization": [o.name for o in c.optimizations],
        }

    def full_product(self) -> list[TestCase]:
        values = self.dimension_values
        cases: list[TestCase] = []
        for combo in itertools.product(*(values[d] for d in self.DIMENSIONS)):
            configuration = Configuration(*combo)
            support, reason = determine_support(self._config, configuration)
            cases.append(TestCase(configuration=configuration, support=support, support_reason=reason))
        return cases

    def stats(self, cases: list[TestCase]) -> MatrixStats:
        return MatrixStats(
            total_combinations=len(cases),
            feasible_combinations=sum(1 for c in cases if c.feasible),
            unsupported_combinations=sum(1 for c in cases if c.support == SupportState.UNSUPPORTED),
            unknown_support_combinations=sum(1 for c in cases if c.support == SupportState.UNKNOWN),
            selected_combinations=sum(1 for c in cases if c.selected),
            dimension_sizes={k: len(v) for k, v in self.dimension_values.items()},
        )


def select_all_feasible(cases: list[TestCase]) -> list[TestCase]:
    """Strategy ``full``: every feasible combination is selected."""
    for case in cases:
        if case.feasible:
            case.selected = True
            case.selection_reason = "full matrix"
    return [c for c in cases if c.selected]

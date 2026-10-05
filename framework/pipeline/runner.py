"""Pipeline runner: matrix -> selection -> execution -> validation -> gate -> classification.

Status mapping (documented, deterministic):

==================  =================  ==================
Support / selected  Execution status   Case status
==================  =================  ==================
UNSUPPORTED         (not executed)     UNSUPPORTED
UNKNOWN, unselected (not executed)     NOT_TESTED
selected            COMPLETED, gate ok PASS
selected            COMPLETED, gate ko FAIL
selected            ERROR              FAIL
selected            BLOCKED            BLOCKED
selected            UNSUPPORTED        UNSUPPORTED (support updated to UNSUPPORTED)
selected            NOT_EXECUTED       NOT_TESTED
==================  =================  ==================
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from framework import __version__ as framework_version
from framework.adapters.base import BaseAdapter, ExecutionRequest
from framework.config import FrameworkConfig
from framework.credit_ledger import CreditLedger
from framework.defects import classify, distribution
from framework.matrix.generator import MatrixGenerator, TestCase, select_all_feasible
from framework.matrix.pairwise import PairwiseSelector
from framework.matrix.risk import RiskScorer, risk_distribution, select_by_risk
from framework.pipeline.result import (
    CaseResult,
    CreditUsageType,
    Environment,
    ExecutionResult,
    ExecutionStatus,
    QualityGateResult,
    Status,
    SupportState,
    utc_now_iso,
)
from framework.quality_gate import QualityGate

STRATEGIES = ("full", "pairwise", "risk", "pairwise_risk")


def _portable_path(path: Path) -> str:
    """Repo-relative POSIX path when possible, so reports never embed local user directories."""
    try:
        return Path(os.path.relpath(path, Path.cwd())).as_posix()
    except ValueError:  # e.g. different drive on Windows
        return path.name


class CreditSafetyError(RuntimeError):
    """Raised when a credit-consuming adapter is used without explicit confirmation."""


def case_status_for(execution: ExecutionResult, gate: QualityGateResult | None) -> Status:
    """Deterministic mapping from execution outcome (+ gate) to the case status (see module docstring)."""
    es = execution.execution_status
    if es == ExecutionStatus.COMPLETED:
        return Status.PASS if gate is not None and gate.passed else Status.FAIL
    if es == ExecutionStatus.ERROR:
        return Status.FAIL
    if es == ExecutionStatus.BLOCKED:
        return Status.BLOCKED
    if es == ExecutionStatus.UNSUPPORTED:
        return Status.UNSUPPORTED
    return Status.NOT_TESTED


@dataclass
class RunReport:
    run_id: str
    timestamp: str
    strategy: str
    environment: Environment
    matrix: dict[str, Any]
    selection: dict[str, Any]
    risk_distribution: dict[str, int]
    cases: list[CaseResult]
    framework_version: str = framework_version
    config_dir: str | None = None
    notes: list[str] = field(default_factory=list)

    # ---- derived summaries --------------------------------------------------
    def status_counts(self) -> dict[str, int]:
        counts = {s.value: 0 for s in Status}
        for c in self.cases:
            counts[c.status.value] += 1
        return counts

    @property
    def tested_cases(self) -> list[CaseResult]:
        return [c for c in self.cases if c.execution is not None and c.status in (Status.PASS, Status.FAIL, Status.BLOCKED)]

    def gate_counts(self) -> dict[str, int]:
        evaluated = [c for c in self.cases if c.gate is not None and c.gate.criteria]
        return {
            "evaluated": len(evaluated),
            "pass": sum(1 for c in evaluated if c.gate.passed),
            "fail": sum(1 for c in evaluated if not c.gate.passed),
        }

    def defect_distribution(self) -> dict[str, int]:
        return distribution([c.defect for c in self.cases])

    def failed_cases(self) -> list[CaseResult]:
        return [c for c in self.cases if c.status in (Status.FAIL, Status.BLOCKED)]

    def summary(self) -> dict[str, Any]:
        counts = self.status_counts()
        tested = counts["PASS"] + counts["FAIL"] + counts["BLOCKED"]
        return {
            "total_combinations": len(self.cases),
            "selected_combinations": sum(1 for c in self.cases if c.selected),
            "tested_combinations": tested,
            **counts,
            "quality_gate": self.gate_counts(),
            "defects": self.defect_distribution(),
            "coverage_percent_of_feasible": round(100.0 * tested / max(1, len(self.cases) - counts["UNSUPPORTED"]), 1),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "framework_version": self.framework_version,
            "strategy": self.strategy,
            "config_dir": self.config_dir,
            "environment": self.environment.to_dict(),
            "summary": self.summary(),
            "matrix": dict(self.matrix),
            "selection": dict(self.selection),
            "risk_distribution": dict(self.risk_distribution),
            "notes": list(self.notes),
            "cases": [c.to_dict() for c in self.cases],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunReport:
        return cls(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            strategy=data.get("strategy", "unknown"),
            environment=Environment.from_dict(data["environment"]),
            matrix=dict(data.get("matrix") or {}),
            selection=dict(data.get("selection") or {}),
            risk_distribution=dict(data.get("risk_distribution") or {}),
            cases=[CaseResult.from_dict(c) for c in data.get("cases", [])],
            framework_version=data.get("framework_version", "unknown"),
            config_dir=data.get("config_dir"),
            notes=list(data.get("notes") or []),
        )


class PipelineRunner:
    def __init__(
        self,
        config: FrameworkConfig,
        adapter: BaseAdapter,
        *,
        strategy: str = "pairwise",
        repeat_runs: int = 2,
        ledger: CreditLedger | None = None,
        allow_credit_consuming: bool = False,
        clock: Callable[[], str] | None = None,
    ) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy '{strategy}', expected one of {STRATEGIES}")
        if adapter.credit_consuming and not allow_credit_consuming:
            raise CreditSafetyError(
                f"adapter '{adapter.name}' may consume credits; pass allow_credit_consuming=True only after explicit user confirmation"
            )
        self._config = config
        self._adapter = adapter
        self._strategy = strategy
        self._repeat_runs = repeat_runs
        self._ledger = ledger
        self._clock = clock or utc_now_iso
        self._generator = MatrixGenerator(config)
        self._gate = QualityGate(config.quality_gate)
        self._scorer = RiskScorer(config)

    # ------------------------------------------------------------------ #
    def build_cases(self) -> tuple[list[TestCase], dict[str, Any]]:
        cases = self._generator.full_product()
        self._scorer.annotate(cases)
        selection = self._select(cases)
        return cases, selection

    def _select(self, cases: list[TestCase]) -> dict[str, Any]:
        info: dict[str, Any] = {"strategy": self._strategy}
        if self._strategy == "full":
            info["selected"] = len(select_all_feasible(cases))
        elif self._strategy == "pairwise":
            info.update(PairwiseSelector().select(cases).to_dict())
        elif self._strategy == "risk":
            selected = select_by_risk(cases, self._config.risk.include_levels)
            info.update({"selected": len(selected), "include_levels": [lvl.value for lvl in self._config.risk.include_levels]})
        else:  # pairwise_risk: union of both
            pw = PairwiseSelector().select(cases)
            before = {c.case_id for c in pw.selected}
            risk_selected = select_by_risk(cases, self._config.risk.include_levels)
            added = [c for c in risk_selected if c.case_id not in before]
            for c in added:
                c.selection_reason = f"risk-added: {c.selection_reason}"
            info.update(pw.to_dict())
            info.update({"strategy": "pairwise_risk", "risk_added": len(added), "selected": len(before) + len(added)})
        return info

    # ------------------------------------------------------------------ #
    def run(self) -> RunReport:
        cases, selection = self.build_cases()
        timestamp = self._clock()
        results = [self._evaluate(case) for case in cases]
        # Content-addressed id: identical inputs and outcomes -> identical id (deterministic),
        # while any change in outcomes (e.g. a different scenario set) yields a new id.
        fingerprint = "|".join(f"{c.case_id}:{c.status.value}" for c in results)
        run_id = "RUN-" + hashlib.sha1(  # noqa: S324 - identifier only
            f"{timestamp}|{self._strategy}|{self._adapter.name}|{fingerprint}".encode()
        ).hexdigest()[:10]
        report = RunReport(
            run_id=run_id,
            timestamp=timestamp,
            strategy=self._strategy,
            environment=self._adapter.describe_environment(),
            matrix=self._generator.stats(cases).to_dict(),
            selection=selection,
            risk_distribution=risk_distribution(cases),
            cases=results,
            config_dir=_portable_path(self._config.source_dir),
        )
        if self._ledger is not None:
            self._ledger.save()
        return report

    def _evaluate(self, case: TestCase) -> CaseResult:
        result = CaseResult(
            configuration=case.configuration,
            status=Status.NOT_TESTED,
            support=case.support,
            support_reason=case.support_reason,
            selected=case.selected,
            selection_reason=case.selection_reason,
            risk=case.risk,
        )
        if case.support == SupportState.UNSUPPORTED:
            result.status = Status.UNSUPPORTED
            return result
        if not case.selected:
            result.status = Status.NOT_TESTED
            return result

        request = ExecutionRequest(
            operation=case.configuration.optimization,
            configuration=case.configuration,
            repeat_runs=self._repeat_runs,
        )
        execution = self._adapter.run(request)
        result.execution = execution
        self._record_credit(execution)

        if execution.execution_status == ExecutionStatus.COMPLETED:
            result.gate = self._gate.evaluate(execution)
            result.support = SupportState.SUPPORTED
        elif execution.execution_status == ExecutionStatus.UNSUPPORTED:
            result.support = SupportState.UNSUPPORTED
            result.support_reason = execution.error.message if execution.error else "reported unsupported at run time"
        result.status = case_status_for(execution, result.gate)

        result.defect = classify(execution, result.gate)
        return result

    def _record_credit(self, execution) -> None:
        if self._ledger is None or execution.credit.usage_type == CreditUsageType.NONE:
            return
        self._ledger.record(
            operation=execution.operation,
            model=execution.configuration.model,
            purpose=f"pipeline run ({self._strategy}) case {execution.configuration.case_id}",
            usage_type=execution.credit.usage_type,
            estimated_credit=execution.credit.estimated,
            actual_credit=execution.credit.actual,
            result=execution.execution_status.value,
            error=execution.error.message if execution.error else None,
            confirmation=execution.credit.usage_type != CreditUsageType.SIMULATED and self._adapter.credit_consuming,
            adapter=self._adapter.name,
            timestamp=execution.timestamp,
        )

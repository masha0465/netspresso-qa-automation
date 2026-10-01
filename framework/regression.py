"""Baseline-run vs current-run regression comparison.

Definitions (no statistical significance is claimed; thresholds are explicit):

PASS
    Every compared metric is within threshold, the status did not degrade,
    artifact integrity did not degrade and the reproducibility level did not
    drop out of the accepted set.
REGRESSION
    At least one of: accuracy drop > threshold, latency/memory/model-size
    increase > threshold, status degraded (PASS -> FAIL/BLOCKED),
    artifact checksum became invalid, reproducibility level dropped out of the
    accepted set.
IMPROVED / UNCHANGED
    Per-metric labels only; they never override a REGRESSION.
NEW / MISSING / NOT_COMPARABLE
    Case exists in only one run, or lacks comparable metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from framework.config import QualityGateConfig
from framework.pipeline.result import CaseResult, ReproducibilityLevel, Status
from framework.pipeline.runner import RunReport
from framework.validation.accuracy import accuracy_drop_pp
from framework.validation.performance import relative_increase_percent
from framework.validation.reproducibility import ACCEPTED

_STATUS_RANK = {Status.PASS: 3, Status.NOT_TESTED: 2, Status.UNSUPPORTED: 2, Status.BLOCKED: 1, Status.FAIL: 0}


@dataclass(frozen=True)
class RegressionThresholds:
    accuracy_max_drop_pp: float = 1.0
    latency_max_increase_percent: float = 10.0
    memory_max_increase_percent: float = 15.0
    model_size_max_increase_percent: float = 10.0

    @classmethod
    def from_quality_gate(cls, gate: QualityGateConfig) -> RegressionThresholds:
        def th(name: str, default: float) -> float:
            c = gate.get(name)
            return c.threshold if c and c.threshold is not None else default

        return cls(
            accuracy_max_drop_pp=th("accuracy", 1.0),
            latency_max_increase_percent=th("latency", 10.0),
            memory_max_increase_percent=th("memory", 15.0),
            model_size_max_increase_percent=th("model_size", 10.0),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "accuracy_max_drop_pp": self.accuracy_max_drop_pp,
            "latency_max_increase_percent": self.latency_max_increase_percent,
            "memory_max_increase_percent": self.memory_max_increase_percent,
            "model_size_max_increase_percent": self.model_size_max_increase_percent,
        }


@dataclass
class MetricDelta:
    metric: str
    baseline: float | None
    current: float | None
    delta: float | None
    threshold: float
    unit: str
    result: str  # PASS | REGRESSION | IMPROVED | UNCHANGED | NOT_COMPARABLE

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class CaseComparison:
    case_id: str
    configuration: dict[str, str]
    baseline_status: str | None
    current_status: str | None
    result: str  # PASS | REGRESSION | NEW | MISSING | NOT_COMPARABLE
    deltas: list[MetricDelta] = field(default_factory=list)
    artifact_changed: bool | None = None
    reproducibility_change: list[str] | None = None
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "configuration": dict(self.configuration),
            "baseline_status": self.baseline_status,
            "current_status": self.current_status,
            "result": self.result,
            "deltas": [d.to_dict() for d in self.deltas],
            "artifact_changed": self.artifact_changed,
            "reproducibility_change": self.reproducibility_change,
            "reasons": list(self.reasons),
        }


@dataclass
class RegressionReport:
    baseline_run_id: str
    current_run_id: str
    baseline_timestamp: str
    current_timestamp: str
    thresholds: RegressionThresholds
    comparisons: list[CaseComparison]
    not_executed_in_both: int = 0

    @property
    def overall(self) -> str:
        return "REGRESSION" if any(c.result == "REGRESSION" for c in self.comparisons) else "PASS"

    def summary(self) -> dict[str, int]:
        counts = {"PASS": 0, "REGRESSION": 0, "NEW": 0, "MISSING": 0, "NOT_COMPARABLE": 0}
        for c in self.comparisons:
            counts[c.result] = counts.get(c.result, 0) + 1
        return counts

    def regressions(self) -> list[CaseComparison]:
        return [c for c in self.comparisons if c.result == "REGRESSION"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_run_id": self.baseline_run_id,
            "current_run_id": self.current_run_id,
            "baseline_timestamp": self.baseline_timestamp,
            "current_timestamp": self.current_timestamp,
            "overall": self.overall,
            "thresholds": self.thresholds.to_dict(),
            "summary": self.summary(),
            "not_executed_in_both": self.not_executed_in_both,
            "comparisons": [c.to_dict() for c in self.comparisons],
        }


class RegressionComparator:
    def __init__(self, thresholds: RegressionThresholds) -> None:
        self._t = thresholds

    def compare(self, baseline: RunReport, current: RunReport) -> RegressionReport:
        base_by_id = {c.case_id: c for c in baseline.cases}
        cur_by_id = {c.case_id: c for c in current.cases}
        comparisons: list[CaseComparison] = []
        skipped = 0
        for case_id in sorted(set(base_by_id) | set(cur_by_id)):
            b, c = base_by_id.get(case_id), cur_by_id.get(case_id)
            if b is not None and c is not None and b.execution is None and c.execution is None and b.status == c.status:
                skipped += 1  # not executed in either run and status unchanged -> nothing to compare
                continue
            comparisons.append(self._compare_case(case_id, b, c))
        return RegressionReport(
            not_executed_in_both=skipped,
            baseline_run_id=baseline.run_id,
            current_run_id=current.run_id,
            baseline_timestamp=baseline.timestamp,
            current_timestamp=current.timestamp,
            thresholds=self._t,
            comparisons=comparisons,
        )

    # ------------------------------------------------------------------ #
    def _compare_case(self, case_id: str, base: CaseResult | None, cur: CaseResult | None) -> CaseComparison:
        cfg = (cur or base).configuration.to_dict()  # type: ignore[union-attr]
        if base is None:
            return CaseComparison(case_id, cfg, None, cur.status.value, "NEW", reasons=["case not present in baseline run"])
        if cur is None:
            return CaseComparison(case_id, cfg, base.status.value, None, "MISSING", reasons=["case missing from current run"])

        comparison = CaseComparison(case_id, cfg, base.status.value, cur.status.value, "PASS")
        if _STATUS_RANK[cur.status] < _STATUS_RANK[base.status]:
            comparison.reasons.append(f"status degraded {base.status.value} -> {cur.status.value}")

        bm = base.execution.metrics if base.execution else None
        cm = cur.execution.metrics if cur.execution else None
        if bm is None or cm is None:
            if not comparison.reasons:
                comparison.result = "NOT_COMPARABLE"
                comparison.reasons.append("metrics unavailable in baseline and/or current run")
            else:
                comparison.result = "REGRESSION"
            return comparison

        comparison.deltas = [
            self._accuracy_delta(bm.accuracy, cm.accuracy),
            self._increase_delta("latency_ms", "ms", bm.latency_ms, cm.latency_ms, self._t.latency_max_increase_percent),
            self._increase_delta("memory_mb", "MB", bm.memory_mb, cm.memory_mb, self._t.memory_max_increase_percent),
            self._increase_delta("model_size_mb", "MB", bm.model_size_mb, cm.model_size_mb, self._t.model_size_max_increase_percent),
        ]
        comparison.reasons.extend(f"{d.metric}: {d.baseline} -> {d.current} (delta {d.delta}, threshold {d.threshold})"
                                  for d in comparison.deltas if d.result == "REGRESSION")

        ba, ca = base.execution.artifact, cur.execution.artifact
        if ba and ca:
            comparison.artifact_changed = ba.checksum_sha256 != ca.checksum_sha256
            if ba.checksum_valid and ca.checksum_valid is False:
                comparison.reasons.append("artifact checksum became invalid")

        br, cr = base.execution.reproducibility.level, cur.execution.reproducibility.level
        if br != cr:
            comparison.reproducibility_change = [br.value, cr.value]
            if br in ACCEPTED and cr not in ACCEPTED and cr != ReproducibilityLevel.NOT_VERIFIED:
                comparison.reasons.append(f"reproducibility degraded {br.value} -> {cr.value}")

        comparison.result = "REGRESSION" if comparison.reasons else "PASS"
        return comparison

    def _accuracy_delta(self, b: float | None, c: float | None) -> MetricDelta:
        if b is None or c is None:
            return MetricDelta("accuracy", b, c, None, self._t.accuracy_max_drop_pp, "pp", "NOT_COMPARABLE")
        drop = accuracy_drop_pp(_m(b), _m(c))  # positive = worse
        result = "REGRESSION" if drop > self._t.accuracy_max_drop_pp else ("IMPROVED" if drop < 0 else ("UNCHANGED" if drop == 0 else "PASS"))
        return MetricDelta("accuracy", b, c, drop, self._t.accuracy_max_drop_pp, "pp (drop)", result)

    @staticmethod
    def _increase_delta(name: str, unit: str, b: float | None, c: float | None, threshold: float) -> MetricDelta:
        inc = relative_increase_percent(b, c)
        if inc is None:
            return MetricDelta(name, b, c, None, threshold, unit, "NOT_COMPARABLE")
        result = "REGRESSION" if inc > threshold else ("IMPROVED" if inc < 0 else ("UNCHANGED" if inc == 0 else "PASS"))
        return MetricDelta(name, b, c, inc, threshold, f"% ({unit})", result)


def _m(accuracy: float):
    from framework.pipeline.result import Metrics

    return Metrics(accuracy=accuracy)

"""Common Result Model shared by every adapter and every QA stage.

Design rules
------------
* Adapter-agnostic: nothing in this module refers to NetsPresso internals.
  A future ``NetsPressoAdapter`` (Python 3.11) maps SDK metadata into these
  structures and writes them as JSON; the main framework (Python 3.14) reads them
  back with :func:`ExecutionResult.from_dict`.
* Two layers are deliberately kept apart:

  - :class:`ExecutionResult` – *what happened* when an adapter executed one
    configuration (raw facts: metrics, artifact, error, environment).
  - :class:`CaseResult` – *the QA verdict* for that configuration after
    validation, Quality Gate and defect classification.

  A completed execution is not automatically a PASS, and a failed execution is
  not automatically a classified defect.
* Every enum is a ``str`` enum so JSON round-trips are trivial and stable.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class Status(enum.StrEnum):
    """Final QA status of a matrix cell.

    NOT_TESTED and UNSUPPORTED are intentionally distinct:
    * NOT_TESTED  – support may exist; the combination simply has not been verified.
    * UNSUPPORTED – the combination is known / explicitly determined to be unsupported.
    Neither of them is a PASS, and UNSUPPORTED is not a FAIL.
    """

    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    NOT_TESTED = "NOT_TESTED"
    UNSUPPORTED = "UNSUPPORTED"


class ExecutionStatus(enum.StrEnum):
    """Raw outcome reported by an adapter (before the Quality Gate)."""

    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    BLOCKED = "BLOCKED"
    UNSUPPORTED = "UNSUPPORTED"
    NOT_EXECUTED = "NOT_EXECUTED"


class SupportState(enum.StrEnum):
    """Static support knowledge for a combination, before execution."""

    SUPPORTED = "SUPPORTED"  # verified by a previous execution or authoritative source
    UNSUPPORTED = "UNSUPPORTED"  # explicitly documented as unsupported
    UNKNOWN = "UNKNOWN"  # no evidence either way -> NOT_TESTED until executed


class DefectCategory(enum.StrEnum):
    AUTH_ERROR = "AUTH_ERROR"
    INPUT_VALIDATION_ERROR = "INPUT_VALIDATION_ERROR"
    MODEL_COMPATIBILITY_ERROR = "MODEL_COMPATIBILITY_ERROR"
    OPTIMIZATION_FAILURE = "OPTIMIZATION_FAILURE"
    CONVERSION_FAILURE = "CONVERSION_FAILURE"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    ACCURACY_REGRESSION = "ACCURACY_REGRESSION"
    PERFORMANCE_REGRESSION = "PERFORMANCE_REGRESSION"
    MEMORY_REGRESSION = "MEMORY_REGRESSION"
    ARTIFACT_ERROR = "ARTIFACT_ERROR"
    REPRODUCIBILITY_ERROR = "REPRODUCIBILITY_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    ENVIRONMENT_ERROR = "ENVIRONMENT_ERROR"
    UNCLASSIFIED = "UNCLASSIFIED"  # evidence insufficient; never guessed


class Severity(enum.StrEnum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ReproducibilityLevel(enum.StrEnum):
    BITWISE_REPRODUCIBLE = "BITWISE_REPRODUCIBLE"
    FUNCTIONALLY_REPRODUCIBLE = "FUNCTIONALLY_REPRODUCIBLE"
    NOT_REPRODUCIBLE = "NOT_REPRODUCIBLE"
    NOT_VERIFIED = "NOT_VERIFIED"


class RiskLevel(enum.StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class CreditUsageType(enum.StrEnum):
    """How a credit figure was obtained. ``ACTUAL`` may only come from a real account query."""

    SIMULATED = "simulated"
    ESTIMATED = "estimated"
    ACTUAL = "actual"
    NONE = "none"


class CriterionStatus(enum.StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"  # failed but criterion is not mandatory
    NOT_APPLICABLE = "NOT_APPLICABLE"  # data missing; reported, never hidden


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, order=True)
class Configuration:
    """One cell of the Model x Device x Runtime x Backend x Optimization matrix."""

    model: str
    device: str
    runtime: str
    backend: str
    optimization: str

    @property
    def key(self) -> str:
        return "|".join((self.model, self.device, self.runtime, self.backend, self.optimization))

    @property
    def case_id(self) -> str:
        digest = hashlib.sha1(self.key.encode("utf-8")).hexdigest()[:8]  # noqa: S324 - identifier only
        return f"TC-{digest}"

    def to_dict(self) -> dict[str, str]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Configuration:
        return cls(
            model=data["model"],
            device=data["device"],
            runtime=data["runtime"],
            backend=data["backend"],
            optimization=data["optimization"],
        )


@dataclass
class Metrics:
    """Generic metric fields. Semantics are defined by the validators, not by any vendor."""

    accuracy: float | None = None  # 0.0 - 1.0 (metric name is model-specific, e.g. mAP50 / top1)
    latency_ms: float | None = None
    memory_mb: float | None = None
    model_size_mb: float | None = None
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Metrics | None:
        if data is None:
            return None
        return cls(
            accuracy=data.get("accuracy"),
            latency_ms=data.get("latency_ms"),
            memory_mb=data.get("memory_mb"),
            model_size_mb=data.get("model_size_mb"),
            extra=dict(data.get("extra") or {}),
        )


@dataclass
class Artifact:
    """Deployment / optimized artifact facts used by artifact validation and traceability."""

    path: str | None = None
    exists: bool = False
    size_bytes: int | None = None
    format: str | None = None  # e.g. "onnx", "tflite", "engine"
    checksum_sha256: str | None = None
    expected_checksum_sha256: str | None = None
    checksum_valid: bool | None = None  # None = not verifiable
    metadata: dict[str, Any] = field(default_factory=dict)  # source model id, sdk version, config ...

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Artifact | None:
        if data is None:
            return None
        return cls(**{k: data.get(k) for k in ("path", "size_bytes", "format", "checksum_sha256",
                                                 "expected_checksum_sha256", "checksum_valid")},
                   exists=bool(data.get("exists", False)),
                   metadata=dict(data.get("metadata") or {}))


@dataclass
class Environment:
    adapter: str
    adapter_version: str
    sdk_name: str | None = None
    sdk_version: str | None = None
    python_version: str | None = None
    platform: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Environment:
        return cls(
            adapter=data["adapter"],
            adapter_version=data.get("adapter_version", "unknown"),
            sdk_name=data.get("sdk_name"),
            sdk_version=data.get("sdk_version"),
            python_version=data.get("python_version"),
            platform=data.get("platform"),
            extra=dict(data.get("extra") or {}),
        )


@dataclass
class Reproducibility:
    level: ReproducibilityLevel = ReproducibilityLevel.NOT_VERIFIED
    runs: int = 0
    checksums: list[str] = field(default_factory=list)
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level.value, "runs": self.runs, "checksums": list(self.checksums), "notes": self.notes}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Reproducibility:
        if data is None:
            return cls()
        return cls(
            level=ReproducibilityLevel(data.get("level", ReproducibilityLevel.NOT_VERIFIED)),
            runs=int(data.get("runs", 0)),
            checksums=list(data.get("checksums") or []),
            notes=data.get("notes"),
        )


@dataclass
class ExecutionError:
    """Raw error facts from an adapter.

    ``kind`` is a *hint* (e.g. an SDK exception class name or an adapter-level
    label). It is evidence for :mod:`framework.defects`, not a classification.
    """

    message: str
    kind: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ExecutionError | None:
        if data is None:
            return None
        return cls(message=data.get("message", ""), kind=data.get("kind"), details=dict(data.get("details") or {}))


@dataclass
class CreditRecord:
    """Credit information attached to an execution. Never fabricate ``ACTUAL`` values."""

    usage_type: CreditUsageType = CreditUsageType.NONE
    estimated: int | None = None
    actual: int | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"usage_type": self.usage_type.value, "estimated": self.estimated, "actual": self.actual, "note": self.note}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CreditRecord:
        if data is None:
            return cls()
        return cls(
            usage_type=CreditUsageType(data.get("usage_type", CreditUsageType.NONE)),
            estimated=data.get("estimated"),
            actual=data.get("actual"),
            note=data.get("note"),
        )


@dataclass
class Defect:
    """Structured defect classification (see framework/defects.py)."""

    category: DefectCategory
    severity: Severity
    message: str
    evidence: list[str] = field(default_factory=list)
    suspected_cause: str | None = None  # "suspected", never asserted as root cause
    reproducibility: str = "unknown"  # deterministic | intermittent | unknown
    affected_configuration: dict[str, str] = field(default_factory=dict)
    secondary_categories: list[DefectCategory] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "evidence": list(self.evidence),
            "suspected_cause": self.suspected_cause,
            "reproducibility": self.reproducibility,
            "affected_configuration": dict(self.affected_configuration),
            "secondary_categories": [c.value for c in self.secondary_categories],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Defect | None:
        if data is None:
            return None
        return cls(
            category=DefectCategory(data["category"]),
            severity=Severity(data["severity"]),
            message=data.get("message", ""),
            evidence=list(data.get("evidence") or []),
            suspected_cause=data.get("suspected_cause"),
            reproducibility=data.get("reproducibility", "unknown"),
            affected_configuration=dict(data.get("affected_configuration") or {}),
            secondary_categories=[DefectCategory(c) for c in data.get("secondary_categories") or []],
        )


@dataclass
class CriterionResult:
    name: str
    status: CriterionStatus
    observed: float | str | None
    threshold: float | str | None
    message: str
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "observed": self.observed,
            "threshold": self.threshold,
            "message": self.message,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CriterionResult:
        return cls(
            name=data["name"],
            status=CriterionStatus(data["status"]),
            observed=data.get("observed"),
            threshold=data.get("threshold"),
            message=data.get("message", ""),
            required=bool(data.get("required", True)),
        )


@dataclass
class QualityGateResult:
    overall: Status  # PASS or FAIL only
    criteria: list[CriterionResult]
    reasons: list[str] = field(default_factory=list)  # human readable failure reasons, never hidden

    @property
    def passed(self) -> bool:
        return self.overall == Status.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall.value,
            "criteria": [c.to_dict() for c in self.criteria],
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> QualityGateResult | None:
        if data is None:
            return None
        return cls(
            overall=Status(data["overall"]),
            criteria=[CriterionResult.from_dict(c) for c in data.get("criteria", [])],
            reasons=list(data.get("reasons") or []),
        )


# --------------------------------------------------------------------------- #
# Execution result (adapter output)
# --------------------------------------------------------------------------- #
def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass
class ExecutionResult:
    """Everything an adapter knows after executing one configuration once."""

    operation: str
    configuration: Configuration
    execution_status: ExecutionStatus
    environment: Environment
    execution_time_s: float = 0.0
    baseline_metrics: Metrics | None = None
    metrics: Metrics | None = None
    artifact: Artifact | None = None
    logs: list[str] = field(default_factory=list)
    error: ExecutionError | None = None
    reproducibility: Reproducibility = field(default_factory=Reproducibility)
    timestamp: str = field(default_factory=utc_now_iso)
    credit: CreditRecord = field(default_factory=CreditRecord)
    scenario_id: str | None = None  # mock scenario reference; None for real runs

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "configuration": self.configuration.to_dict(),
            "execution_status": self.execution_status.value,
            "execution_time_s": self.execution_time_s,
            "baseline_metrics": self.baseline_metrics.to_dict() if self.baseline_metrics else None,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "artifact": self.artifact.to_dict() if self.artifact else None,
            "logs": list(self.logs),
            "error": self.error.to_dict() if self.error else None,
            "environment": self.environment.to_dict(),
            "reproducibility": self.reproducibility.to_dict(),
            "timestamp": self.timestamp,
            "credit": self.credit.to_dict(),
            "scenario_id": self.scenario_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionResult:
        return cls(
            operation=data["operation"],
            configuration=Configuration.from_dict(data["configuration"]),
            execution_status=ExecutionStatus(data["execution_status"]),
            environment=Environment.from_dict(data["environment"]),
            execution_time_s=float(data.get("execution_time_s", 0.0)),
            baseline_metrics=Metrics.from_dict(data.get("baseline_metrics")),
            metrics=Metrics.from_dict(data.get("metrics")),
            artifact=Artifact.from_dict(data.get("artifact")),
            logs=list(data.get("logs") or []),
            error=ExecutionError.from_dict(data.get("error")),
            reproducibility=Reproducibility.from_dict(data.get("reproducibility")),
            timestamp=data.get("timestamp", utc_now_iso()),
            credit=CreditRecord.from_dict(data.get("credit")),
            scenario_id=data.get("scenario_id"),
        )


# --------------------------------------------------------------------------- #
# Case result (QA verdict)
# --------------------------------------------------------------------------- #
@dataclass
class RiskAssessment:
    score: int
    level: RiskLevel
    factors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"score": self.score, "level": self.level.value, "factors": list(self.factors)}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RiskAssessment | None:
        if data is None:
            return None
        return cls(score=int(data["score"]), level=RiskLevel(data["level"]), factors=list(data.get("factors") or []))


@dataclass
class CaseResult:
    """QA verdict for one matrix cell.

    ``status`` is the *test result*; ``defect`` is the *defect classification*.
    They are related but not the same thing: a FAIL may carry an UNCLASSIFIED
    defect when evidence is insufficient, and a BLOCKED case may carry an
    ENVIRONMENT_ERROR defect even though no test verdict was reached.
    """

    configuration: Configuration
    status: Status
    support: SupportState = SupportState.UNKNOWN
    support_reason: str | None = None
    selected: bool = False
    selection_reason: str | None = None
    risk: RiskAssessment | None = None
    execution: ExecutionResult | None = None
    gate: QualityGateResult | None = None
    defect: Defect | None = None

    @property
    def case_id(self) -> str:
        return self.configuration.case_id

    def to_dict(self) -> dict[str, Any]:
        ex = self.execution
        flat: dict[str, Any] = {
            "case_id": self.case_id,
            "status": self.status.value,
            "support": self.support.value,
            "support_reason": self.support_reason,
            "selected": self.selected,
            "selection_reason": self.selection_reason,
            "risk": self.risk.to_dict() if self.risk else None,
            # flattened convenience fields (stable for report consumers)
            "operation": ex.operation if ex else None,
            **self.configuration.to_dict(),
            "execution_time_s": ex.execution_time_s if ex else None,
            "metrics": ex.metrics.to_dict() if ex and ex.metrics else None,
            "baseline_metrics": ex.baseline_metrics.to_dict() if ex and ex.baseline_metrics else None,
            "artifact": ex.artifact.to_dict() if ex and ex.artifact else None,
            "logs": list(ex.logs) if ex else [],
            "error": ex.error.to_dict() if ex and ex.error else None,
            "defect_category": self.defect.category.value if self.defect else None,
            "environment": ex.environment.to_dict() if ex else None,
            "reproducibility": ex.reproducibility.to_dict() if ex else None,
            "timestamp": ex.timestamp if ex else None,
            # full nested objects
            "execution": ex.to_dict() if ex else None,
            "quality_gate": self.gate.to_dict() if self.gate else None,
            "defect": self.defect.to_dict() if self.defect else None,
        }
        return flat

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaseResult:
        return cls(
            configuration=Configuration.from_dict(data),
            status=Status(data["status"]),
            support=SupportState(data.get("support", SupportState.UNKNOWN)),
            support_reason=data.get("support_reason"),
            selected=bool(data.get("selected", False)),
            selection_reason=data.get("selection_reason"),
            risk=RiskAssessment.from_dict(data.get("risk")),
            execution=ExecutionResult.from_dict(data["execution"]) if data.get("execution") else None,
            gate=QualityGateResult.from_dict(data.get("quality_gate")),
            defect=Defect.from_dict(data.get("defect")),
        )

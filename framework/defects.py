"""Structured, evidence-based defect classification.

Test result vs defect classification
------------------------------------
* The **test result** (:class:`~framework.pipeline.result.Status`) answers
  "did this configuration meet the release criteria?".
* The **defect classification** (:class:`~framework.pipeline.result.Defect`)
  answers "what kind of problem does the evidence point to?".

A FAIL may carry an UNCLASSIFIED defect when the evidence is insufficient; a
BLOCKED case may carry an ENVIRONMENT_ERROR defect although no verdict was
reached. The classifier never infers a category from free-text guesses: it
only uses (1) Quality Gate criterion outcomes and (2) explicit error kinds,
including a documented mapping of SDK exception class names observed in the
Phase 1 source inspection (netspresso==1.17.0).
"""

from __future__ import annotations

from framework.pipeline.result import (
    CriterionStatus,
    Defect,
    DefectCategory,
    ExecutionResult,
    ExecutionStatus,
    QualityGateResult,
    ReproducibilityLevel,
    Severity,
)

# Quality-gate criterion -> (category, severity). Order = priority for the primary category.
_CRITERION_MAP: tuple[tuple[str, DefectCategory, Severity], ...] = (
    ("accuracy", DefectCategory.ACCURACY_REGRESSION, Severity.HIGH),
    # proxy evidence points at the accuracy impact area; the message keeps the word "proxy"
    ("output_equivalence_proxy", DefectCategory.ACCURACY_REGRESSION, Severity.MEDIUM),
    ("artifact", DefectCategory.ARTIFACT_ERROR, Severity.HIGH),
    ("structural_validity", DefectCategory.ARTIFACT_ERROR, Severity.HIGH),
    ("reproducibility", DefectCategory.REPRODUCIBILITY_ERROR, Severity.MEDIUM),
    ("memory", DefectCategory.MEMORY_REGRESSION, Severity.MEDIUM),
    ("latency", DefectCategory.PERFORMANCE_REGRESSION, Severity.MEDIUM),
    ("model_size", DefectCategory.PERFORMANCE_REGRESSION, Severity.LOW),
)

# Explicit error-kind hints. Keys are exact strings an adapter may put in ExecutionError.kind.
# SDK exception names come from netspresso/exceptions/*.py (Phase 1 inspection).
_ERROR_KIND_MAP: dict[str, DefectCategory] = {
    # adapter-level labels (already categories)
    **{c.value: c for c in DefectCategory},
    # authentication (SDK has no dedicated exception; adapter labels HTTP 401/403 as this)
    "LoginFailed": DefectCategory.AUTH_ERROR,
    "Unauthorized": DefectCategory.AUTH_ERROR,
    # input validation
    "NotValidInputModelPath": DefectCategory.INPUT_VALIDATION_ERROR,
    "NotValidSlampRatioException": DefectCategory.INPUT_VALIDATION_ERROR,
    "NotValidVbmfRatioException": DefectCategory.INPUT_VALIDATION_ERROR,
    "NotValidChannelAxisRangeException": DefectCategory.INPUT_VALIDATION_ERROR,
    "NotFillInputLayersException": DefectCategory.INPUT_VALIDATION_ERROR,
    "NotSetDatasetException": DefectCategory.INPUT_VALIDATION_ERROR,
    "NotSetModelException": DefectCategory.INPUT_VALIDATION_ERROR,
    "EmptyCompressionParamsException": DefectCategory.INPUT_VALIDATION_ERROR,
    "FileNotFoundErrorException": DefectCategory.INPUT_VALIDATION_ERROR,
    "DirectoryNotFoundException": DefectCategory.INPUT_VALIDATION_ERROR,
    # compatibility
    "NotSupportedFrameworkException": DefectCategory.MODEL_COMPATIBILITY_ERROR,
    "NotSupportedModelException": DefectCategory.MODEL_COMPATIBILITY_ERROR,
    "NotSupportedSuffixException": DefectCategory.MODEL_COMPATIBILITY_ERROR,
    "NotSupportedTaskException": DefectCategory.MODEL_COMPATIBILITY_ERROR,
    "LoadONNXModelException": DefectCategory.MODEL_COMPATIBILITY_ERROR,
    # runtime / server
    "GatewayTimeoutException": DefectCategory.RUNTIME_ERROR,
    "InternalServerErrorException": DefectCategory.RUNTIME_ERROR,
    "FailedUploadModelException": DefectCategory.RUNTIME_ERROR,
    # environment / configuration
    "NotEnoughCreditException": DefectCategory.ENVIRONMENT_ERROR,
    "FailedFetchPackageException": DefectCategory.ENVIRONMENT_ERROR,
    "ConfigurationError": DefectCategory.CONFIGURATION_ERROR,
    "TaskOrYamlPathException": DefectCategory.CONFIGURATION_ERROR,
}

_ERROR_SEVERITY: dict[DefectCategory, Severity] = {
    DefectCategory.AUTH_ERROR: Severity.CRITICAL,
    DefectCategory.CONVERSION_FAILURE: Severity.HIGH,
    DefectCategory.OPTIMIZATION_FAILURE: Severity.HIGH,
    DefectCategory.RUNTIME_ERROR: Severity.HIGH,
    DefectCategory.MODEL_COMPATIBILITY_ERROR: Severity.MEDIUM,
    DefectCategory.INPUT_VALIDATION_ERROR: Severity.MEDIUM,
    DefectCategory.CONFIGURATION_ERROR: Severity.MEDIUM,
    DefectCategory.ENVIRONMENT_ERROR: Severity.MEDIUM,
    DefectCategory.UNCLASSIFIED: Severity.MEDIUM,
}


def _reproducibility_label(execution: ExecutionResult) -> str:
    repro = execution.reproducibility
    if repro.runs >= 2 and repro.level in (
        ReproducibilityLevel.BITWISE_REPRODUCIBLE,
        ReproducibilityLevel.FUNCTIONALLY_REPRODUCIBLE,
    ):
        return "deterministic"
    return "unknown"


def _suspected_cause(category: DefectCategory, execution: ExecutionResult) -> str | None:
    """Only offer a *suspected* cause when the configuration itself supplies a plausible mechanism."""
    optimization = execution.configuration.optimization
    if category == DefectCategory.ACCURACY_REGRESSION and "int8" in optimization:
        return "INT8 quantization precision loss / calibration coverage (suspected, not confirmed)"
    if category == DefectCategory.ACCURACY_REGRESSION and "compression" in optimization:
        return "pruning without fine-tuning / ratio too aggressive for this model (suspected, not confirmed)"
    return None


def classify(execution: ExecutionResult, gate: QualityGateResult | None) -> Defect | None:
    """Return a :class:`Defect` or ``None`` when the evidence shows no defect."""
    affected = execution.configuration.to_dict()

    if execution.execution_status == ExecutionStatus.COMPLETED:
        if gate is None or gate.passed:
            return None
        return _classify_gate_failure(execution, gate, affected)

    if execution.execution_status == ExecutionStatus.UNSUPPORTED:
        return None  # unsupported is a matrix fact, not a defect

    return _classify_execution_error(execution, affected)


def _classify_gate_failure(execution: ExecutionResult, gate: QualityGateResult, affected: dict[str, str]) -> Defect:
    failed = {c.name: c for c in gate.criteria if c.status == CriterionStatus.FAIL and c.required}
    missing = [c for c in gate.criteria if c.status == CriterionStatus.NOT_APPLICABLE and c.required]
    hits = [(cat, sev, failed[name]) for name, cat, sev in _CRITERION_MAP if name in failed]
    if not hits:
        # A required criterion that could not be evaluated is missing evidence, not a measured regression.
        message = (
            "required criteria could not be evaluated (missing data): " + ", ".join(c.name for c in missing)
            if missing
            else "Quality Gate failed without a mapped criterion"
        )
        return Defect(
            category=DefectCategory.UNCLASSIFIED,
            severity=Severity.MEDIUM,
            message=message,
            evidence=list(gate.reasons),
            suspected_cause=None,
            reproducibility=_reproducibility_label(execution),
            affected_configuration=affected,
        )
    primary_cat, primary_sev, primary_crit = hits[0]
    return Defect(
        category=primary_cat,
        severity=primary_sev,
        message=primary_crit.message,
        evidence=[f"{c.name}: {c.message} (observed={c.observed}, threshold={c.threshold})" for _, _, c in hits],
        suspected_cause=_suspected_cause(primary_cat, execution),
        reproducibility=_reproducibility_label(execution),
        affected_configuration=affected,
        secondary_categories=[cat for cat, _, _ in hits[1:] if cat != primary_cat],
    )


def _classify_execution_error(execution: ExecutionResult, affected: dict[str, str]) -> Defect:
    error = execution.error
    kind = error.kind if error else None
    category = _ERROR_KIND_MAP.get(kind) if kind else None
    evidence: list[str] = [f"execution_status={execution.execution_status.value}"]
    if error:
        evidence.append(f"error.kind={kind!r}")
        evidence.append(f"error.message={error.message!r}")
    evidence.extend(execution.logs[-3:])

    if category is None:
        return Defect(
            category=DefectCategory.UNCLASSIFIED,
            severity=Severity.MEDIUM,
            message=(error.message if error else "execution did not complete") + " - insufficient evidence to classify",
            evidence=evidence,
            suspected_cause=None,
            reproducibility="unknown",
            affected_configuration=affected,
        )
    return Defect(
        category=category,
        severity=_ERROR_SEVERITY.get(category, Severity.MEDIUM),
        message=error.message if error else category.value,
        evidence=evidence,
        suspected_cause=None,
        reproducibility="unknown",
        affected_configuration=affected,
    )


def distribution(defects: list[Defect | None]) -> dict[str, int]:
    counts = {c.value: 0 for c in DefectCategory}
    for d in defects:
        if d is not None:
            counts[d.category.value] += 1
    return {k: v for k, v in counts.items() if v}

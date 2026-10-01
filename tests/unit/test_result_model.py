import pytest

from framework.pipeline.result import (
    Artifact,
    CaseResult,
    Configuration,
    CreditRecord,
    CreditUsageType,
    CriterionResult,
    CriterionStatus,
    Defect,
    DefectCategory,
    Environment,
    ExecutionError,
    ExecutionResult,
    ExecutionStatus,
    Metrics,
    QualityGateResult,
    Reproducibility,
    ReproducibilityLevel,
    RiskAssessment,
    RiskLevel,
    Severity,
    Status,
    SupportState,
)

pytestmark = pytest.mark.unit

CFG = Configuration("yolov8n", "intel_xeon_w2233", "onnxruntime", "default", "int8_quantization")


def test_configuration_key_and_case_id_are_stable():
    assert CFG.key == "yolov8n|intel_xeon_w2233|onnxruntime|default|int8_quantization"
    assert CFG.case_id == Configuration.from_dict(CFG.to_dict()).case_id
    assert CFG.case_id.startswith("TC-") and len(CFG.case_id) == 11


def test_status_model_keeps_not_tested_and_unsupported_distinct():
    assert Status.NOT_TESTED != Status.UNSUPPORTED
    assert {s.value for s in Status} == {"PASS", "FAIL", "BLOCKED", "NOT_TESTED", "UNSUPPORTED"}
    assert SupportState.UNKNOWN != SupportState.UNSUPPORTED


def test_defect_categories_cover_required_set():
    required = {
        "AUTH_ERROR", "INPUT_VALIDATION_ERROR", "MODEL_COMPATIBILITY_ERROR", "OPTIMIZATION_FAILURE",
        "CONVERSION_FAILURE", "RUNTIME_ERROR", "ACCURACY_REGRESSION", "PERFORMANCE_REGRESSION",
        "MEMORY_REGRESSION", "ARTIFACT_ERROR", "REPRODUCIBILITY_ERROR", "CONFIGURATION_ERROR",
        "ENVIRONMENT_ERROR", "UNCLASSIFIED",
    }
    assert {c.value for c in DefectCategory} == required


def _execution() -> ExecutionResult:
    return ExecutionResult(
        operation="int8_quantization",
        configuration=CFG,
        execution_status=ExecutionStatus.COMPLETED,
        environment=Environment(adapter="mock", adapter_version="1.0.0", python_version="3.14", platform="test"),
        execution_time_s=1.25,
        baseline_metrics=Metrics(accuracy=0.52, latency_ms=42.0, memory_mb=96.0, model_size_mb=12.4),
        metrics=Metrics(accuracy=0.514, latency_ms=20.0, memory_mb=60.0, model_size_mb=3.4, extra={"flops_g": 1.2}),
        artifact=Artifact(path="mock://a.onnx", exists=True, size_bytes=3_400_000, format="onnx",
                          checksum_sha256="ab", expected_checksum_sha256="ab", checksum_valid=True, metadata={"k": "v"}),
        logs=["step 1", "step 2"],
        error=None,
        reproducibility=Reproducibility(level=ReproducibilityLevel.BITWISE_REPRODUCIBLE, runs=2, checksums=["ab", "ab"]),
        timestamp="2026-10-01T00:00:00+00:00",
        credit=CreditRecord(usage_type=CreditUsageType.SIMULATED, estimated=50),
        scenario_id=None,
    )


def test_execution_result_json_round_trip():
    ex = _execution()
    data = ex.to_dict()
    assert data["execution_status"] == "COMPLETED"
    assert data["credit"]["usage_type"] == "simulated"
    restored = ExecutionResult.from_dict(data)
    assert restored.to_dict() == data
    assert restored.configuration == CFG
    assert restored.metrics.extra == {"flops_g": 1.2}


def test_execution_result_with_error_round_trip():
    ex = _execution()
    ex.execution_status = ExecutionStatus.ERROR
    ex.metrics = None
    ex.error = ExecutionError(message="boom", kind="CONVERSION_FAILURE", details={"code": 42})
    restored = ExecutionResult.from_dict(ex.to_dict())
    assert restored.error.kind == "CONVERSION_FAILURE"
    assert restored.metrics is None


def test_case_result_flattens_required_fields_and_round_trips():
    ex = _execution()
    gate = QualityGateResult(
        overall=Status.FAIL,
        criteria=[CriterionResult("accuracy", CriterionStatus.FAIL, 3.2, 1.0, "drop too big")],
        reasons=["Accuracy: drop too big"],
    )
    defect = Defect(category=DefectCategory.ACCURACY_REGRESSION, severity=Severity.HIGH, message="m",
                    evidence=["e"], suspected_cause=None, affected_configuration=CFG.to_dict())
    case = CaseResult(configuration=CFG, status=Status.FAIL, support=SupportState.SUPPORTED, selected=True,
                      selection_reason="pairwise", risk=RiskAssessment(9, RiskLevel.HIGH, ["x"]),
                      execution=ex, gate=gate, defect=defect)
    flat = case.to_dict()
    for key in ("operation", "model", "device", "runtime", "backend", "status", "execution_time_s", "metrics",
                "artifact", "logs", "error", "defect_category", "environment", "reproducibility", "timestamp"):
        assert key in flat, key
    assert flat["defect_category"] == "ACCURACY_REGRESSION"
    restored = CaseResult.from_dict(flat)
    assert restored.to_dict() == flat


def test_credit_record_never_defaults_to_actual():
    assert CreditRecord().usage_type == CreditUsageType.NONE
    assert CreditRecord.from_dict({"usage_type": "simulated", "estimated": 25}).actual is None

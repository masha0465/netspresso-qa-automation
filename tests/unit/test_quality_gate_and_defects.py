import pytest

from framework.adapters.base import ExecutionRequest
from framework.config import CriterionConfig, QualityGateConfig
from framework.defects import classify, distribution
from framework.pipeline.result import (
    Configuration,
    CriterionStatus,
    DefectCategory,
    ExecutionError,
    ExecutionStatus,
    Metrics,
    Severity,
    Status,
)
from framework.quality_gate import QualityGate, render_text

pytestmark = pytest.mark.unit


def _run(adapter, model, device, runtime, backend, opt, repeat=2):
    cfg = Configuration(model, device, runtime, backend, opt)
    return adapter.run(ExecutionRequest(operation=opt, configuration=cfg, repeat_runs=repeat))


def test_gate_pass_reports_every_criterion(config, adapter):
    ex = _run(adapter, "mobilenet_v2", "intel_xeon_w2233", "onnxruntime", "default", "int8_quantization")
    gate = QualityGate(config.quality_gate).evaluate(ex)
    assert gate.overall == Status.PASS and gate.reasons == []
    assert [c.name for c in gate.criteria] == ["accuracy", "latency", "memory", "model_size", "artifact", "reproducibility"]
    assert all(c.status == CriterionStatus.PASS for c in gate.criteria)
    text = render_text(gate)
    assert "Accuracy            PASS" in text and "Overall             PASS" in text


def test_gate_fail_shows_reason_and_keeps_other_criteria(config, adapter):
    ex = _run(adapter, "yolov8n", "raspberry_pi_4b", "tflite", "xnnpack", "int8_quantization")  # SCN-001
    gate = QualityGate(config.quality_gate).evaluate(ex)
    assert gate.overall == Status.FAIL
    assert len(gate.reasons) == 1 and gate.reasons[0].startswith("Accuracy:")
    by_name = {c.name: c.status for c in gate.criteria}
    assert by_name["accuracy"] == CriterionStatus.FAIL and by_name["latency"] == CriterionStatus.PASS
    assert "Reasons:" in render_text(gate)


def test_optional_criterion_failure_is_warn_not_fail(adapter):
    ex = _run(adapter, "mobilenet_v2", "intel_xeon_w2233", "onnxruntime", "default", "fp16_conversion")
    ex.metrics.model_size_mb = ex.baseline_metrics.model_size_mb * 2
    gate = QualityGate(QualityGateConfig({"model_size": CriterionConfig("model_size", 10.0, False)})).evaluate(ex)
    assert gate.overall == Status.PASS and gate.criteria[0].status == CriterionStatus.WARN


def test_required_criterion_without_data_fails_gate(adapter):
    ex = _run(adapter, "mobilenet_v2", "intel_xeon_w2233", "onnxruntime", "default", "fp16_conversion")
    ex.metrics.accuracy = None
    gate = QualityGate(QualityGateConfig({"accuracy": CriterionConfig("accuracy", 1.0, True)})).evaluate(ex)
    assert gate.overall == Status.FAIL and "not available" in gate.reasons[0]


def test_gate_not_evaluated_for_non_completed_execution(config, adapter):
    ex = _run(adapter, "pidnet_s", "raspberry_pi_4b", "tflite", "default", "fp16_conversion")  # ERROR
    gate = QualityGate(config.quality_gate).evaluate(ex)
    assert gate.overall == Status.FAIL and gate.criteria == [] and "ERROR" in gate.reasons[0]


def test_unknown_criterion_is_rejected():
    with pytest.raises(ValueError, match="no validator"):
        QualityGate(QualityGateConfig({"vibes": CriterionConfig("vibes", 1.0, True)}))


# ---------------------------------------------------------------- defects ----
def test_no_defect_when_gate_passes(config, adapter):
    ex = _run(adapter, "mobilenet_v2", "intel_xeon_w2233", "onnxruntime", "default", "int8_quantization")
    assert classify(ex, QualityGate(config.quality_gate).evaluate(ex)) is None


def test_accuracy_regression_defect_with_suspected_cause(config, adapter):
    ex = _run(adapter, "yolov8n", "raspberry_pi_4b", "tflite", "xnnpack", "int8_quantization")
    d = classify(ex, QualityGate(config.quality_gate).evaluate(ex))
    assert d.category == DefectCategory.ACCURACY_REGRESSION and d.severity == Severity.HIGH
    assert "suspected" in d.suspected_cause.lower()
    assert d.reproducibility == "deterministic"  # 2 bitwise-identical runs recorded
    assert d.affected_configuration["device"] == "raspberry_pi_4b"
    assert any("threshold=1.0" in e for e in d.evidence)


def test_multiple_gate_failures_yield_primary_and_secondary(config, adapter):
    ex = _run(adapter, "mobilenet_v2", "intel_xeon_w2233", "onnxruntime", "openvino_ep", "int8_quantization")  # artifact
    ex.metrics.latency_ms = ex.baseline_metrics.latency_ms * 2  # + latency regression
    d = classify(ex, QualityGate(config.quality_gate).evaluate(ex))
    assert d.category == DefectCategory.ARTIFACT_ERROR
    assert d.secondary_categories == [DefectCategory.PERFORMANCE_REGRESSION]
    assert d.suspected_cause is None


def test_execution_error_kinds_map_to_categories(adapter):
    conv = _run(adapter, "pidnet_s", "raspberry_pi_4b", "tflite", "default", "fp16_conversion")
    d = classify(conv, None)
    assert d.category == DefectCategory.CONVERSION_FAILURE and d.severity == Severity.HIGH
    assert d.reproducibility == "unknown" and d.suspected_cause is None

    blocked = _run(adapter, "mobilenet_v2", "jetson_orin_nano", "tensorrt", "default", "automatic_compression")
    assert classify(blocked, None).category == DefectCategory.ENVIRONMENT_ERROR

    conv.error = ExecutionError(message="sdk", kind="NotSupportedModelException")
    assert classify(conv, None).category == DefectCategory.MODEL_COMPATIBILITY_ERROR
    conv.error = ExecutionError(message="sdk", kind="NotEnoughCreditException")
    assert classify(conv, None).category == DefectCategory.ENVIRONMENT_ERROR
    conv.error = ExecutionError(message="401", kind="Unauthorized")
    assert classify(conv, None).severity == Severity.CRITICAL


def test_insufficient_evidence_is_unclassified_not_guessed(adapter):
    ex = _run(adapter, "yolov8n", "jetson_orin_nano", "onnxruntime", "default", "automatic_compression")  # SCN-007
    d = classify(ex, None)
    assert d.category == DefectCategory.UNCLASSIFIED and d.suspected_cause is None
    assert "insufficient evidence" in d.message
    ex.error = ExecutionError(message="Something about accuracy went wrong", kind="SomeNewException")
    assert classify(ex, None).category == DefectCategory.UNCLASSIFIED  # free text is never used for inference


def test_unsupported_is_not_a_defect(adapter):
    ex = _run(adapter, "yolov8n", "intel_xeon_w2233", "tflite", "xnnpack", "automatic_compression")
    assert ex.execution_status == ExecutionStatus.UNSUPPORTED and classify(ex, None) is None


def test_distribution_counts_only_present_categories(adapter):
    d = classify(_run(adapter, "pidnet_s", "raspberry_pi_4b", "tflite", "default", "fp16_conversion"), None)
    assert distribution([d, None, d]) == {"CONVERSION_FAILURE": 2}


def test_metrics_helper_types():
    assert Metrics().to_dict()["accuracy"] is None

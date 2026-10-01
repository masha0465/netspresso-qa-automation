import inspect

import pytest

import framework.adapters.mock_adapter as mock_module
from framework.adapters.base import ExecutionRequest
from framework.pipeline.result import (
    Configuration,
    CreditUsageType,
    ExecutionStatus,
    ReproducibilityLevel,
    SupportState,
)

pytestmark = pytest.mark.unit


def _req(model="mobilenet_v2", device="intel_xeon_w2233", runtime="onnxruntime", backend="default",
         optimization="int8_quantization", repeat=2):
    cfg = Configuration(model, device, runtime, backend, optimization)
    return ExecutionRequest(operation=optimization, configuration=cfg, repeat_runs=repeat)


def test_mock_adapter_is_deterministic(adapter):
    a = adapter.run(_req()).to_dict()
    b = adapter.run(_req()).to_dict()
    assert a == b


def test_mock_adapter_uses_no_random_module():
    source = inspect.getsource(mock_module)
    assert "import random" not in source and "numpy.random" not in source


def test_default_run_completes_with_improved_metrics(adapter):
    ex = adapter.run(_req())
    assert ex.execution_status == ExecutionStatus.COMPLETED
    assert ex.metrics.latency_ms < ex.baseline_metrics.latency_ms
    assert ex.metrics.model_size_mb < ex.baseline_metrics.model_size_mb
    assert 0 <= ex.baseline_metrics.accuracy - ex.metrics.accuracy < 0.01
    assert ex.artifact.exists and ex.artifact.checksum_valid is True
    assert ex.artifact.metadata["configuration_key"] == ex.configuration.key
    assert ex.reproducibility.level == ReproducibilityLevel.BITWISE_REPRODUCIBLE
    assert ex.scenario_id is None


def test_credit_is_always_simulated(adapter):
    ex = adapter.run(_req())
    assert ex.credit.usage_type == CreditUsageType.SIMULATED
    assert ex.credit.actual is None
    assert ex.credit.estimated == 50  # client-side SDK constant for quantization, budgeting only


def test_single_run_reports_not_verified_reproducibility(adapter):
    ex = adapter.run(_req(repeat=1))
    assert ex.reproducibility.level == ReproducibilityLevel.NOT_VERIFIED


def test_scenario_accuracy_regression(adapter):
    ex = adapter.run(_req("yolov8n", "raspberry_pi_4b", "tflite", "xnnpack", "int8_quantization"))
    assert ex.scenario_id == "SCN-001"
    assert (ex.baseline_metrics.accuracy - ex.metrics.accuracy) * 100 > 3.0


def test_scenario_error_blocked_unsupported(adapter):
    err = adapter.run(_req("pidnet_s", "raspberry_pi_4b", "tflite", "default", "fp16_conversion"))
    assert err.execution_status == ExecutionStatus.ERROR and err.error.kind == "CONVERSION_FAILURE"
    assert err.metrics is None

    blocked = adapter.run(_req("mobilenet_v2", "jetson_orin_nano", "tensorrt", "default", "automatic_compression"))
    assert blocked.execution_status == ExecutionStatus.BLOCKED and blocked.scenario_id == "SCN-008"

    unsupported = adapter.run(_req("yolov8n", "intel_xeon_w2233", "tflite", "xnnpack", "automatic_compression"))
    assert unsupported.execution_status == ExecutionStatus.UNSUPPORTED


def test_scenario_without_evidence_has_no_error_kind(adapter):
    ex = adapter.run(_req("yolov8n", "jetson_orin_nano", "onnxruntime", "default", "automatic_compression"))
    assert ex.execution_status == ExecutionStatus.ERROR and ex.error.kind is None


def test_scenario_artifact_and_reproducibility_failures(adapter):
    bad_artifact = adapter.run(_req("mobilenet_v2", "intel_xeon_w2233", "onnxruntime", "openvino_ep", "int8_quantization"))
    assert bad_artifact.artifact.checksum_valid is False
    assert bad_artifact.artifact.checksum_sha256 != bad_artifact.artifact.expected_checksum_sha256

    flaky = adapter.run(_req("mobilenet_v2", "jetson_orin_nano", "tensorrt", "default", "int8_quantization"))
    assert flaky.reproducibility.level == ReproducibilityLevel.NOT_REPRODUCIBLE
    assert len(set(flaky.reproducibility.checksums)) == 2


def test_check_support_and_environment(adapter):
    state, reason = adapter.check_support(Configuration("yolov8n", "raspberry_pi_4b", "tensorrt", "default", "fp16_conversion"))
    assert state == SupportState.UNSUPPORTED and "NVIDIA" in reason
    state, _ = adapter.check_support(Configuration("yolov8n", "intel_xeon_w2233", "onnxruntime", "default", "fp16_conversion"))
    assert state == SupportState.UNKNOWN
    env = adapter.describe_environment()
    assert env.adapter == "mock" and env.sdk_version is None and env.extra["network"] is False
    assert adapter.credit_consuming is False

"""Safety-boundary and mapping tests for the real NetsPresso adapter.

None of these tests import or need the netspresso SDK. Real-path mapping is
tested against an injected fake SDK module; the safety tests assert that the
adapter cannot reach the SDK in default / unauthorized modes.
"""

from __future__ import annotations

import enum
import importlib
import json
import sys
import types

import pytest

from framework.adapters.base import ExecutionRequest
from framework.adapters.factory import build_adapter, resolve_settings
from framework.adapters.mock_adapter import MockAdapter
from framework.adapters.netspresso_adapter import (
    CREDIT_BASIS,
    REAL_IMPLEMENTED_OPERATIONS,
    SDK_OPERATIONS,
    ExecutionMode,
    MissingApiKeyError,
    NetsPressoAdapter,
    RealExecutionNotAuthorizedError,
    RealExecutionNotImplementedError,
    SdkUnavailableError,
)
from framework.config import ConfigurationError, ProviderConfig, load_provider
from framework.credit_ledger import CreditLedger
from framework.defects import classify
from framework.pipeline.result import Configuration, CreditUsageType, DefectCategory, ExecutionStatus, Status
from framework.pipeline.runner import CreditSafetyError, PipelineRunner, case_status_for
from framework.quality_gate import QualityGate
from scripts import netspresso_dry_run, run_real_netspresso

pytestmark = pytest.mark.unit

FAKE_KEY = "FAKE-TEST-KEY-0123456789abcdef-NOT-REAL"
CFG = Configuration("mobilenet_v2", "intel_xeon_w2233", "onnxruntime", "default", "int8_quantization")
COMP_CFG = Configuration("graphmodule_sample", "intel_xeon_w2233", "onnxruntime", "default", "automatic_compression")
AUTHORIZED = ProviderConfig(provider="netspresso", mode="real", confirm_credit_use=True)


@pytest.fixture
def forbid_sdk_import(monkeypatch):
    """Fail loudly if anything tries to import the netspresso SDK."""
    real_import = importlib.import_module

    def guard(name, package=None):
        if name == "netspresso" or name.startswith("netspresso."):
            raise AssertionError("netspresso SDK import attempted during a safe-mode test")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", guard)
    yield
    assert not any(m == "netspresso" or m.startswith("netspresso.") for m in sys.modules)


@pytest.fixture
def dry_adapter(config, forbid_sdk_import):
    return NetsPressoAdapter(config, env={})


# ---------------------------------------------------------------- policy -------
def test_default_policy_is_dry_run_and_not_credit_consuming(config):
    assert config.provider == ProviderConfig()  # configs/netspresso.yaml carries the safe defaults
    adapter = NetsPressoAdapter(config, env={})
    assert adapter.mode == ExecutionMode.DRY_RUN
    assert adapter.credit_consuming is False
    assert adapter.api_calls_executed == 0 and adapter.operations_executed == 0


def test_dry_run_needs_no_api_key(dry_adapter):
    assert dry_adapter.api_key_present is False
    plan = dry_adapter.quantize(CFG)
    assert plan.api_call_executed is False and plan.credit_consumed == 0 and plan.actual_credit is None
    ex = dry_adapter.run(ExecutionRequest(operation="int8_quantization", configuration=CFG))
    assert ex.execution_status == ExecutionStatus.NOT_EXECUTED
    assert ex.credit.usage_type == CreditUsageType.NONE and ex.credit.actual is None
    assert ex.credit.estimated == 50 and CREDIT_BASIS in (ex.credit.note or "")


def test_api_key_is_never_exposed(config, forbid_sdk_import):
    adapter = NetsPressoAdapter(config, env={"NETSPRESSO_API_KEY": FAKE_KEY})
    assert adapter.api_key_present is True
    plan = adapter.convert(CFG, token_like_param=FAKE_KEY)  # even a leaked parameter value is redacted
    ex = adapter.run(ExecutionRequest(operation="fp16_conversion", configuration=CFG))
    surfaces = [repr(adapter), plan.render(), json.dumps(plan.to_dict()), json.dumps(ex.to_dict()),
                json.dumps(adapter.describe_environment().to_dict()), "\n".join(ex.logs)]
    for text in surfaces:
        assert FAKE_KEY not in text
    assert "NETSPRESSO_API_KEY present" in plan.render()


def test_real_mode_without_confirmation_is_rejected_before_any_sdk_access(config, forbid_sdk_import):
    settings = ProviderConfig(provider="netspresso", mode="real", confirm_credit_use=False)
    adapter = NetsPressoAdapter(config, settings=settings, env={"NETSPRESSO_API_KEY": FAKE_KEY})
    assert adapter.mode == ExecutionMode.REAL_RUN_UNAUTHORIZED and adapter.credit_consuming is True
    for call in (adapter.validate_connection, lambda: adapter.optimize(CFG), lambda: adapter.quantize(CFG),
                 lambda: adapter.convert(CFG), lambda: adapter.profile(CFG),
                 lambda: adapter.run(ExecutionRequest(operation="automatic_compression", configuration=COMP_CFG))):
        with pytest.raises(RealExecutionNotAuthorizedError, match="No API call was made"):
            call()
    with pytest.raises(RealExecutionNotAuthorizedError):
        adapter._load_sdk()  # noqa: SLF001
    assert adapter.api_calls_executed == 0 and adapter.operations_executed == 0


def test_authorized_but_unimplemented_operations_are_refused_before_sdk_import(config, forbid_sdk_import):
    adapter = NetsPressoAdapter(config, settings=AUTHORIZED, env={"NETSPRESSO_API_KEY": FAKE_KEY})
    assert adapter.mode == ExecutionMode.REAL_RUN_AUTHORIZED and adapter.credit_consuming is True
    assert REAL_IMPLEMENTED_OPERATIONS == ("automatic_compression",)
    with pytest.raises(RealExecutionNotImplementedError):
        adapter.validate_connection()
    with pytest.raises(RealExecutionNotImplementedError):
        adapter.run(ExecutionRequest(operation="int8_quantization", configuration=CFG))
    with pytest.raises(RealExecutionNotImplementedError):
        adapter.profile(CFG)
    assert adapter.api_calls_executed == 0 and adapter.operations_executed == 0


def test_authorized_compression_without_api_key_stops_before_sdk_import(config, forbid_sdk_import):
    adapter = NetsPressoAdapter(config, settings=AUTHORIZED, env={})
    with pytest.raises(MissingApiKeyError, match="NETSPRESSO_API_KEY is not set"):
        adapter.run(ExecutionRequest(operation="automatic_compression", configuration=COMP_CFG))
    assert adapter.operations_executed == 0


def test_authorized_compression_in_python_314_reports_sdk_unavailable(config, monkeypatch, tmp_path):
    model = tmp_path / "m.pt"
    model.write_bytes(b"x" * 10)
    real_import = importlib.import_module
    monkeypatch.setattr(importlib, "import_module",
                        lambda name, package=None: (_ for _ in ()).throw(ImportError(name)) if name.startswith("netspresso") else real_import(name, package))
    adapter = NetsPressoAdapter(config, settings=AUTHORIZED, env={"NETSPRESSO_API_KEY": FAKE_KEY})
    req = ExecutionRequest(operation="automatic_compression", configuration=COMP_CFG,
                           parameters={"input_model_path": str(model), "output_dir": str(tmp_path / "out"), "input_shapes": [{"batch": 1, "channel": 3, "dimension": [224, 224]}]})
    with pytest.raises(SdkUnavailableError, match="Python 3.11"):
        adapter.run(req)
    assert adapter.operations_executed == 0


def test_runner_refuses_real_mode_adapter_without_allow_flag(config):
    with pytest.raises(CreditSafetyError):
        PipelineRunner(config, NetsPressoAdapter(config, settings=AUTHORIZED, env={}), strategy="pairwise")


# ---------------------------------------------------------------- fake SDK real path ----
class _FakeStatus(enum.Enum):
    COMPLETED = "completed"
    ERROR = "error"


def _fake_sdk(monkeypatch, tmp_path, *, outcome="completed", credit_before=500, credit_after=475, exit_on_init=False):
    """Install a fake `netspresso` + `netspresso.enums` module pair via importlib.import_module."""
    calls = {"init": 0, "compress": 0, "get_user": 0}

    class Framework(enum.Enum):
        PYTORCH = "pytorch"
        ONNX = "onnx"

    def user(total):
        return types.SimpleNamespace(credit_info=types.SimpleNamespace(total=total))

    class FakeCompressor:
        def automatic_compression(self, *, input_model_path, output_dir, input_shapes, framework, compression_ratio):
            calls["compress"] += 1
            assert framework is Framework.PYTORCH and compression_ratio == 0.5 and input_shapes[0]["batch"] == 1
            out = tmp_path / "out_1"
            out.mkdir()
            if outcome == "error":
                detail = types.SimpleNamespace(name="NotSupportedModelException", error_code="E42", message="unsupported op (fake)",
                                               data=types.SimpleNamespace(error_log="trace..."))
                return types.SimpleNamespace(status=_FakeStatus.ERROR, error_detail=detail)
            compressed = out / "compressed.pt"
            compressed.write_bytes(b"c" * 4000)
            (out / "metadata.json").write_text("{}", encoding="utf-8")
            return types.SimpleNamespace(
                status=_FakeStatus.COMPLETED,
                compressed_model_path=compressed.as_posix(),
                compressed_onnx_model_path=(out / "compressed.onnx").as_posix(),
                compression_info=types.SimpleNamespace(method="PR_L2", ratio=compression_ratio),
                results=types.SimpleNamespace(
                    original_model=types.SimpleNamespace(size=10.0, flops=200.0, number_of_parameters=1000, model_id="orig"),
                    compressed_model=types.SimpleNamespace(size=4.0, flops=90.0, number_of_parameters=450, model_id="cmp-123"),
                ),
            )

    class FakeNetsPresso:
        def __init__(self, *, api_key, dev_mode=False):
            calls["init"] += 1
            assert api_key == FAKE_KEY and dev_mode is False
            if exit_on_init:
                raise SystemExit(1)
            self.user_info = user(credit_before)

        def compressor_v2(self):
            return FakeCompressor()

        def get_user(self):
            calls["get_user"] += 1
            return user(credit_after)

    sdk = types.SimpleNamespace(__version__="1.17.0-fake", NetsPresso=FakeNetsPresso)
    enums = types.SimpleNamespace(Framework=Framework)
    real_import = importlib.import_module

    def fake_import(name, package=None):
        if name == "netspresso":
            return sdk
        if name == "netspresso.enums":
            return enums
        if name == "loguru":
            raise ImportError("no loguru in test")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    return calls


def _comp_request(tmp_path):
    model = tmp_path / "graphmodule.pt"
    model.write_bytes(b"m" * 10_000)
    return ExecutionRequest(
        operation="automatic_compression",
        configuration=COMP_CFG,
        parameters={"input_model_path": model.as_posix(), "output_dir": (tmp_path / "out").as_posix(),
                    "input_shapes": [{"batch": 1, "channel": 3, "dimension": [224, 224]}], "framework": "pytorch",
                    "compression_ratio": 0.5, "sdk_log_path": (tmp_path / "sdk.log").as_posix()},
    )


def test_real_path_maps_completed_metadata_and_observed_credit(config, monkeypatch, tmp_path):
    calls = _fake_sdk(monkeypatch, tmp_path)
    adapter = NetsPressoAdapter(config, settings=AUTHORIZED, env={"NETSPRESSO_API_KEY": FAKE_KEY})
    ex = adapter.run(_comp_request(tmp_path))
    assert calls == {"init": 1, "compress": 1, "get_user": 1}  # exactly one service operation
    assert adapter.operations_executed == 1 and adapter.api_calls_executed == 1
    assert ex.execution_status == ExecutionStatus.COMPLETED
    assert ex.environment.sdk_version == "1.17.0-fake" and ex.environment.extra["sdk_loaded"] is True
    assert ex.credit.usage_type == CreditUsageType.ACTUAL and ex.credit.actual == 25 and ex.credit.estimated == 25
    assert "before (500) - after (475)" in ex.credit.note and CREDIT_BASIS in ex.credit.note
    assert ex.metrics.model_size_mb == 0.004 and ex.baseline_metrics.model_size_mb == 0.01
    assert ex.metrics.accuracy is None and ex.metrics.latency_ms is None  # never fabricated
    assert ex.metrics.extra["sdk_compressed_flops"] == 90.0 and ex.baseline_metrics.extra["sdk_original_number_of_parameters"] == 1000.0
    assert ex.artifact.exists and ex.artifact.size_bytes == 4000 and ex.artifact.checksum_valid is None
    assert ex.artifact.metadata["sdk_model_id"] == "cmp-123" and ex.artifact.metadata["compression_method"] == "PR_L2"
    assert ex.reproducibility.level.value == "NOT_VERIFIED" and ex.reproducibility.runs == 1
    assert FAKE_KEY not in json.dumps(ex.to_dict())

    gate = QualityGate(config.quality_gate).evaluate(ex)
    assert not gate.passed  # accuracy/latency/memory missing, artifact checksum unverifiable, single run
    defect = classify(ex, gate)
    assert defect.category == DefectCategory.UNCLASSIFIED and "missing data" in defect.message
    assert case_status_for(ex, gate) == Status.FAIL


def test_real_path_maps_sdk_error_status_without_retry(config, monkeypatch, tmp_path):
    calls = _fake_sdk(monkeypatch, tmp_path, outcome="error", credit_after=500)
    adapter = NetsPressoAdapter(config, settings=AUTHORIZED, env={"NETSPRESSO_API_KEY": FAKE_KEY})
    ex = adapter.run(_comp_request(tmp_path))
    assert calls["compress"] == 1
    assert ex.execution_status == ExecutionStatus.ERROR
    assert ex.error.kind == "NotSupportedModelException" and ex.error.details["error_log"] == "trace..."
    assert ex.credit.usage_type == CreditUsageType.ACTUAL and ex.credit.actual == 0  # observed: nothing deducted
    assert classify(ex, None).category == DefectCategory.MODEL_COMPATIBILITY_ERROR
    assert case_status_for(ex, None) == Status.FAIL


def test_real_path_version_gate_exit_is_blocked_without_operation(config, monkeypatch, tmp_path):
    calls = _fake_sdk(monkeypatch, tmp_path, exit_on_init=True)
    adapter = NetsPressoAdapter(config, settings=AUTHORIZED, env={"NETSPRESSO_API_KEY": FAKE_KEY})
    ex = adapter.run(_comp_request(tmp_path))
    assert calls["compress"] == 0 and adapter.operations_executed == 0
    assert ex.execution_status == ExecutionStatus.BLOCKED and ex.error.kind == "SdkVersionGateExit"
    assert ex.credit.usage_type == CreditUsageType.NONE
    assert case_status_for(ex, None) == Status.BLOCKED


# ---------------------------------------------------------------- plan ---------
def test_dry_run_plan_is_structured_and_uses_verified_sdk_names(dry_adapter, root):
    plan = dry_adapter.plan("int8_quantization", CFG)
    d = plan.to_dict()
    assert d["provider"] == "NetsPresso" and d["execution_mode"] == "DRY_RUN"
    assert d["sdk_service"] == "quantizer" and d["sdk_method"] == "automatic_quantization"
    assert d["sdk_device_name"] == "Intel-Xeon" and d["sdk_framework"] == "onnx"
    assert d["estimated_credit"] == 50 and "CLIENT_SIDE_PRE_CHECK_CONSTANT" in d["credit_basis"]
    assert d["actual_credit"] is None and d["credit_consumed"] == 0 and d["sdk_loaded"] is False
    text = plan.render()
    for token in ("Provider:", "Execution:         DRY_RUN", "Operation:         int8_quantization", "Model:             mobilenet_v2",
                  "Estimated Credit:  50", "Actual Credit:     NOT MEASURED", "API call:          NOT EXECUTED", "Credit consumed:   0"):
        assert token in text, token

    surface = json.loads((root / "docs" / "research_cache" / "sdk_surface.json").read_text(encoding="utf-8"))
    facade = surface["classes"]["NetsPresso"]["public_methods"]
    class_for = {"compressor_v2": "CompressorV2", "converter_v2": "ConverterV2", "quantizer": "Quantizer", "profiler": "Profiler"}
    for op in SDK_OPERATIONS.values():
        if op.service_factory == "NetsPresso":
            continue
        assert op.service_factory in facade, op.service_factory
        if op.service_factory in class_for:
            assert op.method in surface["classes"][class_for[op.service_factory]]["public_methods"], op.method
        assert op.service_task in surface["enums"]["ServiceTask"], op.service_task


def test_operation_table_matches_optimization_config_constants(config):
    for opt in config.optimizations:
        assert SDK_OPERATIONS[opt.name].client_credit_constant == opt.sdk_credit_constant
    assert SDK_OPERATIONS["validate_connection"].client_credit_constant is None
    assert SDK_OPERATIONS["graph_optimize"].client_precheck is False


def test_unknown_operation_rejected(dry_adapter):
    with pytest.raises(ValueError, match="unknown operation"):
        dry_adapter.plan("teleport")


def test_plan_marks_unsupported_combinations(dry_adapter):
    unsupported = Configuration("yolov8n", "raspberry_pi_4b", "tensorrt", "default", "fp16_conversion")
    plan = dry_adapter.plan("fp16_conversion", unsupported)
    assert any("UNSUPPORTED" in p for p in plan.preconditions)


# ---------------------------------------------------------------- pipeline / ledger ----
def test_dry_run_pipeline_executes_nothing_and_leaves_ledger_untouched(config, forbid_sdk_import, tmp_path):
    ledger = CreditLedger(tmp_path / "ledger.json")
    report = PipelineRunner(config, NetsPressoAdapter(config, env={}), strategy="pairwise", ledger=ledger).run()
    counts = report.status_counts()
    assert counts["PASS"] == 0 and counts["FAIL"] == 0 and counts["BLOCKED"] == 0
    planned = [c for c in report.cases if c.execution is not None]
    assert planned and all(c.status == Status.NOT_TESTED for c in planned)
    assert all(c.execution.execution_status == ExecutionStatus.NOT_EXECUTED for c in planned)
    assert ledger.operations == [] and ledger.used_credit == 0 and ledger.simulated_total == 0
    assert report.environment.adapter == "netspresso" and report.environment.extra["api_calls_executed"] == 0


def test_ledger_budget_helper_and_details(tmp_path):
    ledger = CreditLedger(tmp_path / "ledger.json")
    ok = ledger.budget_check([25, 25], reserve=100)
    assert ok["affordable"] and ok["remaining_after"] == 450 and ok["estimated_total"] == 50
    assert not ledger.budget_check([50] * 9, reserve=100)["affordable"]
    assert ledger.operations == []  # budget_check never records
    entry = ledger.record(operation="automatic_compression", model="m", purpose="p", usage_type="actual", estimated_credit=25,
                          actual_credit=25, confirmation=True, adapter="netspresso", details={"credit_before": 500, "credit_after": 475})
    assert entry.details["credit_after"] == 475 and ledger.operations[0]["details"]["credit_before"] == 500


def test_mock_adapter_unchanged_by_factory_defaults(config):
    adapter = build_adapter(config)
    assert isinstance(adapter, MockAdapter) and adapter.credit_consuming is False
    real = build_adapter(config, provider="netspresso", env={})
    assert isinstance(real, NetsPressoAdapter) and real.mode == ExecutionMode.DRY_RUN
    with pytest.raises(ConfigurationError):
        resolve_settings(config, provider="cloud")
    with pytest.raises(ConfigurationError):
        resolve_settings(config, mode="yolo")


def test_provider_config_rejects_credentials_in_yaml(tmp_path):
    path = tmp_path / "netspresso.yaml"
    path.write_text("provider:\n  type: netspresso\nnetspresso:\n  api_key: not-allowed\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="credential"):
        load_provider(path)
    path.write_text("execution:\n  mode: sometimes\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="execution.mode"):
        load_provider(path)
    assert load_provider(tmp_path / "missing.yaml") == ProviderConfig()


# ---------------------------------------------------------------- CLI ----------
def test_cli_dry_run_single_and_matrix(forbid_sdk_import, capsys, tmp_path):
    assert netspresso_dry_run.main(["--operation", "profile", "--device", "jetson_orin_nano", "--runtime", "tensorrt",
                                    "--json", str(tmp_path / "plan.json")]) == 0
    out = capsys.readouterr().out
    for token in ("Provider:          NetsPresso", "Execution:         DRY_RUN", "Operation:         profile",
                  "Estimated Credit:  25", "Actual Credit:     NOT MEASURED", "API call:          NOT EXECUTED",
                  "Credit consumed:   0", "SDK calls made:    0"):
        assert token in out, token
    assert json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))["plan"]["sdk_method"] == "profile_model"

    assert netspresso_dry_run.main(["--matrix", "pairwise"]) == 0
    out = capsys.readouterr().out
    assert "Planned operations:" in out and ("EXCEEDS BUDGET" in out or "-> OK" in out)
    assert "Credit consumed:   0" in out


def test_cli_real_mode_paths_never_call_sdk(forbid_sdk_import, capsys):
    assert netspresso_dry_run.main(["--mode", "real"]) == 2
    assert "REJECTED" in capsys.readouterr().out
    assert netspresso_dry_run.main(["--mode", "real", "--confirm-credit-use"]) == 3
    assert "REFUSED" in capsys.readouterr().out


def test_real_run_script_aborts_safely(forbid_sdk_import, capsys, monkeypatch, tmp_path):
    monkeypatch.delenv("NETSPRESSO_API_KEY", raising=False)
    monkeypatch.setattr(run_real_netspresso, "ENV_FILE", tmp_path / "no.env")
    assert run_real_netspresso.main([]) == 2  # no confirmation flag
    assert "ABORT" in capsys.readouterr().out
    assert run_real_netspresso.main(["--confirm-credit-use"]) == 4  # no API key
    assert "NETSPRESSO_API_KEY is not set" in capsys.readouterr().out
    assert run_real_netspresso.parse_shape("1,3,224,224") == [{"batch": 1, "channel": 3, "dimension": [224, 224]}]

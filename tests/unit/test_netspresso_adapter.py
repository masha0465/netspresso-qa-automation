"""Safety-boundary tests for the real NetsPresso adapter.

None of these tests import or need the netspresso SDK. They assert that the
adapter cannot reach it in default / unauthorized modes.
"""

from __future__ import annotations

import importlib
import json
import sys

import pytest

from framework.adapters.base import ExecutionRequest
from framework.adapters.factory import build_adapter, resolve_settings
from framework.adapters.mock_adapter import MockAdapter
from framework.adapters.netspresso_adapter import (
    CREDIT_BASIS,
    SDK_OPERATIONS,
    ExecutionMode,
    NetsPressoAdapter,
    RealExecutionNotAuthorizedError,
    RealExecutionNotImplementedError,
)
from framework.config import ConfigurationError, ProviderConfig, load_provider
from framework.credit_ledger import CreditLedger
from framework.pipeline.result import Configuration, CreditUsageType, ExecutionStatus, Status
from framework.pipeline.runner import CreditSafetyError, PipelineRunner
from scripts import netspresso_dry_run

pytestmark = pytest.mark.unit

FAKE_KEY = "FAKE-TEST-KEY-0123456789abcdef-NOT-REAL"
CFG = Configuration("mobilenet_v2", "intel_xeon_w2233", "onnxruntime", "default", "int8_quantization")


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
    assert adapter.api_calls_executed == 0


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
    plan = adapter.convert(CFG)
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
                 lambda: adapter.run(ExecutionRequest(operation="profile", configuration=CFG))):
        with pytest.raises(RealExecutionNotAuthorizedError, match="No API call was made"):
            call()
    with pytest.raises(RealExecutionNotAuthorizedError):
        adapter._load_sdk()  # noqa: SLF001
    assert adapter.api_calls_executed == 0


def test_authorized_real_run_is_explicitly_not_implemented_in_phase_5a(config, forbid_sdk_import):
    settings = ProviderConfig(provider="netspresso", mode="real", confirm_credit_use=True)
    adapter = NetsPressoAdapter(config, settings=settings, env={"NETSPRESSO_API_KEY": FAKE_KEY})
    assert adapter.mode == ExecutionMode.REAL_RUN_AUTHORIZED and adapter.credit_consuming is True
    with pytest.raises(RealExecutionNotImplementedError, match="Phase 5-A"):
        adapter.validate_connection()
    with pytest.raises(RealExecutionNotImplementedError):
        adapter.run(ExecutionRequest(operation="int8_quantization", configuration=CFG))
    assert adapter.api_calls_executed == 0


def test_runner_refuses_real_mode_adapter_without_allow_flag(config):
    settings = ProviderConfig(provider="netspresso", mode="real", confirm_credit_use=True)
    with pytest.raises(CreditSafetyError):
        PipelineRunner(config, NetsPressoAdapter(config, settings=settings, env={}), strategy="pairwise")


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

    # every SDK name in the operation table exists in the Phase 1 introspection record
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


# ---------------------------------------------------------------- pipeline -----
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


def test_repository_ledger_unchanged_and_budget_helper(root):
    ledger = CreditLedger(root / "reports" / "credit_usage.json")
    assert ledger.used_credit == 0 and ledger.remaining_estimate == 500 and ledger.operations == []
    ok = ledger.budget_check([25, 25], reserve=100)
    assert ok["affordable"] and ok["remaining_after"] == 450 and ok["estimated_total"] == 50
    assert not ledger.budget_check([50] * 9, reserve=100)["affordable"]
    assert ledger.operations == []  # budget_check never records


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
    assert "NOT IMPLEMENTED" in capsys.readouterr().out

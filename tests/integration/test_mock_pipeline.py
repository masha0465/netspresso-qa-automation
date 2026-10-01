import pytest

from framework.adapters.base import BaseAdapter
from framework.adapters.mock_adapter import MockAdapter
from framework.credit_ledger import CreditLedger
from framework.pipeline.result import DefectCategory, Status, SupportState
from framework.pipeline.runner import STRATEGIES, CreditSafetyError, PipelineRunner
from tests.conftest import fixed_clock

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", params=STRATEGIES)
def report(request):
    from framework.config import load_config
    from tests.conftest import CONFIG_DIR

    cfg = load_config(CONFIG_DIR)
    return PipelineRunner(cfg, MockAdapter(cfg, clock=fixed_clock), strategy=request.param, clock=fixed_clock).run()


def test_status_accounting_is_consistent(report):
    s = report.summary()
    counts = report.status_counts()
    assert sum(counts.values()) == s["total_combinations"] == 324
    assert s["tested_combinations"] == counts["PASS"] + counts["FAIL"] + counts["BLOCKED"]
    assert s["tested_combinations"] + counts["NOT_TESTED"] + counts["UNSUPPORTED"] == 324
    assert s["selected_combinations"] >= s["tested_combinations"]


def test_unsupported_and_not_tested_are_never_executed(report):
    for c in report.cases:
        if c.status == Status.NOT_TESTED:
            assert c.execution is None and c.gate is None and c.defect is None
        if c.status == Status.UNSUPPORTED and c.support_reason and "structural" in c.support_reason:
            assert c.execution is None
        if c.selected is False:
            assert c.execution is None


def test_verdict_rules(report):
    for c in report.cases:
        if c.status == Status.PASS:
            assert c.gate is not None and c.gate.passed and c.defect is None and c.support == SupportState.SUPPORTED
        if c.status == Status.FAIL:
            assert c.defect is not None
            if c.gate and c.gate.criteria:
                assert not c.gate.passed and c.gate.reasons
        if c.status == Status.BLOCKED:
            assert c.defect is not None and c.defect.category == DefectCategory.ENVIRONMENT_ERROR


def test_run_is_deterministic(report):
    from framework.config import load_config
    from tests.conftest import CONFIG_DIR

    cfg = load_config(CONFIG_DIR)
    again = PipelineRunner(cfg, MockAdapter(cfg, clock=fixed_clock), strategy=report.strategy, clock=fixed_clock).run()
    assert again.to_dict() == report.to_dict()


def test_full_run_contains_expected_examples(config):
    report = PipelineRunner(config, MockAdapter(config, clock=fixed_clock), strategy="full", clock=fixed_clock).run()
    by_key = {c.configuration.key: c for c in report.cases}
    acc = by_key["yolov8n|raspberry_pi_4b|tflite|xnnpack|int8_quantization"]
    assert acc.status == Status.FAIL and acc.defect.category == DefectCategory.ACCURACY_REGRESSION
    conv = by_key["pidnet_s|raspberry_pi_4b|tflite|default|fp16_conversion"]
    assert conv.status == Status.FAIL and conv.defect.category == DefectCategory.CONVERSION_FAILURE
    blocked = by_key["mobilenet_v2|jetson_orin_nano|tensorrt|default|automatic_compression"]
    assert blocked.status == Status.BLOCKED
    runtime_unsupported = by_key["yolov8n|intel_xeon_w2233|tflite|xnnpack|automatic_compression"]
    assert runtime_unsupported.status == Status.UNSUPPORTED and runtime_unsupported.execution is not None
    unclassified = by_key["yolov8n|jetson_orin_nano|onnxruntime|default|automatic_compression"]
    assert unclassified.defect.category == DefectCategory.UNCLASSIFIED
    passing = by_key["mobilenet_v2|intel_xeon_w2233|onnxruntime|default|int8_quantization"]
    assert passing.status == Status.PASS
    assert report.status_counts()["NOT_TESTED"] == 0  # full strategy leaves nothing untested


def test_pairwise_run_statistics(config):
    report = PipelineRunner(config, MockAdapter(config, clock=fixed_clock), strategy="pairwise", clock=fixed_clock).run()
    sel = report.selection
    assert sel["pair_coverage_percent"] == 100.0
    assert sel["selected"] < sel["candidates_feasible"] and sel["reduction_percent"] > 50
    assert report.status_counts()["NOT_TESTED"] > 0  # honest: unselected cells stay NOT_TESTED


def test_ledger_receives_only_simulated_entries(config, tmp_path):
    ledger = CreditLedger(tmp_path / "ledger.json")
    PipelineRunner(config, MockAdapter(config, clock=fixed_clock), strategy="pairwise", ledger=ledger, clock=fixed_clock).run()
    assert ledger.operations and all(op["usage_type"] == "simulated" for op in ledger.operations)
    assert ledger.used_credit == 0 and ledger.remaining_estimate == 500 and ledger.simulated_total > 0
    assert (tmp_path / "ledger.json").exists()


class _CreditHungryAdapter(BaseAdapter):
    name = "fake-real"
    credit_consuming = True

    def describe_environment(self):
        raise AssertionError("must not be called")

    def check_support(self, configuration):
        raise AssertionError("must not be called")

    def run(self, request):
        raise AssertionError("must not be called")


def test_credit_consuming_adapter_is_refused_without_confirmation(config):
    with pytest.raises(CreditSafetyError):
        PipelineRunner(config, _CreditHungryAdapter(), strategy="pairwise")


def test_unknown_strategy_rejected(config, adapter):
    with pytest.raises(ValueError):
        PipelineRunner(config, adapter, strategy="everything")

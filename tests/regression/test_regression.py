import pytest

from framework.adapters.mock_adapter import MockAdapter
from framework.pipeline.result import Status
from framework.pipeline.runner import PipelineRunner
from framework.regression import RegressionComparator, RegressionThresholds
from framework.reporter import write_regression_html, write_regression_json
from tests.conftest import fixed_clock

pytestmark = pytest.mark.regression


@pytest.fixture(scope="module")
def runs(config, regressed_config):
    baseline = PipelineRunner(config, MockAdapter(config, clock=fixed_clock), strategy="full", clock=fixed_clock).run()
    current = PipelineRunner(regressed_config, MockAdapter(regressed_config, clock=fixed_clock), strategy="full", clock=fixed_clock).run()
    return baseline, current


def test_thresholds_come_from_quality_gate(config):
    t = RegressionThresholds.from_quality_gate(config.quality_gate)
    assert t.accuracy_max_drop_pp == 1.0 and t.latency_max_increase_percent == 10.0 and t.memory_max_increase_percent == 15.0


def test_self_comparison_is_pass(config, runs):
    baseline, _ = runs
    rep = RegressionComparator(RegressionThresholds.from_quality_gate(config.quality_gate)).compare(baseline, baseline)
    assert rep.overall == "PASS" and rep.summary()["REGRESSION"] == 0
    assert rep.not_executed_in_both == baseline.status_counts()["UNSUPPORTED"] - sum(
        1 for c in baseline.cases if c.status == Status.UNSUPPORTED and c.execution is not None
    )


def test_regressed_run_is_detected_with_reasons(config, runs, tmp_path):
    baseline, current = runs
    rep = RegressionComparator(RegressionThresholds.from_quality_gate(config.quality_gate)).compare(baseline, current)
    assert rep.overall == "REGRESSION"
    regs = {c.configuration["model"] + "|" + c.configuration["device"] + "|" + c.configuration["runtime"] + "|"
            + c.configuration["backend"] + "|" + c.configuration["optimization"]: c for c in rep.regressions()}

    lat = regs["mobilenet_v2|intel_xeon_w2233|onnxruntime|default|int8_quantization"]
    assert any(r.startswith("latency_ms") for r in lat.reasons) and "status degraded PASS -> FAIL" in lat.reasons
    acc = regs["yolov8n|intel_xeon_w2233|onnxruntime|default|fp16_conversion"]
    assert any(r.startswith("accuracy") for r in acc.reasons)
    repro = regs["mobilenet_v2|raspberry_pi_4b|tflite|default|fp16_conversion"]
    assert any("reproducibility degraded" in r for r in repro.reasons)
    assert repro.reproducibility_change == ["BITWISE_REPRODUCIBLE", "NOT_REPRODUCIBLE"]

    # Known-failing case unchanged -> not a new regression; removed SCN-002 -> improvement, not regression
    assert "yolov8n|raspberry_pi_4b|tflite|xnnpack|int8_quantization" not in regs
    improved = next(c for c in rep.comparisons if c.configuration["model"] == "pidnet_s" and c.configuration["device"] == "jetson_orin_nano"
                    and c.configuration["runtime"] == "tensorrt" and c.configuration["optimization"] == "fp16_conversion")
    assert improved.result == "PASS" and improved.baseline_status == "FAIL" and improved.current_status == "PASS"
    assert len(rep.regressions()) == 3

    j = write_regression_json(rep, tmp_path / "reg.json")
    h = write_regression_html(rep, tmp_path / "reg.html")
    assert j.exists() and "REGRESSION" in h.read_text(encoding="utf-8")


def test_new_and_missing_cases(config, runs):
    baseline, current = runs
    truncated = PipelineRunner(config, MockAdapter(config, clock=fixed_clock), strategy="full", clock=fixed_clock).run()
    removed = truncated.cases.pop(0)
    rep = RegressionComparator(RegressionThresholds()).compare(truncated, baseline)
    new = [c for c in rep.comparisons if c.result == "NEW"]
    assert len(new) == 1 and new[0].case_id == removed.case_id
    rep2 = RegressionComparator(RegressionThresholds()).compare(baseline, truncated)
    assert [c.case_id for c in rep2.comparisons if c.result == "MISSING"] == [removed.case_id]


def test_comparison_is_deterministic_and_sorted(config, runs):
    baseline, current = runs
    comparator = RegressionComparator(RegressionThresholds.from_quality_gate(config.quality_gate))
    a = comparator.compare(baseline, current).to_dict()
    b = comparator.compare(baseline, current).to_dict()
    assert a == b
    ids = [c["case_id"] for c in a["comparisons"]]
    assert ids == sorted(ids)

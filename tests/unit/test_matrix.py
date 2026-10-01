import math

import pytest

from framework.matrix.generator import MatrixGenerator, determine_support, select_all_feasible
from framework.matrix.pairwise import PairwiseSelector, pairs_of
from framework.matrix.risk import RiskScorer, risk_distribution, select_by_risk
from framework.pipeline.result import Configuration, RiskLevel, SupportState

pytestmark = pytest.mark.unit


def test_full_product_size_and_order(config):
    gen = MatrixGenerator(config)
    cases = gen.full_product()
    expected = math.prod(len(v) for v in gen.dimension_values.values())
    assert len(cases) == expected == 324
    assert cases[0].configuration == Configuration("yolov8n", "intel_xeon_w2233", "onnxruntime", "default", "fp16_conversion")
    assert [c.case_id for c in cases] == [c.case_id for c in gen.full_product()]  # deterministic
    assert len({c.case_id for c in cases}) == len(cases)


def test_support_states(config):
    structural = determine_support(config, Configuration("yolov8n", "intel_xeon_w2233", "onnxruntime", "xnnpack", "fp16_conversion"))
    assert structural[0] == SupportState.UNSUPPORTED and "structural" in structural[1]
    documented = determine_support(config, Configuration("yolov8n", "raspberry_pi_4b", "openvino", "default", "fp16_conversion"))
    assert documented[0] == SupportState.UNSUPPORTED and "Intel" in documented[1]
    unknown = determine_support(config, Configuration("yolov8n", "jetson_orin_nano", "tensorrt", "default", "fp16_conversion"))
    assert unknown == (SupportState.UNKNOWN, None)  # never assumed SUPPORTED


def test_stats_and_full_selection(config):
    gen = MatrixGenerator(config)
    cases = gen.full_product()
    selected = select_all_feasible(cases)
    stats = gen.stats(cases)
    assert stats.total_combinations == 324
    assert stats.unsupported_combinations + stats.feasible_combinations == 324
    assert stats.selected_combinations == len(selected) == stats.feasible_combinations
    assert stats.reduction_percent == 0.0
    assert all(c.support != SupportState.UNSUPPORTED for c in selected)


def test_pairwise_covers_every_feasible_pair_deterministically(config):
    gen = MatrixGenerator(config)
    result = PairwiseSelector().select(gen.full_product())
    assert result.coverage_percent == 100.0
    assert result.covered_pairs == result.feasible_pairs
    assert 0 < len(result.selected) < result.total_candidates
    assert result.reduction_percent > 50
    assert all(c.feasible for c in result.selected)

    covered = set().union(*(pairs_of(c) for c in result.selected))
    feasible = set().union(*(pairs_of(c) for c in gen.full_product() if c.feasible))
    assert feasible <= covered

    again = PairwiseSelector().select(gen.full_product())
    assert [c.case_id for c in again.selected] == [c.case_id for c in result.selected]


def test_pairwise_reports_infeasible_pairs_separately(config):
    result = PairwiseSelector().select(MatrixGenerator(config).full_product())
    assert result.total_pairs_in_full_matrix == result.feasible_pairs + len(result.infeasible_pairs)
    rendered = result.to_dict()["infeasible_pairs"]
    assert any("backend=xnnpack" in p and "runtime=onnxruntime" in p for p in rendered)


def test_pairwise_small_hand_matrix():
    from framework.matrix.generator import TestCase

    combos = [
        TestCase(Configuration(m, d, r, "default", "fp16_conversion"), SupportState.UNKNOWN)
        for m in ("a", "b") for d in ("x", "y") for r in ("p", "q")
    ]
    res = PairwiseSelector(dimensions=("model", "device", "runtime")).select(combos)
    assert res.coverage_percent == 100.0
    assert 4 <= len(res.selected) < 8


def test_risk_scoring_explains_factors(config):
    scorer = RiskScorer(config)
    high = scorer.assess(Configuration("pidnet_s", "jetson_orin_nano", "tensorrt", "default", "automatic_compression"))
    assert high.level == RiskLevel.HIGH
    assert "model.new_model (+3)" in high.factors and "runtime.previous_regression (+3)" in high.factors
    low = scorer.assess(Configuration("mobilenet_v2", "intel_xeon_w2233", "onnxruntime", "default", "fp16_conversion"))
    assert low.level == RiskLevel.LOW and low.score == 2
    lossy = scorer.assess(Configuration("mobilenet_v2", "intel_xeon_w2233", "onnxruntime", "default", "int8_quantization"))
    assert lossy.score == low.score + config.risk.weights["lossy_precision"]


def test_risk_selection_only_includes_configured_levels(config):
    cases = MatrixGenerator(config).full_product()
    RiskScorer(config).annotate(cases)
    selected = select_by_risk(cases, (RiskLevel.HIGH,))
    assert selected and all(c.risk.level == RiskLevel.HIGH and c.feasible for c in selected)
    assert all(c.selection_reason.startswith("risk HIGH") for c in selected)
    dist = risk_distribution(cases)
    assert sum(dist.values()) == len(cases)

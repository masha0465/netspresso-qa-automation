"""Phase 5-E R1 tests (TC-Y8-R1-001 .. 013): upstream traceable fx export + functional equivalence.

Pure-Python checks of ``framework.evaluation.fx_traceability`` plus assertions over the R1 records written by
``scripts/yolov8_r1_traceable_export.py`` / ``scripts/yolov8_r1_fork_crosscheck.py`` (0 credits, no torch here).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from framework.evaluation.detection_head import (
    expected_body_output_shapes,
    expected_decoded_output_shape,
    parse_head_meta,
)
from framework.evaluation.fx_traceability import (
    DIVERGENCE_CATEGORIES,
    MANIFEST_REQUIRED_KEYS,
    PATCH_ID,
    TRACEABILITY_BOUNDARY,
    build_export_manifest,
    classify_divergence,
    compare_detections,
    equivalence_stats,
    equivalence_verdict,
    first_divergence,
    summarize_graph,
    validate_manifest,
)

pytestmark = pytest.mark.unit

HEX64 = re.compile(r"[0-9a-f]{64}")
R1_SCRIPTS = ("yolov8_r1_traceable_export.py", "yolov8_r1_fork_crosscheck.py")


def _latest(root: Path, suffix: str, filename: str) -> dict:
    runs = sorted(p for p in (root / "reports" / "yolov8_baseline").glob(f"2*Z_{suffix}") if (p / filename).is_file())
    assert runs, f"R1 record '{suffix}/{filename}' expected (run the R1 scripts first)"
    return json.loads((runs[-1] / filename).read_text(encoding="utf-8"))


@pytest.fixture
def r1(root):
    return _latest(root, "r1", "r1_result.json")


@pytest.fixture
def torch201(root):
    return _latest(root, "r1_torch201_load", "r1_torch201_load.json")


@pytest.fixture
def crosscheck(root):
    return _latest(root, "r1_fork_crosscheck", "r1_fork_crosscheck.json")


# ---------------------------------------------------------------- TC-Y8-R1-001 upstream baseline reproducibility ----
def test_r1_001_upstream_baseline_reproduced(r1):
    rep = r1["baseline_reproduction"]
    assert rep["current"]["status"] == "MEASURED" and rep["previous_record"] is not None
    assert abs(rep["current"]["metrics"]["mAP50-95"] - rep["previous_record"]["map50_95"]) < 5e-5 and rep["reproduced"] is True
    assert "ultralytics 8.4" in r1["evaluator"]


# ---------------------------------------------------------------- TC-Y8-R1-002 traceability wrapper / boundary ----
def test_r1_002_traceability_boundary_and_graph_summary():
    assert TRACEABILITY_BOUNDARY["patch_id"] == PATCH_ID and len(TRACEABILITY_BOUNDARY["patch_steps"]) == 2
    assert any("NMS" in s for s in TRACEABILITY_BOUNDARY["out_of_graph"]) and any("DFL" in s for s in TRACEABILITY_BOUNDARY["out_of_graph"])
    clean = summarize_graph([{"op": "placeholder", "target": "x"}, {"op": "call_module", "target": "layers.0.conv", "module_class": "torch.nn.modules.conv.Conv2d"},
                             {"op": "call_function", "target": "cat"}, {"op": "call_method", "target": "split"}, {"op": "output", "target": "output"}])
    assert clean["leaf_clean"] and clean["ops"] == {"call_function": 1, "call_method": 1, "call_module": 1, "output": 1, "placeholder": 1}
    dirty_leaf = summarize_graph([{"op": "call_module", "target": "layers.2", "module_class": "ultralytics.nn.modules.block.C2f"}, {"op": "output", "target": "output"}])
    assert not dirty_leaf["leaf_clean"] and dirty_leaf["non_torch_nn_leaves"] == ["layers.2:ultralytics.nn.modules.block.C2f"]
    decode_in_graph = summarize_graph([{"op": "call_method", "target": "sigmoid"}, {"op": "call_function", "target": "dist2bbox"}, {"op": "output", "target": "output"}])
    assert not decode_in_graph["leaf_clean"] and decode_in_graph["out_of_graph_markers_found_in_graph"] == ["dist2bbox", "sigmoid"]


def test_r1_002_wrapper_applied_in_export_record(r1):
    exp = r1["export"]
    assert exp["patch_identifier"] == PATCH_ID and exp["graph"]["leaf_clean"] is True
    assert exp["graph"]["out_of_graph_markers_found_in_graph"] == [] and exp["graph"]["non_torch_nn_leaves"] == []
    assert exp["patched_c2f_instances"] and all(p["bottlenecks"] >= 1 for p in exp["patched_c2f_instances"])
    ta = r1["traceability_analysis"]
    assert ta["direct_trace_eval"]["status"] == "FAILED" and ta["upstream_detect_train_output"]["type"] == "dict"  # why a wrapper is needed


# ---------------------------------------------------------------- TC-Y8-R1-003 fx export manifest ----
def test_r1_003_export_manifest():
    m = build_export_manifest(source_model_sha256="a" * 64, source_model_version="v", source_code_version="ultralytics 8.4.173", patch_identifier=PATCH_ID,
                              input_shape=[1, 3, 640, 640], fx_version="torch.fx", torch_version="2.14.1", python_version="3.11.9", timestamp="t",
                              artifact_sha256="b" * 64, artifact_size_bytes=1)
    assert validate_manifest(m) == [] and m["credit_consuming"] is False and m["traceability_boundary"]["patch_id"] == PATCH_ID
    assert validate_manifest({k: m[k] for k in MANIFEST_REQUIRED_KEYS if k != "artifact_sha256"}) == ["artifact_sha256"]
    with pytest.raises(ValueError, match="missing keys"):
        build_export_manifest(source_model_sha256="x")


def test_r1_003_export_record(r1):
    exp = r1["export"]
    assert validate_manifest(exp) == [] and HEX64.fullmatch(exp["artifact_sha256"]) and exp["artifact_size_bytes"] > 1_000_000
    assert exp["source_code_version"].startswith("ultralytics 8.4") and "unmodified" in exp["source_code_version"]
    assert exp["input_shape"] == [1, 3, 640, 640] and exp["torch_version"].startswith("2.")


# ---------------------------------------------------------------- TC-Y8-R1-004 artifact structural validation ----
def test_r1_004_artifact_validation(r1):
    av = r1["artifact_validation"]
    assert av["status"] == "PASS" and av["structural"]["status"] == "PASS" and av["structural"]["details"]["object_kind"] == "graph_module"
    assert av["precheck"]["status"] == "READY" and av["precheck"]["checks"]["fx_is_graph_module"] is True and av["precheck"]["checks"]["output_schema"] is True


# ---------------------------------------------------------------- TC-Y8-R1-005 head re-attach (upstream helper + official fork path) ----
def test_r1_005_head_reattach(r1, crosscheck):
    assert r1["reattach"]["status"] == "PASS" and r1["reattach"]["dfl_weights_equal_checkpoint"] is True
    fork = crosscheck["fork_reattach_r1_body"]
    assert fork["status"] == "PASS" and fork["class"] == "DetectionModel_netspresso" and fork["decoded_shape"] == [1, 84, 8400]
    assert fork["decoded_vs_upstream_pre_nms"]["cosine"] > 0.99999 and fork["decoded_vs_upstream_pre_nms"]["relative_l2_diff"] < 1e-4


# ---------------------------------------------------------------- TC-Y8-R1-006 / 007 input + output schema ----
def test_r1_006_007_schema(r1):
    pre = r1["artifact_validation"]["precheck"]
    assert pre["checks"]["input_shape_compatible"] is True and pre["details"]["input_shape"] == [1, 3, 640, 640]
    head = parse_head_meta({"nc": 80, "nl": 3, "anchors": [[0.0]], "stride": [8.0, 16.0, 32.0], "strides": [[0.0]], "inplace": True})
    assert r1["reattach"]["body_output_shapes"] == expected_body_output_shapes([1, 3, 640, 640], head) == [[1, 144, 80, 80], [1, 144, 40, 40], [1, 144, 20, 20]]
    assert r1["reattach"]["decoded_output_shape"] == expected_decoded_output_shape([1, 3, 640, 640], head) == [1, 84, 8400]


# ---------------------------------------------------------------- TC-Y8-R1-008 pre-NMS equivalence ----
def test_r1_008_equivalence_arithmetic():
    same = equivalence_stats([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert same["cosine"] == pytest.approx(1.0) and same["max_abs_diff"] == 0.0 and same["bitwise_identical"] is True
    off = equivalence_stats([1.0, 0.0], [0.0, 1.0])
    assert off["cosine"] == pytest.approx(0.0) and off["relative_l2_diff"] == pytest.approx(2 ** 0.5)
    assert equivalence_stats([1.0], [1.0, 2.0])["status"] == "SHAPE_MISMATCH" and equivalence_stats([], [])["status"] == "EMPTY"
    v = equivalence_verdict(same, min_cosine=0.999999, max_relative_l2_diff=1e-4)
    assert v["verdict"] == "EQUIVALENT" and "EXAMPLE" in v["thresholds"]["basis"]
    bad = equivalence_verdict({"status": "COMPUTED", "cosine": 0.9944, "max_abs_diff": 50.0, "relative_l2_diff": 0.106}, min_cosine=0.999999, max_relative_l2_diff=1e-4)
    assert bad["verdict"] == "NOT_EQUIVALENT" and len(bad["failed_checks"]) == 2  # the old fork body numbers must not pass
    assert equivalence_verdict({"status": "SHAPE_MISMATCH"}, min_cosine=0.9, max_relative_l2_diff=1.0)["verdict"] == "NOT_COMPARABLE"


def test_r1_008_pre_nms_equivalence_record(r1):
    fe = r1["functional_equivalence"]
    assert fe["images"] >= 8 and fe["verdict_pre_nms_decoded"]["verdict"] == "EQUIVALENT"
    agg = fe["aggregate"]
    assert agg["raw_boxes"]["all_bitwise_identical"] and agg["raw_scores"]["all_bitwise_identical"]  # body maps == upstream raw head outputs
    assert agg["decoded_all"]["min_cosine"] > 0.999999 and agg["decoded_all"]["max_relative_l2_diff"] < 1e-4
    assert agg["decoded_all"]["max_relative_l2_diff"] <= agg["noise_floor_fused_vs_unfused_decoded"]["max_relative_l2_diff"] * 10 + 1e-12  # within ordinary fp32 noise


# ---------------------------------------------------------------- TC-Y8-R1-009 decoded detection equivalence ----
def test_r1_009_detection_comparison():
    ref = [[10, 10, 50, 50, 0.9, 1], [100, 100, 200, 220, 0.6, 3]]
    same = compare_detections(ref, [list(b) for b in ref])
    assert same["all_matched"] and same["max_coordinate_abs_diff"] == 0 and same["max_confidence_abs_diff"] == 0
    shifted = compare_detections(ref, [[10.5, 10, 50, 50, 0.89, 1], [100, 100, 200, 220, 0.6, 3]])
    assert shifted["all_matched"] and shifted["max_coordinate_abs_diff"] == pytest.approx(0.5) and shifted["max_confidence_abs_diff"] == pytest.approx(0.01)
    missing = compare_detections(ref, [[10, 10, 50, 50, 0.9, 1]])
    assert not missing["all_matched"] and missing["unmatched_ref"] == 1 and missing["count_equal"] is False
    wrong_cls = compare_detections(ref, [[10, 10, 50, 50, 0.9, 2], [100, 100, 200, 220, 0.6, 3]])
    assert wrong_cls["matched"] == 1 and wrong_cls["class_ids_ref"] != wrong_cls["class_ids_cand"]


def test_r1_009_one_image_record(r1):
    one = r1["one_image_diagnostic"]
    assert one["status"] == "PASS" and one["image"] == "000000000009.jpg"
    for s in one["settings"].values():
        c = s["comparison"]
        assert c["all_matched"] and c["count_equal"] and c["ref_count"] > 1 and c["max_coordinate_abs_diff"] < 0.01 and c["max_confidence_abs_diff"] < 1e-4


# ---------------------------------------------------------------- TC-Y8-R1-010 COCO128 mAP comparison ----
def test_r1_010_coco128_map_comparison(r1):
    de = r1["detection_evaluation"]
    assert de["baseline"]["status"] == "MEASURED" and de["r1"]["status"] == "MEASURED" and de["baseline"]["evaluator"] == de["r1"]["evaluator"]
    cmp = de["comparison"]
    assert cmp["status"] == "MEASURED" and cmp["drop_pp"] is not None and abs(cmp["drop_pp"]) <= 0.05  # PROJECT-DEFINED EXAMPLE: same model -> within 0.05 pp
    assert cmp["is_output_equivalence_proxy"] is False and r1["r1_criteria"]["coco128_map_equivalent"] is True
    assert de["r1"]["metrics"]["mAP50-95"] > 0.4  # the fork path gave 0.135 for the same weights; R1 must not


# ---------------------------------------------------------------- TC-Y8-R1-011 torch 2.0.1 load ----
def test_r1_011_torch201_load(torch201, r1):
    assert torch201["mode"] == "torch-only" and torch201["torch_version"].startswith("2.0.1")
    tl = torch201["torch_load"]
    assert tl["status"] == "PASS" and tl["is_graph_module"] and tl["output_shapes"] == [[1, 144, 80, 80], [1, 144, 40, 40], [1, 144, 20, 20]]
    assert tl["max_abs_diff"] < 1e-3 and "not a server-acceptance claim" in tl["meaning"]  # cross-torch-version fp noise, not bitwise
    assert torch201["r1_body"]["sha256"] == r1["export"]["artifact_sha256"] and torch201["r1_body"]["matches_dump"] is True


# ---------------------------------------------------------------- TC-Y8-R1-012 raw checksum / reproducibility ----
def test_r1_012_checksums_and_reproducibility(r1, crosscheck):
    rep = r1["export"]["reproducibility"]
    assert rep["status"] == "PASS" and rep["bitwise_identical"] and rep["head_meta_identical"]
    assert HEX64.fullmatch(rep["export_1_sha256"]) and rep["export_1_sha256"] == rep["export_2_sha256"] == r1["export"]["artifact_sha256"]
    assert crosscheck["r1_body"]["sha256"] == r1["export"]["artifact_sha256"]


# ---------------------------------------------------------------- divergence classification (R2 evidence) ----
def test_r1_divergence_helpers():
    layers = [{"index": i, "layer": f"layer_{i:02d}", "layer_type": "Conv", "stats": {"status": "COMPUTED", "cosine": 1.0, "max_abs_diff": 0.0, "relative_l2_diff": 0.0}} for i in range(4)]
    layers.append({"index": 4, "layer": "layer_04", "layer_type": "C2f", "stats": {"status": "COMPUTED", "cosine": 0.98, "max_abs_diff": 1.0, "relative_l2_diff": 0.2}})
    fd = first_divergence(layers, max_relative_l2_diff=1e-4)
    assert fd["layer"] == "layer_04" and fd["layer_type"] == "C2f"
    assert first_divergence(layers[:4], max_relative_l2_diff=1e-4) is None
    assert classify_divergence(None, input_identical=True, weights_identical=True, layer_count=23)["category"] is None
    c = classify_divergence(fd, input_identical=True, weights_identical=True, layer_count=23)
    assert c["category"] == "version difference" and c["verified"] is False and c["category"] in DIVERGENCE_CATEGORIES
    assert classify_divergence(fd, input_identical=False, weights_identical=True, layer_count=23)["category"] == "preprocessing"
    assert classify_divergence(fd, input_identical=True, weights_identical=None, layer_count=23)["category"] == "unknown"
    assert classify_divergence({"index": 22, "layer": "detect"}, input_identical=True, weights_identical=True, layer_count=23)["category"] == "Detect head"


def test_r1_fork_divergence_record_is_consistent(crosscheck):
    diff = crosscheck["fork_model_layer_diff"]
    cls, fd, probe = diff["classification"], diff["first_divergence"], crosscheck["fork_c2f_root_cause_probe"]
    assert cls["category"] in DIVERGENCE_CATEGORIES and cls["confidence"] in ("HIGH", "MEDIUM", "LOW")
    assert fd is not None and fd["layer"] == "layer_04"  # first C2f with n >= 2 in yolov8n
    assert all(rec["stats"]["relative_l2_diff"] < 1e-5 for rec in diff["layers"] if rec["index"] < 4)  # layers 0-3 agree (torch 2.0.1 vs 2.14 fp noise only)
    assert fd["relative_l2_diff"] > 0.1  # layer 4 (first C2f with n >= 2) diverges massively
    if probe["root_cause_verified"]:
        assert cls["verified"] is True and cls["confidence"] == "HIGH" and probe["first_divergence_after"] is None
    else:
        assert cls["verified"] is False  # unverified mechanisms must not be presented as verified
    assert crosscheck["old_fork_body_vs_upstream"]["min_cosine"] < 0.9999  # the old fork artifact is not equivalent (documents the superseded blocker)


def test_r1_fork_evaluator_with_r1_body_isolates_model_code(root):
    """Official fork re-attach path + fork validator on the R1 body must NOT reproduce the 0.135 result of the fork model code."""
    runs = sorted(p for p in (root / "reports" / "yolov8_baseline").glob("2*Z_offline_eval") if (p / "offline_detection_eval.json").is_file())
    recs = [json.loads((p / "offline_detection_eval.json").read_text(encoding="utf-8")) for p in runs]
    r1_runs = [r for r in recs if "R1" in r.get("fixture_label", "")]
    assert r1_runs, "fork-evaluator run of the R1 body expected"
    r = r1_runs[-1]
    assert r["precheck"]["status"] == "READY" and r["reattach"]["class"] == "DetectionModel_netspresso"
    assert r["fixture_accuracy"]["metrics"]["mAP50-95"] > 0.4 and r["baseline_accuracy_same_evaluator"]["metrics"]["mAP50-95"] < 0.2
    assert r["credit_consuming"] is False and r["netspresso_api_calls"] == 0 and r["ledger_unchanged"] is True


# ---------------------------------------------------------------- TC-Y8-R1-013 zero-credit guard ----
def test_r1_013_zero_credit_guard(root, r1, torch201, crosscheck):
    for script in R1_SCRIPTS:
        src = (root / "scripts" / script).read_text(encoding="utf-8")
        assert not re.search(r"^\s*(import|from)\s+netspresso\b", src, re.M), script
        assert "netspresso_adapter" not in src and "NetsPresso(" not in src and "automatic_compression(" not in src, script
        assert "os.environ" not in src and "getenv" not in src, script
        assert "ledger_before" in src and "No NetsPresso API call was made" in src, script
    for rec in (r1, torch201, crosscheck):
        assert rec["credit_consuming"] is False and rec["netspresso_api_calls"] == 0 and rec["ledger_unchanged"] is True
        dumped = json.dumps(rec)
        assert "NETSPRESSO_API_KEY" not in dumped and not re.search(r"[A-Za-z]:[\\\\/]Users|/Users/|/home/", dumped)
    import framework.evaluation.fx_traceability  # noqa: F401

    assert not any(m == "netspresso" or m.startswith("netspresso.") for m in sys.modules)
    ledger = json.loads((root / "reports" / "credit_usage.json").read_text(encoding="utf-8"))
    assert (ledger["used_credit"], ledger["remaining_estimate"], len(ledger["operations"])) == (50, 450, 2)  # after E5-1 (one more real operation, 25 credits observed)

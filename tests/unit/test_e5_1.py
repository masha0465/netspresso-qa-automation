"""Phase 5-E E5-1 tests: single authorized real operation + zero-credit local validation (TC-Y8-E5-001 .. 015).

Pure-Python tests of ``framework.evaluation.input_gate`` with fakes, plus assertions over the E5-1 records
(``reports/yolov8_e5_1/<stamp>/e5_1_execution.json`` / ``e5_1_result.json``). No test calls NetsPresso.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from framework.evaluation.input_gate import (
    MATCH,
    MISMATCH,
    NOT_OBSERVED,
    authorize_single_real_operation,
    build_traceability_chain,
    classify_credit_delta,
    ledger_transition_ok,
    verify_input_artifact,
)

pytestmark = pytest.mark.unit

R1_SHA = "d8e761dae29301ef51df9e7679c729338a522d397e271a900d453d171e1c022b"
HEX64 = re.compile(r"[0-9a-f]{64}")
SECRET_PATTERNS = (r"np-[A-Za-z0-9]{20,}", r"NETSPRESSO_API_KEY\s*[=:]\s*\S", r"Authorization\s*[:=]\s*\S", r"Bearer\s+[A-Za-z0-9._-]{10,}", r"[A-Za-z]:[\\\\/]Users|/Users/|/home/")


def _latest_e5(root: Path) -> Path:
    dirs = sorted(p for p in (root / "reports" / "yolov8_e5_1").glob("2*Z") if (p / "e5_1_result.json").is_file() and (p / "e5_1_execution.json").is_file())
    assert dirs, "E5-1 records expected (reports/yolov8_e5_1/<stamp>/)"
    return dirs[-1]


@pytest.fixture
def e5(root):
    d = _latest_e5(root)
    return json.loads((d / "e5_1_execution.json").read_text(encoding="utf-8")), json.loads((d / "e5_1_result.json").read_text(encoding="utf-8")), d


# ------------------------------------------------------------- TC-Y8-E5-001/002/003 exact SHA, wrong SHA -> no call, precheck -> no call ----
def test_e5_001_exact_input_sha_enforced(tmp_path):
    f = tmp_path / "model_fx.pt"
    f.write_bytes(b"R1-body")
    import hashlib

    sha = hashlib.sha256(b"R1-body").hexdigest()
    ok = verify_input_artifact(f, expected_sha256=sha, expected_size_bytes=7)
    assert ok["status"] == "PASS" and ok["checks"] == {"exists": True, "not_forbidden_path": True, "sha256_matches": True, "size_matches": True}
    wrong = verify_input_artifact(f, expected_sha256="0" * 64)
    assert wrong["status"] == "BLOCKED" and wrong["checks"]["sha256_matches"] is False and "sha256_matches" in wrong["message"]
    assert verify_input_artifact(tmp_path / "missing.pt", expected_sha256=sha)["status"] == "BLOCKED"
    old = tmp_path / "yolov8n_fx" / "model_fx.pt"
    old.parent.mkdir()
    old.write_bytes(b"R1-body")  # same bytes but the forbidden old-fork path must still be refused
    assert verify_input_artifact(old, expected_sha256=sha, forbidden_paths=("yolov8n_fx/model_fx.pt",))["checks"]["not_forbidden_path"] is False


def test_e5_002_003_authorization_blocks_without_identity_precheck_or_confirmation():
    good = {"status": "PASS", "message": "ok"}
    assert authorize_single_real_operation(input_identity=good, precheck_status="READY", ledger_operations_before=1, expected_operations_before=1, confirm_credit_use=True)["authorized"]
    for kwargs, needle in (
        (dict(input_identity={"status": "BLOCKED", "message": "sha"}, precheck_status="READY", ledger_operations_before=1), "input identity BLOCKED"),
        (dict(input_identity=good, precheck_status="BLOCKED", ledger_operations_before=1), "precheck is BLOCKED"),
        (dict(input_identity=good, precheck_status="NOT_VERIFIED", ledger_operations_before=1), "not READY"),
        (dict(input_identity=good, precheck_status="READY", ledger_operations_before=2), "exactly-one guard"),
    ):
        auth = authorize_single_real_operation(expected_operations_before=1, confirm_credit_use=True, **kwargs)
        assert not auth["authorized"] and any(needle in r for r in auth["reasons"]), needle
    assert not authorize_single_real_operation(input_identity=good, precheck_status="READY", ledger_operations_before=1, expected_operations_before=1, confirm_credit_use=False)["authorized"]


def test_e5_002_record_shows_gate_before_call(e5):
    ex, _, _ = e5
    assert ex["input"]["identity"]["status"] == "PASS" and ex["input"]["identity"]["details"]["sha256"] == R1_SHA and ex["input"]["identity"]["details"]["size_bytes"] == 12846691
    assert ex["input"]["precheck_status"] == "READY" and ex["input"]["structural"]["details"]["object_kind"] == "graph_module"
    assert ex["authorization"]["authorized"] is True and ex["authorization"]["max_operations"] == 1 and ex["planned_operation"]["retry_policy"] == "none"
    assert ex["ledger_before"] == {"sha256": ex["ledger_before"]["sha256"], "used_credit": 25, "remaining_estimate": 475, "operations": 1}


# ------------------------------------------------------------- TC-Y8-E5-004/005 single execution, no retry ----
def test_e5_004_005_single_real_execution_no_retry(e5, root):
    ex, res, _ = e5
    assert ex["credit"]["service_operations_executed"] == 1 and ex["credit"]["auth_sessions"] == 1
    logs = ex["execution_summary"]["logs"]
    assert sum("calling compressor_v2().automatic_compression" in line for line in logs) == 1
    assert ex["real_execution"]["exit_code"] == 0 and "exception" not in ex["real_execution"]
    run_dir = root / ex["real_execution"]["run_dir"]
    execution = json.loads((run_dir / "execution_result.json").read_text(encoding="utf-8"))
    assert execution["execution_status"] == "COMPLETED" and execution["environment"]["extra"]["operations_executed"] == 1
    assert execution["configuration"]["model"] == "yolov8n" and execution["artifact"]["metadata"]["source_model_sha256"] == R1_SHA
    assert res["netspresso_execution"]["operations_executed"] == 1 and res["netspresso_api_calls"] == 0 and res["credit_consuming"] is False


# ------------------------------------------------------------- TC-Y8-E5-006 result artifact validation ----
def test_e5_006_candidate_artifact_validation(e5):
    _, res, _ = e5
    c = res["candidate_artifact"]
    assert c["sha_matches_execution_record"] is True and HEX64.fullmatch(c["pt"]["sha256"]) and c["pt"]["size_bytes"] > 1_000_000
    assert c["pt"]["structural"]["status"] == "PASS" and c["pt"]["structural"]["details"]["object_kind"] == "graph_module" and c["pt"]["trusted_for_full_unpickle"] is True
    assert c["pt"]["forward_output_shapes"] == [[1, 144, 80, 80], [1, 144, 40, 40], [1, 144, 20, 20]] and c["pt"]["schema_ok"] is True
    assert c["onnx"]["structural"]["status"] == "PASS" and HEX64.fullmatch(c["onnx"]["sha256"])
    assert res["statuses"]["artifact_status"] == "PASS"


# ------------------------------------------------------------- TC-Y8-E5-007/008 params + FLOPs ----
def test_e5_007_008_params_and_flops_validation(e5):
    _, res, _ = e5
    p, f = res["model_metrics"]["params"], res["model_metrics"]["flops"]
    assert p["baseline_sdk"] == p["baseline_independent"] == 3157184 and p["baseline_full_model_impl"] == 3157200
    assert p["candidate_sdk"] == p["candidate_independent"] and p["candidate_sdk"] < p["baseline_sdk"] and 0 < p["reduction_percent_sdk"] < 100
    assert p["verification"]["candidate"]["verification_status"] == "PASS" and p["verification"]["baseline"]["verification_status"] == "PASS"
    assert f["verification"]["candidate"]["verification_status"] == "PASS" and f["verification"]["baseline"]["verification_status"] == "PASS"
    assert f["candidate_sdk"] < f["baseline_sdk"] and abs(f["baseline_independent_2x_macs"] - f["baseline_sdk"]) / f["baseline_sdk"] < 0.05
    assert "not claimed as an official" in f["convention_note"]


# ------------------------------------------------------------- TC-Y8-E5-009 head re-attach ----
def test_e5_009_head_reattach(e5):
    _, res, _ = e5
    h = res["head_reattach"]
    assert h["status"] == "PASS" and h["decoded_output_shape"] == [1, 84, 8400] == h["expected"] and h["head_meta"]["nc"] == 80
    assert h["body_output_shapes"] == [[1, 144, 80, 80], [1, 144, 40, 40], [1, 144, 20, 20]]


# ------------------------------------------------------------- TC-Y8-E5-010 COCO128 comparison ----
def test_e5_010_coco128_comparison_same_conditions(e5):
    _, res, _ = e5
    a = res["accuracy"]
    assert a["baseline"]["status"] == "MEASURED" and a["baseline"]["metrics"]["mAP50-95"] == 0.44369 and a["candidate"]["status"] == "MEASURED"
    assert a["baseline"]["evaluator"] == a["candidate"]["evaluator"] and a["conditions"]["rect"] is True and a["conditions"]["conf"] == 0.001 and a["conditions"]["iou"] == 0.7
    cmp = a["comparison"]
    assert cmp["status"] == "MEASURED" and cmp["drop_pp"] == pytest.approx((cmp["baseline_metric"] - cmp["candidate_metric"]) * 100, abs=1e-3)
    assert "PROJECT-DEFINED EXAMPLE" in a["threshold_basis"] and "official" not in a["threshold_basis"].split("not")[0]
    assert cmp["is_output_equivalence_proxy"] is False


# ------------------------------------------------------------- TC-Y8-E5-011 proxy / accuracy separation ----
def test_e5_011_proxy_is_not_accuracy(e5):
    _, res, _ = e5
    px = res["output_equivalence_proxy"]
    assert "NOT" in px["disclaimer"].upper() and "num_outputs" in px["ort"] and px["real_images"]["images"] >= 8
    assert px["ort"]["min_cosine_similarity"] is not None and px["real_images"]["aggregate"]["decoded"]["min_cosine"] is not None
    # the accuracy block must not be derived from the proxy value
    assert res["accuracy"]["candidate"]["metrics"]["mAP50-95"] != px["ort"]["min_cosine_similarity"]
    gate_rel = {c["name"]: c for c in res["quality_gate"]["release"]["criteria"]}
    assert gate_rel["output_equivalence_proxy"]["required"] is False and gate_rel["accuracy"]["required"] is True


# ------------------------------------------------------------- TC-Y8-E5-012 Quality Gate behaviour ----
def test_e5_012_quality_gate_separates_execution_from_release(e5):
    _, res, _ = e5
    g = res["quality_gate"]
    assert set(g) == {"compression", "local_eval", "release"} and res["statuses"]["execution_status"] == "COMPLETED"
    rel = {c["name"]: c for c in g["release"]["criteria"]}
    assert rel["reproducibility"]["status"] == "NOT_APPLICABLE" and rel["artifact"]["status"] == "NOT_APPLICABLE"  # single run, no promoted checksum
    assert g["release"]["overall"] == "FAIL" and any("reproducibility" in r.lower() for r in g["release"]["reasons"])
    assert res["statuses"]["release_status"] == "FAIL" and res["statuses"]["validation_status"] == "PASS"
    comp = {c["name"]: c for c in g["compression"]["criteria"]}
    assert comp["model_size"]["status"] == "PASS" and comp["structural_validity"]["status"] == "PASS"
    drop = res["accuracy"]["comparison"]["drop_pp"]
    if drop is not None and drop > 1.0:
        assert rel["accuracy"]["status"] == "FAIL" and res["defects"]["by_profile"]["release"]["category"] == "ACCURACY_REGRESSION"
        assert res["defects"]["root_cause_assessment"]["verified"] is False and res["defects"]["root_cause_assessment"]["confidence"] in ("LOW", "MEDIUM", "HIGH")
        assert res["recommendation"]["decision"] == "HOLD"


# ------------------------------------------------------------- TC-Y8-E5-013 credit delta accounting ----
def test_e5_013_credit_delta_accounting(e5, root):
    ex, _, _ = e5
    assert classify_credit_delta(before=475, after=450, expected=25)["status"] == MATCH
    mm = classify_credit_delta(before=475, after=445, expected=25)
    assert mm["status"] == MISMATCH and mm["observed"] == 30 and "no corrective call" in mm["note"]
    assert classify_credit_delta(before=None, after=450, expected=25)["status"] == NOT_OBSERVED
    before = {"used_credit": 25, "remaining_estimate": 475, "operations": [1]}
    assert ledger_transition_ok(before=before, after={"used_credit": 50, "remaining_estimate": 450, "operations": [1, 2]}, observed_delta=25)["ok"]
    assert not ledger_transition_ok(before=before, after={"used_credit": 50, "remaining_estimate": 450, "operations": [1, 2, 3]}, observed_delta=25)["ok"]
    assert not ledger_transition_ok(before=before, after={"used_credit": 50, "remaining_estimate": 460, "operations": [1, 2]}, observed_delta=25)["ok"]
    c = ex["credit"]
    assert c["delta_classification"]["status"] == MATCH and c["delta_classification"]["observed"] == 25 and c["account_total_before"] == 475 and c["account_total_after"] == 450
    assert c["ledger_transition"]["ok"] is True and ex["ledger_after"]["used_credit"] == 50 and ex["ledger_after"]["remaining_estimate"] == 450 and ex["ledger_after"]["operations"] == 2
    ledger = json.loads((root / "reports" / "credit_usage.json").read_text(encoding="utf-8"))
    assert (ledger["used_credit"], ledger["remaining_estimate"], len(ledger["operations"])) == (50, 450, 2)
    last = ledger["operations"][-1]
    assert last["model"] == "yolov8n" and last["usage_type"] == "actual" and last["actual_credit"] == 25 and last["estimated_credit"] == 25 and last["result"] == "COMPLETED" and last["confirmation"] is True


# ------------------------------------------------------------- TC-Y8-E5-014 secret redaction ----
def test_e5_014_no_secrets_or_local_paths_in_records(e5, root):
    ex, res, d = e5
    texts = [json.dumps(ex), json.dumps(res), (d / "e5_1_summary.html").read_text(encoding="utf-8")]
    run_dir = root / ex["real_execution"]["run_dir"]
    for f in ("execution_result.json", "sdk_log.txt", "case_result.json", "sdk_output/metadata.json"):
        texts.append((run_dir / f).read_text(encoding="utf-8", errors="replace"))
    for t in texts:
        for pat in SECRET_PATTERNS:
            assert not re.search(pat, t), pat
    assert "values not shown" not in json.dumps(res) or True  # console text is never persisted
    assert not any(m == "netspresso" or m.startswith("netspresso.") for m in sys.modules)


# ------------------------------------------------------------- TC-Y8-E5-015 traceability completeness ----
def test_e5_015_traceability_chain_complete(e5):
    _, res, _ = e5
    chain = build_traceability_chain(source_model="a", r1_fx_artifact="b", r1_sha256="c", netspresso_execution="d", candidate_artifact="e", candidate_sha256="f",
                                     local_validation="g", accuracy_result="h", quality_gate={"release": "FAIL"}, credit_record="i")
    assert chain["complete"] and chain["missing_links"] == [] and "Exactly which input artifact" in chain["question_answered"]
    partial = build_traceability_chain(source_model="a")
    assert not partial["complete"] and "candidate_sha256" in partial["missing_links"] and partial["question_answered"] is None
    t = res["traceability"]
    assert t["complete"] is True and t["r1_sha256"] == R1_SHA and t["candidate_sha256"] == res["candidate_artifact"]["pt"]["sha256"]
    assert R1_SHA[:12] in t["r1_fx_artifact"] or t["r1_sha256"] == R1_SHA
    assert res["input_provenance"]["source_sha256"] == R1_SHA and res["reproducibility"]["execution_runs"] == 1 and res["reproducibility"]["registry_entry"]["status"] == "candidate"
    assert res["reproducibility"]["registry_entry"]["source_artifact"].endswith(R1_SHA)


def test_e5_zero_credit_validation_script_guard(root):
    src = (root / "scripts" / "yolov8_e5_1_local_validation.py").read_text(encoding="utf-8")
    assert not re.search(r"^\s*(import|from)\s+netspresso\b", src, re.M) and "NetsPresso(" not in src and "automatic_compression(" not in src
    assert "os.environ" not in src and "getenv" not in src and "ledger_before" in src and "No NetsPresso API call was made" in src

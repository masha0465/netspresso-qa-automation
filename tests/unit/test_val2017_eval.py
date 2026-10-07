"""Phase 5-E COCO val2017 independent-evaluation tests (TC-Y8-VAL-001 .. 006).

Pure-Python tests of the unseen-validation helpers in ``framework.evaluation.recovery`` plus consistency assertions over
``reports/yolov8_e5_1_finetune/<stamp>_val2017/`` records (skipped when the evaluation has not been run). No NetsPresso, no torch.
"""

from __future__ import annotations

import json
import re

import pytest

from framework.evaluation.recovery import (
    VAL_INCONCLUSIVE,
    VAL_OVERFIT,
    VAL_PARTIAL,
    VAL_PASS_CANDIDATE,
    VAL_REGRESSION,
    classify_unseen_recovery,
    generalization_gap,
    unseen_next_step,
)

pytestmark = pytest.mark.unit

SHAS = {"baseline": "d8e761dae29301ef51df9e7679c729338a522d397e271a900d453d171e1c022b", "e5_1": "21d8cbd1361cfd3ef55c0c132d11f3fad294dbde6f175e8082bc0233b860e0f1",
        "phase_c": "dba5cd214f0f4098c2708366fb8ee2a3f2d4e5681b7668fc49f36aa148a4d9ee"}
SECRET_PATTERNS = (r"np-[A-Za-z0-9]{20,}", r"NETSPRESSO_API_KEY\s*[=:]\s*\S", r"Authorization\s*[:=]\s*\S", r"Bearer\s+[A-Za-z0-9._-]{10,}", r"[A-Za-z]:[\\\\/]Users|/Users/|/home/")


@pytest.fixture
def val_record(root):
    dirs = sorted(p for p in (root / "reports" / "yolov8_e5_1_finetune").glob("2*Z_val2017") if (p / "val2017_summary.json").is_file())
    if not dirs:
        pytest.skip("val2017 evaluation not run")
    d = dirs[-1]
    return d, json.loads((d / "val2017_summary.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- TC-Y8-VAL-001 generalization gap ----
def test_val_001_generalization_gap():
    g = generalization_gap(smoke=0.46688, unseen=0.30)
    assert g["status"] == "MEASURED" and g["gap_pp"] == pytest.approx(16.688) and g["unseen_over_smoke_percent"] == pytest.approx(64.26, abs=0.01)
    assert generalization_gap(smoke=0.44369, unseen=None)["status"] == "N/A"


# ---------------------------------------------------------------- TC-Y8-VAL-002 unseen recovery classification (CASE A-D) ----
def test_val_002_unseen_classification_cases():
    base = dict(baseline=0.370, compressed=0.0, baseline_smoke=0.44369, candidate_smoke=0.46688)
    assert classify_unseen_recovery(candidate=0.365, **base)["decision"] == VAL_PASS_CANDIDATE  # drop 0.5 pp -> CASE A
    b = classify_unseen_recovery(candidate=0.345, **base)
    assert b["decision"] == VAL_PARTIAL and b["case"] == "B" and b["near_pass"] is True  # drop 2.5 pp
    ov = classify_unseen_recovery(candidate=0.20, **base)  # drop 17 pp, candidate gap 26.7 vs baseline gap 7.4 -> excess 19.3 > 10 -> overfit
    assert ov["decision"] == VAL_OVERFIT and ov["case"] == "C" and ov["excess_gap_vs_baseline_pp"] > 10
    d = classify_unseen_recovery(baseline=0.370, compressed=0.0, candidate=0.30, baseline_smoke=None, candidate_smoke=None)  # no smoke info: recovered 30 pp, drop 7 pp -> CASE D
    assert d["decision"] == VAL_PARTIAL and d["case"] == "D"
    r = classify_unseen_recovery(baseline=0.370, compressed=0.30, candidate=0.31, baseline_smoke=None, candidate_smoke=None)  # +1 pp only, drop 6 pp -> regression
    assert r["decision"] == VAL_REGRESSION and r["case"] == "C"
    assert classify_unseen_recovery(baseline=None, compressed=0.0, candidate=0.3)["decision"] == VAL_INCONCLUSIVE
    assert "EXAMPLE" in classify_unseen_recovery(candidate=0.365, **base)["thresholds"]["basis"]


# ---------------------------------------------------------------- TC-Y8-VAL-003 next step is single-valued and never spends credits ----
def test_val_003_next_step_table():
    for dec, label in ((VAL_PASS_CANDIDATE, "E5-3 CONSIDER"), (VAL_PARTIAL, "LOCAL RECOVERY"), (VAL_REGRESSION, "REVIEW"), (VAL_OVERFIT, "HOLD"), (VAL_INCONCLUSIVE, "STOP")):
        ns = unseen_next_step(dec)
        assert ns["label"] == label and ns["netspresso_additional_execution"] == "HOLD"
    assert unseen_next_step("???")["label"] == "STOP"


# ---------------------------------------------------------------- TC-Y8-VAL-004 record consistency ----
def test_val_004_record_consistency(val_record):
    d, r = val_record
    assert r["training_performed"] is False and r["credit_consuming"] is False and r["netspresso_api_calls"] == 0 and r["netspresso_additional_execution"] == "HOLD"
    cs = r["credit_safety"]
    assert cs["ledger_unchanged"] is True and cs["api_calls"] == 0 and cs["sdk_execution"] == 0 and (cs["ledger_after"]["used"], cs["ledger_after"]["remaining"], cs["ledger_after"]["operations"]) == (50, 450, 2)
    ds = r["dataset"]
    assert ds["images"] == 5000 and ds["contamination_check"]["content_sha256_overlap"] == 0 and ds["contamination_check"]["filename_overlap"] == 0 and ds["coco128_train_equals_val"] is True
    rows = {row["model"]: row for row in r["accuracy"]["val2017"]}
    assert set(rows) == {"baseline", "e5_1", "phase_c"} and all(rows[k]["sha256"] == SHAS[k] for k in SHAS)
    b, c1, pc = rows["baseline"]["mAP50-95"], rows["e5_1"]["mAP50-95"], rows["phase_c"]["mAP50-95"]
    assert r["recovery"]["remaining_drop_pp"] == pytest.approx((b - pc) * 100, abs=1e-3) and r["recovery"]["e5_1_to_phase_c_pp"] == pytest.approx((pc - c1) * 100, abs=1e-3)
    assert r["classification"] == classify_unseen_recovery(baseline=b, compressed=c1, candidate=pc, baseline_smoke=0.44369, candidate_smoke=0.46688) and r["decision"] == r["classification"]["decision"]
    assert r["generalization"]["phase_c"]["gap_pp"] == pytest.approx((0.46688 - pc) * 100, abs=1e-3)
    assert r["evaluation_config"]["conf"] == 0.001 and r["evaluation_config"]["iou"] == 0.7 and r["evaluation_config"]["rect"] is True and r["evaluation_config"]["imgsz"] == 640
    assert all(v["immutable_across_evaluations"] for v in r["model_integrity"].values())
    assert r["baseline_promotion"].startswith("NOT") and r["reproducibility"]["level"] == "NOT_VERIFIED"
    rel = {c["name"]: c for c in r["quality_gate"]["release"]["criteria"]}
    assert rel["reproducibility"]["status"] == "NOT_APPLICABLE" and r["quality_gate"]["release"]["overall"] == "FAIL"  # single run, no promoted checksum
    for f in ("dataset_manifest.json", "evaluation_config.json", "validation_results.json", "model_integrity.json", "environment_fingerprint.json", "val2017_summary.json", "val2017_summary.html"):
        assert (d / f).is_file(), f
    for k in SHAS:
        v = json.loads((d / f"validation_{k}.json").read_text(encoding="utf-8"))
        assert v["immutable"] is True and v["evaluation_config"]["grad_enabled"] is False and v["evaluation_config"]["training"] is False and v["ledger"]["unchanged"] is True


# ---------------------------------------------------------------- TC-Y8-VAL-005 proxy is not accuracy ----
def test_val_005_proxy_separated(val_record):
    _, r = val_record
    px = r["output_equivalence_proxy"]
    assert "NOT" in px["disclaimer"].upper() and set(px["aggregate"]) == {"baseline_vs_phase_c", "e5_1_vs_phase_c", "baseline_vs_e5_1"}
    rows = {row["model"]: row for row in r["accuracy"]["val2017"]}
    assert rows["phase_c"]["mAP50-95"] != px["aggregate"]["baseline_vs_phase_c"]["min_cosine"]


# ---------------------------------------------------------------- TC-Y8-VAL-006 zero-credit guard + secrets ----
def test_val_006_script_guards_and_secrets(root, val_record):
    src = (root / "scripts" / "yolov8_val2017_eval.py").read_text(encoding="utf-8")
    assert not re.search(r"^\s*(import|from)\s+netspresso\b", src, re.M) and "NetsPresso(" not in src and "automatic_compression(" not in src
    env_uses = [line.strip() for line in src.splitlines() if "os.environ" in line]
    assert env_uses and all('"NETSPRESSO_API_KEY" in os.environ' in line for line in env_uses)
    assert "torch.optim" not in src and ".backward(" not in src and "optimizer.step" not in src and "set_grad_enabled(False)" in src  # no training code (docstring may say "no optimizer")
    d, r = val_record
    for text in (json.dumps(r), (d / "val2017_summary.html").read_text(encoding="utf-8"), (d / "dataset_manifest.json").read_text(encoding="utf-8")):
        for pat in SECRET_PATTERNS:
            assert not re.search(pat, text), pat

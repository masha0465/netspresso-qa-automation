"""Phase 5-E fine-tuning recovery tests (TC-Y8-FT-001 .. 012).

Pure-Python tests of ``framework.evaluation.recovery`` plus consistency assertions over the recovery records written by
``scripts/yolov8_e5_1_finetune.py`` (reports/yolov8_e5_1_finetune/<stamp>_phase<X>/finetune_result.json). No NetsPresso, no torch.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from framework.evaluation.recovery import (
    BLOCKED,
    FINETUNE_CONFIG_REQUIRED,
    NEXT_STEPS,
    PARTIAL_RECOVERY,
    RECOVERY_FAILED,
    RECOVERY_SUCCESS,
    architecture_unchanged,
    classify_recovery,
    epoch_trend,
    recommend_next_step,
    three_way_comparison,
    validate_finetune_config,
)

pytestmark = pytest.mark.unit

E5_SHA = "21d8cbd1361cfd3ef55c0c132d11f3fad294dbde6f175e8082bc0233b860e0f1"
R1_SHA = "d8e761dae29301ef51df9e7679c729338a522d397e271a900d453d171e1c022b"
HEX64 = re.compile(r"[0-9a-f]{64}")
SECRET_PATTERNS = (r"np-[A-Za-z0-9]{20,}", r"NETSPRESSO_API_KEY\s*[=:]\s*\S", r"Authorization\s*[:=]\s*\S", r"Bearer\s+[A-Za-z0-9._-]{10,}", r"[A-Za-z]:[\\\\/]Users|/Users/|/home/")


def _records(root: Path) -> list[tuple[Path, dict]]:
    dirs = sorted(p for p in (root / "reports" / "yolov8_e5_1_finetune").glob("2*Z_phase*") if (p / "finetune_result.json").is_file())
    assert dirs, "fine-tuning records expected (reports/yolov8_e5_1_finetune/<stamp>_phase<X>/)"
    return [(d, json.loads((d / "finetune_result.json").read_text(encoding="utf-8"))) for d in dirs]


@pytest.fixture
def records(root):
    return _records(root)


@pytest.fixture
def final(records):
    return records[-1]


# ---------------------------------------------------------------- TC-Y8-FT-001 three-way comparison + recovery arithmetic ----
def test_ft_001_three_way_and_recovery_percent():
    t = three_way_comparison(baseline=0.44369, compressed=0.0, fine_tuned=0.22)
    assert t["status"] == "MEASURED" and t["baseline_to_compressed_pp"] == pytest.approx(-44.369) and t["baseline_to_fine_tuned_pp"] == pytest.approx(-22.369)
    assert t["compressed_to_fine_tuned_pp"] == pytest.approx(22.0) and t["remaining_drop_pp"] == pytest.approx(22.369) and t["recovery_percent"] == pytest.approx(49.58, abs=0.01)
    assert three_way_comparison(baseline=0.44369, compressed=0.0, fine_tuned=None)["status"] == "N/A"
    full = three_way_comparison(baseline=0.4, compressed=0.0, fine_tuned=0.4)
    assert full["recovery_percent"] == 100.0 and full["remaining_drop_pp"] == 0.0
    assert three_way_comparison(baseline=0.4, compressed=0.4, fine_tuned=0.4)["recovery_percent"] is None  # nothing was lost


# ---------------------------------------------------------------- TC-Y8-FT-002 recovery classification ----
def test_ft_002_recovery_classification_never_success_on_mere_improvement():
    base = dict(baseline=0.44369, compressed=0.0)
    assert classify_recovery(three_way_comparison(**base, fine_tuned=0.44), max_drop_pp=1.0)["decision"] == RECOVERY_SUCCESS
    assert classify_recovery(three_way_comparison(**base, fine_tuned=0.42), max_drop_pp=1.0)["decision"] == PARTIAL_RECOVERY  # improved a lot, still > 1 pp
    assert classify_recovery(three_way_comparison(**base, fine_tuned=0.06), max_drop_pp=1.0)["decision"] == PARTIAL_RECOVERY  # +6 pp material
    assert classify_recovery(three_way_comparison(**base, fine_tuned=0.02), max_drop_pp=1.0)["decision"] == RECOVERY_FAILED  # +2 pp immaterial
    assert classify_recovery(three_way_comparison(**base, fine_tuned=0.0), max_drop_pp=1.0)["decision"] == RECOVERY_FAILED
    assert classify_recovery(three_way_comparison(**base, fine_tuned=0.44), max_drop_pp=1.0, structural_ok=False)["decision"] == BLOCKED
    assert classify_recovery(three_way_comparison(**base, fine_tuned=None), max_drop_pp=1.0)["decision"] == BLOCKED
    assert "EXAMPLE" in classify_recovery(three_way_comparison(**base, fine_tuned=0.44), max_drop_pp=1.0)["thresholds"]["basis"]


# ---------------------------------------------------------------- TC-Y8-FT-003 next-step recommendation ----
def test_ft_003_next_step_is_exactly_one_of_a_to_e_and_never_spends_on_weak_recovery():
    base = dict(baseline=0.44369, compressed=0.0)
    assert recommend_next_step(RECOVERY_SUCCESS, three_way_comparison(**base, fine_tuned=0.44))["code"] == "A"
    assert recommend_next_step(PARTIAL_RECOVERY, three_way_comparison(**base, fine_tuned=0.30))["code"] == "B"  # 67 % recovered -> more local epochs
    assert recommend_next_step(PARTIAL_RECOVERY, three_way_comparison(**base, fine_tuned=0.08))["code"] == "C"  # 18 % -> ratio too aggressive
    assert recommend_next_step(RECOVERY_FAILED, three_way_comparison(**base, fine_tuned=0.0))["code"] == "E"
    assert recommend_next_step(BLOCKED, three_way_comparison(**base, fine_tuned=None))["code"] == "B"
    for code, label in NEXT_STEPS.items():
        assert code in "ABCDE" and label
    for dec in (PARTIAL_RECOVERY, RECOVERY_FAILED, BLOCKED):
        text = " ".join(recommend_next_step(dec, three_way_comparison(**base, fine_tuned=0.08))["rationale"]).lower()
        assert "75 credits" not in text or "not" in text or "do not" in text


# ---------------------------------------------------------------- TC-Y8-FT-004 config validation / architecture guard / trend ----
def test_ft_004_config_architecture_and_trend_helpers():
    cfg = dict.fromkeys(FINETUNE_CONFIG_REQUIRED, "x")
    assert validate_finetune_config(cfg) == [] and validate_finetune_config({**cfg, "seed": None}) == ["seed"]
    assert architecture_unchanged(params_before=882996, params_after=882996, macs_before=1916046400, macs_after=1916046400)["status"] == "PASS"
    assert architecture_unchanged(params_before=882996, params_after=900000, macs_before=1, macs_after=1)["status"] == "FAIL"
    assert architecture_unchanged(params_before=None, params_after=1, macs_before=None, macs_after=None)["status"] == "NOT_APPLICABLE"
    hist = [{"epoch": 1, "map50_95": 0.01}, {"epoch": 2}, {"epoch": 3, "map50_95": 0.05}, {"epoch": 4, "map50_95": 0.04}]
    tr = epoch_trend(hist)
    assert tr["best_epoch"] == 3 and tr["final_value"] == 0.04 and tr["still_improving_at_end"] is False and tr["epochs_recorded"] == 3
    assert epoch_trend([])["status"] == "N/A"


# ---------------------------------------------------------------- TC-Y8-FT-005 E5-1 candidate loading + source SHA traceability (record) ----
def test_ft_005_candidate_loading_and_source_traceability(records):
    for _, r in records:
        v = r["e5_1_verification"]
        assert all(v["checks"].values()) and v["candidate_identity"]["status"] == "PASS" and v["candidate_identity"]["details"]["sha256"] == E5_SHA
        assert v["e5_1_files_immutable"] is True and v["files_sha256"]["sdk_output.pt"] == E5_SHA
        assert r["finetune_config"]["e5_1_candidate_sha256"] == E5_SHA and r["finetune_config"]["r1_input_sha256"] == R1_SHA
        if r["finetune_config"]["phase"] == "A":
            assert r["finetune_config"]["source_artifact_sha256"] == E5_SHA  # phase A starts from the NetsPresso candidate itself
        assert r["traceability"]["candidate_sha256"] == E5_SHA and r["traceability"]["r1_sha256"] == R1_SHA and r["traceability"]["complete"] is True


# ---------------------------------------------------------------- TC-Y8-FT-006 output artifact + fine-tuning configuration (record) ----
def test_ft_006_output_artifact_and_config(final):
    d, r = final
    cfg = r["finetune_config"]
    assert validate_finetune_config(cfg) == [] and cfg["workers"] == 0 and cfg["device"] == "cpu" and cfg["dataset"]["identity"] == "coco128.yaml" and cfg["dataset"]["train_equals_val"] is True
    av = r["artifact_validation"]
    assert HEX64.fullmatch(av["pt"]["sha256"]) and av["pt"]["sha256"] != E5_SHA and av["pt"]["path"].startswith("outputs/models/yolov8n_e5_1_finetuned/")
    assert av["pt"]["structural"]["status"] == "PASS" and av["pt"]["structural"]["details"]["object_kind"] == "graph_module"
    assert av["pt"]["forward_output_shapes"] == [[1, 144, 80, 80], [1, 144, 40, 40], [1, 144, 20, 20]] and av["pt"]["decoded_output_shape"] == [1, 84, 8400]
    assert av["params"]["fine_tuned"] == 882996 and av["architecture_unchanged"]["status"] == "PASS" and av["status"] == "PASS"
    assert r["reproducibility"]["output_artifact_sha256"] == av["pt"]["sha256"] and r["reproducibility"]["training_runs"] == 1
    assert (d / "finetune_config.json").is_file() and (d / "finetune_accuracy.json").is_file() and (d / "finetune_summary.html").is_file()


# ---------------------------------------------------------------- TC-Y8-FT-007 three-way accuracy comparison (record) ----
def test_ft_007_three_way_record_consistency(final):
    _, r = final
    a = r["accuracy"]
    assert a["baseline"]["metrics"]["mAP50-95"] == 0.44369 and a["compressed"]["metrics"]["mAP50-95"] == 0.0 and a["fine_tuned"]["status"] == "MEASURED"
    assert a["baseline"]["evaluator"] == a["fine_tuned"]["evaluator"] and a["conditions"]["rect"] is True and a["conditions"]["conf"] == 0.001 and a["conditions"]["iou"] == 0.7
    t = a["three_way"]
    recomputed = three_way_comparison(baseline=t["baseline"], compressed=t["compressed"], fine_tuned=t["fine_tuned"])
    assert recomputed == t
    assert r["recovery"]["decision"] in (RECOVERY_SUCCESS, PARTIAL_RECOVERY, RECOVERY_FAILED, BLOCKED)
    assert r["recovery"]["decision"] == classify_recovery(t, max_drop_pp=a["threshold_pp"], structural_ok=r["artifact_validation"]["status"] == "PASS")["decision"]
    assert "PROJECT-DEFINED EXAMPLE" in a["threshold_basis"] and "never declared" in a["threshold_basis"]


# ---------------------------------------------------------------- TC-Y8-FT-008 proxy vs accuracy separation ----
def test_ft_008_proxy_is_not_accuracy(final):
    _, r = final
    px = r["output_equivalence_proxy"]
    assert "NOT" in px["disclaimer"].upper() and px["real_images"]["images"] >= 8
    agg = px["real_images"]["aggregate"]
    assert set(agg) >= {"baseline_vs_fine_tuned", "compressed_vs_fine_tuned", "baseline_vs_compressed"}  # phase C adds previous_vs_fine_tuned
    assert agg["baseline_vs_compressed"]["min_cosine"] == pytest.approx(0.984, abs=0.005)  # matches the E5-1 observation
    assert r["accuracy"]["fine_tuned"]["metrics"]["mAP50-95"] != agg["baseline_vs_fine_tuned"]["min_cosine"]
    rel = {c["name"]: c for c in r["quality_gate"]["release"]["criteria"]}
    assert rel["output_equivalence_proxy"]["required"] is False and rel["accuracy"]["required"] is True


# ---------------------------------------------------------------- TC-Y8-FT-009 Quality Gate behaviour ----
def test_ft_009_quality_gate(final):
    _, r = final
    g = r["quality_gate"]
    assert set(g) == {"compression", "local_eval", "release"}
    rel = {c["name"]: c for c in g["release"]["criteria"]}
    assert rel["reproducibility"]["status"] == "NOT_APPLICABLE" and g["release"]["overall"] == "FAIL"  # single local run, no val2017 -> release never PASS here
    comp = {c["name"]: c for c in g["compression"]["criteria"]}
    assert comp["model_size"]["status"] == "PASS" and comp["structural_validity"]["status"] == "PASS"
    remaining = r["accuracy"]["three_way"]["remaining_drop_pp"]
    if remaining > r["accuracy"]["threshold_pp"]:
        assert rel["accuracy"]["status"] == "FAIL" and r["defects"]["release"]["category"] == "ACCURACY_REGRESSION"
    else:
        assert rel["accuracy"]["status"] == "PASS"


# ---------------------------------------------------------------- TC-Y8-FT-010 no NetsPresso / no key / E5-1 immutable ----
def test_ft_010_zero_credit_and_historical_immutability(records, root):
    src = (root / "scripts" / "yolov8_e5_1_finetune.py").read_text(encoding="utf-8")
    assert not re.search(r"^\s*(import|from)\s+netspresso\b", src, re.M) and "NetsPresso(" not in src and "automatic_compression(" not in src
    # the only environment access allowed is the presence guard (`"NETSPRESSO_API_KEY" in os.environ`); the value is never read
    env_uses = [line.strip() for line in src.splitlines() if "os.environ" in line]
    assert env_uses and all('"NETSPRESSO_API_KEY" in os.environ' in line for line in env_uses), env_uses
    assert "getenv" not in src and "os.environ[" not in src and "os.environ.get" not in src and "No NetsPresso API call was made" in src
    for _, r in records:
        cs = r["credit_safety"]
        assert cs["ledger_unchanged"] is True and cs["api_calls"] == 0 and cs["sdk_execution"] == 0 and r["netspresso_api_calls"] == 0 and r["credit_consuming"] is False
        assert (cs["ledger_before"]["used"], cs["ledger_before"]["remaining"], cs["ledger_before"]["operations"]) == (50, 450, 2) == (cs["ledger_after"]["used"], cs["ledger_after"]["remaining"], cs["ledger_after"]["operations"])
        assert r["recovery_assessment"]["historical_records_modified"] is False
        assert set(r["recovery_assessment"]["historical_e5_1_defects"].values()) >= {"E5-1-release-ACCURACY_REGRESSION", "E5-1-local_eval-ACCURACY_REGRESSION"}
    e5 = json.loads((root / "reports" / "yolov8_e5_1" / "20261005T040009Z" / "e5_1_result.json").read_text(encoding="utf-8"))
    assert e5["accuracy"]["candidate"]["metrics"]["mAP50-95"] == 0.0 and e5["candidate_artifact"]["pt"]["sha256"] == E5_SHA  # history untouched
    ledger = json.loads((root / "reports" / "credit_usage.json").read_text(encoding="utf-8"))
    assert (ledger["used_credit"], ledger["remaining_estimate"], len(ledger["operations"])) == (50, 450, 2)
    assert not any(m == "netspresso" or m.startswith("netspresso.") for m in sys.modules)


# ---------------------------------------------------------------- TC-Y8-FT-011 secrets / paths ----
def test_ft_011_no_secrets_or_local_paths(records):
    for d, r in records:
        for text in (json.dumps(r), (d / "finetune_summary.html").read_text(encoding="utf-8"), (d / "finetune_config.json").read_text(encoding="utf-8")):
            for pat in SECRET_PATTERNS:
                assert not re.search(pat, text), pat


# ---------------------------------------------------------------- TC-Y8-FT-012 phase chaining ----
def test_ft_012_phase_b_chains_from_phase_a(records):
    phases = {r["finetune_config"]["phase"]: r for _, r in records}
    if "B" in phases:
        b = phases["B"]
        assert b["experiment"]["phase_a_reference"] is not None and b["finetune_config"]["source_artifact_sha256"] == b["experiment"]["phase_a_reference"]["fine_tuned_sha256"]
        assert "via phase A" in b["traceability"]["local_validation"]
    if "C" in phases:
        c, b = phases["C"], phases["B"]
        assert c["finetune_config"]["source_artifact_sha256"] == b["artifact_validation"]["pt"]["sha256"] and c["experiment"]["prev_reference"]["fine_tuned_sha256"] == b["artifact_validation"]["pt"]["sha256"]
        assert [p["phase"] for p in c["experiment"]["previous_phases"]] == ["A", "B"] and "phase B" in c["traceability"]["local_validation"] and "phase A" in c["traceability"]["local_validation"]


# ---------------------------------------------------------------- TC-Y8-FT-013 continuation helpers (phase C) ----
def test_ft_013_continuation_helpers():
    from framework.evaluation.recovery import (
        MILESTONES,
        PHASE_C_CONTINUE,
        PHASE_C_GO_VAL2017,
        PHASE_C_PLATEAU,
        PHASE_C_STOP,
        continuation_metrics,
        milestones_reached,
        phase_c_decision,
        phase_c_next_step,
        plateau_detected,
    )

    assert MILESTONES["M3"] == pytest.approx(0.44369 - 0.01) and milestones_reached(0.41) == {"M1": True, "M2": True, "M3": False} and milestones_reached(None) == {"M1": False, "M2": False, "M3": False}
    cm = continuation_metrics(current=0.30, previous_best=0.28965, baseline=0.44369, compressed=0.0)
    assert cm["delta_from_previous_best"] == pytest.approx(0.01035) and cm["remaining_gap"] == pytest.approx(0.14369) and cm["recovery_rate"] == pytest.approx(0.6761, abs=1e-4)
    assert plateau_detected([0.29, 0.291, 0.292, 0.293])["plateau"] is True and plateau_detected([0.29, 0.30, 0.31, 0.33])["plateau"] is False and plateau_detected([0.29, 0.30])["plateau"] is False
    assert phase_c_decision(best_value=0.44, previous_best=0.29, stopped_reason="completed", plateau=False, final_dropped_below_best=False, structural_ok=True)["decision"] == PHASE_C_GO_VAL2017
    assert phase_c_decision(best_value=0.33, previous_best=0.29, stopped_reason="plateau", plateau=True, final_dropped_below_best=False, structural_ok=True)["decision"] == PHASE_C_PLATEAU
    assert phase_c_decision(best_value=0.33, previous_best=0.29, stopped_reason="non-finite loss", plateau=False, final_dropped_below_best=False, structural_ok=True)["decision"] == PHASE_C_STOP
    assert phase_c_decision(best_value=0.33, previous_best=0.29, stopped_reason="completed", plateau=False, final_dropped_below_best=False, structural_ok=False)["decision"] == PHASE_C_STOP
    cont = phase_c_decision(best_value=0.36, previous_best=0.29, stopped_reason="completed", plateau=False, final_dropped_below_best=True, structural_ok=True)
    assert cont["decision"] == PHASE_C_CONTINUE and cont["case"] == "C" and "rollback" in cont["reason"]
    for d in (PHASE_C_GO_VAL2017, PHASE_C_PLATEAU, PHASE_C_STOP, PHASE_C_CONTINUE):
        ns = phase_c_next_step({"decision": d}, improvement_from_prev=0.05)
        assert ns["label"] in ("MORE LOCAL FINE-TUNING", "COCO VAL2017", "HOLD", "STOP", "REVIEW BEFORE NETSPRESSO") and ns["netspresso"] == "HOLD"


def test_ft_014_phase_c_record(records):
    phases = {r["finetune_config"]["phase"]: (d, r) for d, r in records}
    if "C" not in phases:
        pytest.skip("phase C not run")
    d, r = phases["C"]
    t = r["training"]
    assert t["epoch_offset"] == 40 and t["global_epoch_range"][0] == 41 and all(h["global_epoch"] == 40 + h["epoch"] for h in t["history"])
    assert t["validation_points"] and all("continuation" in v and v["continuation"]["status"] == "MEASURED" for v in t["validation_points"])
    assert t["best"]["map50_95"] is not None and t["best"]["map50_95"] >= r["accuracy"]["fine_tuned"]["metrics"]["mAP50-95"] - 1e-9
    assert r["continuation_decision"]["decision"] in ("PHASE_C_CONTINUE", "PHASE_C_PLATEAU", "PHASE_C_STOP", "PHASE_C_GO_VAL2017") and r["netspresso_next_execution"] == "HOLD"
    assert [row["model"] for row in r["comparison_table"]] == ["baseline", "E5-1 compressed", "fine-tuned A", "fine-tuned B", "fine-tuned C"]
    assert all(row["params"] == 882996 for row in r["comparison_table"][1:]) and r["comparison_table"][0]["params"] == 3157184
    assert "previous_vs_fine_tuned" in r["output_equivalence_proxy"]["real_images"]["aggregate"] and r["performance"].get("previous") and r["performance"].get("e5_1")
    # phaseC_result.json is a byte-identical duplicate of finetune_result.json and is intentionally excluded from the
    # public repository; FT-014 validates the canonical committed phase C record instead.
    for f in ("finetune_config.json", "training_history.json", "validation_results.json", "finetune_result.json", "environment_fingerprint.json", "artifact_validation.json", "recovery_summary.json", "finetune_summary.html"):
        assert (d / f).is_file(), f

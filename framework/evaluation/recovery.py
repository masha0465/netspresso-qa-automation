"""Accuracy-recovery bookkeeping for a locally fine-tuned compressed candidate (pure Python).

Used by ``scripts/yolov8_e5_1_finetune.py``. No torch here: the arithmetic and the decision rules are
unit-testable on the main interpreter.

Terminology (kept strict):
* *baseline*  - the untouched upstream model's metric (COCO128 smoke, 0.44369 for YOLOv8n).
* *compressed* - the NetsPresso-generated E5-1 candidate BEFORE any local training (0.0).
* *fine-tuned* - the locally fine-tuned artifact derived from the compressed candidate (NOT a NetsPresso artifact).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

RECOVERY_SUCCESS, PARTIAL_RECOVERY, RECOVERY_FAILED, BLOCKED = "RECOVERY_SUCCESS", "PARTIAL_RECOVERY", "RECOVERY_FAILED", "BLOCKED"
NEXT_STEPS = {
    "A": "PROCEED TO COCO VAL2017",
    "B": "HOLD - MORE LOCAL FINE-TUNING",
    "C": "HOLD - REDUCE COMPRESSION RATIO",
    "D": "NO-GO - CHANGE MODEL",
    "E": "NO-GO - ABANDON THIS COMPRESSION PATH",
}
FINETUNE_CONFIG_REQUIRED = ("seed", "epochs", "batch", "imgsz", "optimizer", "lr", "weight_decay", "freeze", "augmentation", "workers",
                            "torch_version", "ultralytics_version", "dataset", "source_artifact_sha256", "phase")


def _pp(a: float, b: float) -> float:
    return round((a - b) * 100.0, 4)


def three_way_comparison(*, baseline: float | None, compressed: float | None, fine_tuned: float | None, metric: str = "mAP50-95") -> dict[str, Any]:
    """Baseline / compressed / fine-tuned deltas in percentage points plus the recovery share.

    recovery_percent = (fine_tuned - compressed) / (baseline - compressed) * 100, i.e. how much of the lost
    accuracy was won back (100 = fully back to baseline). None when any input is missing or the loss is zero.
    """
    if baseline is None or compressed is None or fine_tuned is None:
        return {"status": "N/A", "metric": metric, "reason": "one or more metrics not measured", "baseline": baseline, "compressed": compressed, "fine_tuned": fine_tuned}
    lost = baseline - compressed
    recovery = round((fine_tuned - compressed) / lost * 100.0, 2) if lost > 0 else None
    return {
        "status": "MEASURED", "metric": metric, "baseline": baseline, "compressed": compressed, "fine_tuned": fine_tuned,
        "baseline_to_compressed_pp": _pp(compressed, baseline), "baseline_to_fine_tuned_pp": _pp(fine_tuned, baseline), "compressed_to_fine_tuned_pp": _pp(fine_tuned, compressed),
        "remaining_drop_pp": _pp(baseline, fine_tuned), "recovery_percent": recovery,
        "relative_to_baseline_percent": round(fine_tuned / baseline * 100.0, 2) if baseline else None,
    }


def classify_recovery(comparison: Mapping[str, Any], *, max_drop_pp: float, material_improvement_pp: float = 5.0, unusable_below: float = 0.05,
                      structural_ok: bool = True, evaluation_ok: bool = True) -> dict[str, Any]:
    """Map the three-way comparison onto RECOVERY_SUCCESS / PARTIAL_RECOVERY / RECOVERY_FAILED / BLOCKED.

    * BLOCKED            - training/evaluation/structure could not be completed.
    * RECOVERY_SUCCESS   - remaining drop <= ``max_drop_pp`` (PROJECT-DEFINED EXAMPLE) and structure intact.
    * PARTIAL_RECOVERY   - improved by >= ``material_improvement_pp`` and the metric is above ``unusable_below``, but still over the threshold.
    * RECOVERY_FAILED    - otherwise (near zero or immaterial gain).
    "Improved" alone never yields RECOVERY_SUCCESS.
    """
    if not structural_ok or not evaluation_ok or comparison.get("status") != "MEASURED":
        return {"decision": BLOCKED, "reason": "structure/evaluation incomplete" if comparison.get("status") == "MEASURED" else f"comparison {comparison.get('status')}",
                "thresholds": {"max_drop_pp": max_drop_pp, "material_improvement_pp": material_improvement_pp, "unusable_below": unusable_below, "basis": "PROJECT-DEFINED EXAMPLE"}}
    remaining, gain, ft = comparison["remaining_drop_pp"], comparison["compressed_to_fine_tuned_pp"], comparison["fine_tuned"]
    thresholds = {"max_drop_pp": max_drop_pp, "material_improvement_pp": material_improvement_pp, "unusable_below": unusable_below, "basis": "PROJECT-DEFINED EXAMPLE (smoke-diagnostic, not release)"}
    if remaining <= max_drop_pp:
        return {"decision": RECOVERY_SUCCESS, "reason": f"remaining drop {remaining} pp <= {max_drop_pp} pp on the smoke set", "thresholds": thresholds}
    if gain >= material_improvement_pp and ft >= unusable_below:
        return {"decision": PARTIAL_RECOVERY, "reason": f"gain {gain} pp (recovery {comparison.get('recovery_percent')} %) but remaining drop {remaining} pp > {max_drop_pp} pp", "thresholds": thresholds}
    return {"decision": RECOVERY_FAILED, "reason": f"gain {gain} pp, fine-tuned {ft}: near-zero or immaterial", "thresholds": thresholds}


def recommend_next_step(decision: str, comparison: Mapping[str, Any], *, structural_ok: bool = True) -> dict[str, Any]:
    """Exactly one of A-E. Never recommends spending credits after a weak recovery."""
    if decision == BLOCKED or not structural_ok:
        return {"code": "B", "label": NEXT_STEPS["B"], "rationale": ["recovery experiment incomplete; resolve locally (0 credits) before any credit decision"]}
    rec = comparison.get("recovery_percent")
    if decision == RECOVERY_SUCCESS:
        return {"code": "A", "label": NEXT_STEPS["A"], "rationale": [f"smoke recovery {rec} %; release accuracy still needs COCO val2017 on this fine-tuned candidate (0 NetsPresso credits)",
                                                                   "E5-3 convert/profile (75 credits) only after the val2017 result - not executed here"]}
    if decision == PARTIAL_RECOVERY:
        if rec is not None and rec >= 50.0:
            return {"code": "B", "label": NEXT_STEPS["B"], "rationale": [f"material recovery ({rec} %) with a short CPU run suggests more local epochs / data close the gap", "do not spend 75 credits on profiling yet"]}
        return {"code": "C", "label": NEXT_STEPS["C"], "rationale": [f"only {rec} % of the lost accuracy came back; ratio 0.5 pruning removed too much capacity for a short recovery",
                                                                   "a lower ratio costs another 25 credits and still requires fine-tuning - decide after a longer local run if time allows"]}
    return {"code": "E", "label": NEXT_STEPS["E"], "rationale": ["accuracy stays unusable after fine-tuning; do not spend credits on this path", "re-evaluate the compression strategy (ratio/method) before any further operation"]}


def validate_finetune_config(cfg: Mapping[str, Any]) -> list[str]:
    """Return missing/empty required keys (empty list = valid)."""
    return [k for k in FINETUNE_CONFIG_REQUIRED if cfg.get(k) in (None, "", [], {})]


def architecture_unchanged(*, params_before: int | None, params_after: int | None, macs_before: int | None, macs_after: int | None, tolerance_percent: float = 0.5) -> dict[str, Any]:
    """Fine-tuning must not change the architecture: parameter count and MAC estimate stay within tolerance."""
    def rel(a, b):
        return None if a in (None, 0) or b is None else abs(a - b) / a * 100.0
    p, m = rel(params_before, params_after), rel(macs_before, macs_after)
    ok = (p is not None and p <= tolerance_percent) and (m is None or m <= tolerance_percent)
    return {"status": "PASS" if ok else ("NOT_APPLICABLE" if p is None else "FAIL"), "params_rel_diff_percent": p, "macs_rel_diff_percent": m, "tolerance_percent": tolerance_percent,
            "params_equal": params_before == params_after}


# --------------------------------------------------------------------------- continuation (phase C) helpers
MILESTONES = {"M1": 0.35, "M2": 0.40, "M3": 0.43369}  # M3 = baseline 0.44369 - 1.0 pp example threshold
PHASE_C_CONTINUE, PHASE_C_PLATEAU, PHASE_C_STOP, PHASE_C_GO_VAL2017 = "PHASE_C_CONTINUE", "PHASE_C_PLATEAU", "PHASE_C_STOP", "PHASE_C_GO_VAL2017"
NEXT_STEP_LABELS = ("MORE LOCAL FINE-TUNING", "COCO VAL2017", "HOLD", "STOP", "REVIEW BEFORE NETSPRESSO")


def milestones_reached(value: float | None, milestones: Mapping[str, float] = MILESTONES) -> dict[str, bool]:
    return {k: (value is not None and value >= v) for k, v in milestones.items()}


def continuation_metrics(*, current: float | None, previous_best: float | None, baseline: float, compressed: float) -> dict[str, Any]:
    """delta_from_previous / delta_from_baseline / recovery_rate / remaining_gap for one validation point."""
    if current is None:
        return {"status": "N/A"}
    lost = baseline - compressed
    return {"status": "MEASURED", "current": current, "previous_best": previous_best,
            "delta_from_previous_best": None if previous_best is None else round(current - previous_best, 5),
            "delta_from_baseline": round(current - baseline, 5), "recovery_rate": round((current - compressed) / lost, 4) if lost > 0 else None,
            "remaining_gap": round(baseline - current, 5), "milestones": milestones_reached(current)}


def plateau_detected(validation_values: Sequence[float], *, min_improvement: float = 0.005, points: int = 3) -> dict[str, Any]:
    """True when the last ``points`` validation points each improved the running best by less than ``min_improvement``."""
    vals = [v for v in validation_values if v is not None]
    if len(vals) < points + 1:
        return {"plateau": False, "reason": f"fewer than {points + 1} validation points", "improvements": []}
    improvements = []
    best = vals[0]
    for v in vals[1:]:
        improvements.append(round(v - best, 5))
        best = max(best, v)
    recent = improvements[-points:]
    return {"plateau": all(i < min_improvement for i in recent), "recent_improvements": recent, "min_improvement": min_improvement, "points": points}


def phase_c_decision(*, best_value: float | None, previous_best: float, stopped_reason: str | None, plateau: bool, final_dropped_below_best: bool, structural_ok: bool,
                     m3: float = MILESTONES["M3"]) -> dict[str, Any]:
    """PHASE_C_GO_VAL2017 / PHASE_C_PLATEAU / PHASE_C_STOP / PHASE_C_CONTINUE (never a Release verdict)."""
    if not structural_ok or best_value is None or (stopped_reason and stopped_reason not in ("plateau", "completed")):
        return {"decision": PHASE_C_STOP, "case": "STOP", "reason": stopped_reason or "structure/evaluation incomplete"}
    if best_value >= m3:
        return {"decision": PHASE_C_GO_VAL2017, "case": "A", "reason": f"best {best_value} >= {m3} (baseline - 1.0 pp example threshold); release still needs val2017/checksum/reproducibility"}
    if plateau:
        return {"decision": PHASE_C_PLATEAU, "case": "D", "reason": "last validation points improved by < 0.005 each; low ROI for more epochs"}
    case = "B" if best_value >= MILESTONES["M2"] else "C" if best_value >= MILESTONES["M1"] else "below M1"
    note = "; final epoch below best -> best checkpoint kept (CASE E rollback)" if final_dropped_below_best else ""
    return {"decision": PHASE_C_CONTINUE, "case": case, "reason": f"best {best_value} < {m3}, still improving (case {case}){note}"}


def phase_c_next_step(decision: Mapping[str, Any], *, improvement_from_prev: float | None) -> dict[str, Any]:
    d = decision.get("decision")
    if d == PHASE_C_GO_VAL2017:
        return {"label": "COCO VAL2017", "netspresso": "HOLD", "rationale": ["smoke accuracy reached the example threshold band; the next evidence is release data (val2017, 0 credits), not NetsPresso"]}
    if d == PHASE_C_PLATEAU:
        return {"label": "REVIEW BEFORE NETSPRESSO", "netspresso": "HOLD", "rationale": ["accuracy plateaued below the threshold; review data/ratio/strategy before any credit-consuming step"]}
    if d == PHASE_C_STOP:
        return {"label": "STOP", "netspresso": "HOLD", "rationale": ["training/evaluation integrity problem; do not continue or spend credits until resolved"]}
    return {"label": "MORE LOCAL FINE-TUNING", "netspresso": "HOLD",
            "rationale": [f"continuation still gains accuracy (+{improvement_from_prev} over the previous phase best); more local epochs are cheap (0 credits)", "NetsPresso execution remains HOLD until the curve flattens or the threshold band is reached"]}


# --------------------------------------------------------------------------- unseen-validation (val2017) helpers
VAL_PASS_CANDIDATE, VAL_PARTIAL, VAL_REGRESSION, VAL_OVERFIT, VAL_INCONCLUSIVE = ("VAL2017_PASS_CANDIDATE", "VAL2017_PARTIAL_RECOVERY", "VAL2017_ACCURACY_REGRESSION",
                                                                                 "VAL2017_OVERFIT", "VAL2017_INCONCLUSIVE")


def generalization_gap(*, smoke: float | None, unseen: float | None) -> dict[str, Any]:
    """smoke (train==val COCO128) minus unseen (val2017) metric, in absolute units and percentage points."""
    if smoke is None or unseen is None:
        return {"status": "N/A", "smoke": smoke, "unseen": unseen}
    return {"status": "MEASURED", "smoke": smoke, "unseen": unseen, "gap": round(smoke - unseen, 5), "gap_pp": round((smoke - unseen) * 100.0, 3),
            "unseen_over_smoke_percent": round(unseen / smoke * 100.0, 2) if smoke else None}


def classify_unseen_recovery(*, baseline: float | None, compressed: float | None, candidate: float | None, baseline_smoke: float | None = None, candidate_smoke: float | None = None,
                             pass_pp: float = 1.0, near_pp: float = 3.0, material_pp: float = 5.0, overfit_gap_pp: float = 10.0) -> dict[str, Any]:
    """Classify the unseen-set result (CASE A-D) and map it to VAL2017_* decisions.

    * CASE A  drop <= pass_pp           -> VAL2017_PASS_CANDIDATE (accuracy criterion only; other release criteria separate)
    * CASE B  pass_pp < drop <= near_pp -> NEAR_PASS, still VAL2017_PARTIAL_RECOVERY (residual accuracy risk)
    * CASE C  drop > near_pp            -> VAL2017_ACCURACY_REGRESSION, or VAL2017_OVERFIT when the candidate's smoke-vs-unseen gap
                                           exceeds ``overfit_gap_pp`` by far more than the baseline's own gap
    * CASE D  recovered materially vs the compressed model but still far from baseline -> VAL2017_PARTIAL_RECOVERY
    Thresholds are PROJECT-DEFINED EXAMPLES.
    """
    thresholds = {"pass_pp": pass_pp, "near_pp": near_pp, "material_pp": material_pp, "overfit_gap_pp": overfit_gap_pp, "basis": "PROJECT-DEFINED EXAMPLE (unseen validation)"}
    if baseline is None or candidate is None:
        return {"case": None, "decision": VAL_INCONCLUSIVE, "reason": "baseline and/or candidate not measured on the unseen set", "thresholds": thresholds}
    drop_pp = round((baseline - candidate) * 100.0, 3)
    recovery_pp = None if compressed is None else round((candidate - compressed) * 100.0, 3)
    cand_gap = None if candidate_smoke is None else (candidate_smoke - candidate) * 100.0
    base_gap = None if baseline_smoke is None else (baseline_smoke - baseline) * 100.0
    excess_gap = None if cand_gap is None or base_gap is None else round(cand_gap - base_gap, 3)
    out = {"drop_pp": drop_pp, "recovery_pp": recovery_pp, "candidate_smoke_gap_pp": None if cand_gap is None else round(cand_gap, 3), "baseline_smoke_gap_pp": None if base_gap is None else round(base_gap, 3),
           "excess_gap_vs_baseline_pp": excess_gap, "thresholds": thresholds}
    if drop_pp <= pass_pp:
        return {**out, "case": "A", "decision": VAL_PASS_CANDIDATE, "reason": f"unseen drop {drop_pp} pp <= {pass_pp} pp: accuracy criterion PASS candidate (other release criteria evaluated separately)"}
    if drop_pp <= near_pp:
        return {**out, "case": "B", "decision": VAL_PARTIAL, "near_pass": True, "reason": f"unseen drop {drop_pp} pp in ({pass_pp}, {near_pp}] pp: NEAR_PASS, residual accuracy risk"}
    if excess_gap is not None and excess_gap > overfit_gap_pp:
        return {**out, "case": "C", "decision": VAL_OVERFIT, "reason": f"unseen drop {drop_pp} pp > {near_pp} pp and the candidate's smoke-vs-unseen gap exceeds the baseline's by {excess_gap} pp (> {overfit_gap_pp}): COCO128 overfitting"}
    if recovery_pp is not None and recovery_pp >= material_pp:
        return {**out, "case": "D", "decision": VAL_PARTIAL, "reason": f"recovered {recovery_pp} pp over the compressed model but unseen drop {drop_pp} pp > {near_pp} pp"}
    return {**out, "case": "C", "decision": VAL_REGRESSION, "reason": f"unseen drop {drop_pp} pp > {near_pp} pp"}


def unseen_next_step(decision: str) -> dict[str, Any]:
    """Exactly one of HOLD / REVIEW / E5-3 CONSIDER / LOCAL RECOVERY / STOP; NetsPresso stays HOLD in every case here."""
    table = {
        VAL_PASS_CANDIDATE: ("E5-3 CONSIDER", "accuracy held on unseen data; E5-3 (convert+profile, 75 credits) may be *considered* after checksum/reproducibility - separate approval, nothing executed"),
        VAL_PARTIAL: ("LOCAL RECOVERY", "residual accuracy gap on unseen data; continue local recovery (more/other training data, epochs) before any credit decision"),
        VAL_REGRESSION: ("REVIEW", "unseen accuracy regressed beyond the near-pass band; review training strategy/ratio before more work"),
        VAL_OVERFIT: ("HOLD", "smoke gains did not transfer to unseen data (COCO128 memorization); hold all credit decisions, change the training data strategy"),
        VAL_INCONCLUSIVE: ("STOP", "evaluation incomplete; fix the evaluation before drawing conclusions"),
    }
    label, why = table.get(decision, ("STOP", "unknown decision"))
    return {"label": label, "netspresso_additional_execution": "HOLD", "rationale": why}


def epoch_trend(history: Sequence[Mapping[str, Any]], key: str = "map50_95") -> dict[str, Any]:
    """Summarise per-epoch validation history: best epoch/value, final value, monotonic-ish convergence hint."""
    vals = [(h.get("epoch"), h.get(key)) for h in history if h.get(key) is not None]
    if not vals:
        return {"status": "N/A", "epochs_recorded": 0}
    best = max(vals, key=lambda t: t[1])
    return {"status": "MEASURED", "epochs_recorded": len(vals), "best_epoch": best[0], "best_value": best[1], "final_epoch": vals[-1][0], "final_value": vals[-1][1],
            "still_improving_at_end": len(vals) >= 2 and vals[-1][1] > vals[-2][1]}

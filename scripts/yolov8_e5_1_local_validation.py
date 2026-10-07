#!/usr/bin/env python
"""Phase 5-E E5-1: ZERO-CREDIT local validation of the NetsPresso-compressed YOLOv8n fx body.

Runs in the upstream environment (``.venv-yolo``: ultralytics 8.4.173, torch 2.14, onnxruntime):

    env -u NETSPRESSO_API_KEY .venv-yolo/Scripts/python scripts/yolov8_e5_1_local_validation.py

Inputs: the recorded real run (reports/real_runs/<stamp>_automatic_compression) and the E5-1 execution record
(reports/yolov8_e5_1/<stamp>/e5_1_execution.json). Outputs e5_1_result.json + e5_1_summary.html next to it.

Steps (all local):
    1. candidate artifact validation (pt: trusted load, GraphModule, forward; onnx: checker/graph)       §7
    2. params / FLOPs: SDK-reported vs independent (torch numel, ONNX MAC estimate, 2xMACs convention)     §8-9
    3. detection head re-attach with the R1 decode head (same code path as R1)                             §10
    4. COCO128 smoke accuracy with the SAME conditions as the baseline (rect=True, conf 0.001, IoU 0.7)    §11
    5. output equivalence proxy (ORT identical input + real images) - labelled proxy, never accuracy       §12
    6. local ORT latency / memory, process-isolated, baseline (R1 body+head) vs candidate (compressed+head)
    7. Quality Gate profiles compression / local_eval / release, defect classification                     §13-14
    8. registry candidate (not promoted), traceability chain, recommendation                               §15-21

No netspresso SDK import, no API key access, no network. Thresholds are PROJECT-DEFINED EXAMPLES.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from framework.baselines import BaselineEntry, BaselineRegistry, make_artifact_id  # noqa: E402
from framework.config import load_config  # noqa: E402
from framework.defects import classify  # noqa: E402
from framework.evaluation.artifact_structure import validate_onnx, validate_pt  # noqa: E402
from framework.evaluation.detection_accuracy import DetectionMetrics, compare_detection_accuracy  # noqa: E402
from framework.evaluation.detection_head import (  # noqa: E402
    expected_body_output_shapes,
    expected_decoded_output_shape,
    load_head_meta,
    validate_output_schema,
)
from framework.evaluation.environment import environment_fingerprint  # noqa: E402
from framework.evaluation.input_gate import build_traceability_chain  # noqa: E402
from framework.evaluation.local_ort import (  # noqa: E402
    PROXY_DISCLAIMER,
    compare_outputs,
    measure_in_subprocess,
)
from framework.evaluation.model_stats import (  # noqa: E402
    count_onnx_parameters,
    count_torch_parameters,
    estimate_onnx_macs,
    verify,
    verify_flops,
)
from framework.pipeline.result import (  # noqa: E402
    CaseResult,
    Configuration,
    CreditRecord,
    CreditUsageType,
    Environment,
    ExecutionResult,
    ExecutionStatus,
    Metrics,
    Reproducibility,
    ReproducibilityLevel,
    SupportState,
    utc_now_iso,
)
from framework.pipeline.runner import case_status_for  # noqa: E402
from framework.quality_gate import QualityGate, render_text  # noqa: E402
from framework.validation.artifact import artifact_from_file, compute_sha256  # noqa: E402

LEDGER = ROOT / "reports" / "credit_usage.json"
DATASETS = ROOT / "datasets"
R1_DIR = ROOT / "outputs" / "models" / "yolov8n_fx_r1"
DISCLAIMER = ("E5-1 local validation of the NetsPresso-compressed artifact. No NetsPresso API call was made in this step. "
              "Local measurements are development-machine evidence, not NetsPresso server-side measurements.")
BASELINE_FULL_MODEL = {"params": 3157200, "flops_impl_g": 8.855, "source": "reports/yolov8_baseline/20261005T024950Z/baseline.json (ultralytics get_num_params/get_flops, full model incl. DFL)"}
REG_MAX = 16


def portable(text: str) -> str:
    root_fwd = ROOT.as_posix() + "/"
    root_back = str(ROOT) + "\\"
    return text.replace(root_back.replace("\\", "\\\\"), "").replace(root_back, "").replace(root_fwd, "")


def dump(path: Path, data: object) -> None:
    path.write_text(portable(json.dumps(data, indent=2, ensure_ascii=False, default=str)) + "\n", encoding="utf-8")


def latest(pattern: str, must_have: str) -> Path | None:
    runs = sorted(p for p in ROOT.glob(pattern) if (p / must_have).is_file())
    return runs[-1] if runs else None


def metrics_from_record(d: dict) -> DetectionMetrics:
    m = d.get("metrics") or {}
    return DetectionMetrics(status=d["status"], reason=d.get("reason"), map50_95=m.get("mAP50-95"), map50=m.get("mAP50"), map75=m.get("mAP75"), precision=m.get("precision"),
                            recall=m.get("recall"), dataset=d.get("dataset"), dataset_version=d.get("dataset_version"), split=d.get("split"), images=d.get("images"),
                            instances=d.get("instances"), imgsz=d.get("imgsz"), conf=d.get("conf"), iou=d.get("iou"), evaluator=d.get("evaluator"), model_label=d.get("model_label"))


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=None, help="reports/real_runs/<stamp>_automatic_compression (default: latest yolov8n run)")
    ap.add_argument("--e5-dir", type=Path, default=None, help="reports/yolov8_e5_1/<stamp> containing e5_1_execution.json (default: latest)")
    ap.add_argument("--r1-record", type=Path, default=None, help="R1 result record providing the upstream baseline (default: latest *_r1/r1_result.json)")
    ap.add_argument("--weights", type=Path, default=ROOT / "outputs" / "models" / "yolov8n.pt", help="only used for class names")
    ap.add_argument("--data", default="coco128.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--images", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--allow-untrusted-candidate", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args(argv)

    ledger_before = hashlib.sha256(LEDGER.read_bytes()).hexdigest()
    config = load_config(ROOT / "configs")
    policy = config.local_eval
    e5_dir = args.e5_dir or latest("reports/yolov8_e5_1/2*Z", "e5_1_execution.json")
    if e5_dir is None:
        print("ABORT: no E5-1 execution record found")
        return 2
    e5_exec = json.loads((e5_dir / "e5_1_execution.json").read_text(encoding="utf-8"))
    run_dir = args.run_dir or (ROOT / e5_exec["real_execution"]["run_dir"])
    r1_record_path = args.r1_record or latest("reports/yolov8_baseline/2*Z_r1", "r1_result.json") / "r1_result.json"
    r1 = json.loads(r1_record_path.read_text(encoding="utf-8"))
    execution_prev = json.loads((run_dir / "execution_result.json").read_text(encoding="utf-8"))
    sdk_meta = json.loads((run_dir / "sdk_output" / "metadata.json").read_text(encoding="utf-8"))
    cand_pt, cand_onnx = run_dir / "sdk_output" / "sdk_output.pt", run_dir / "sdk_output" / "sdk_output.onnx"
    r1_pt, r1_meta_path = R1_DIR / "model_fx.pt", R1_DIR / "netspresso_head_meta.json"
    cfg = Configuration.from_dict(execution_prev["configuration"])
    input_shape = [1, 3, args.imgsz, args.imgsz]
    out = e5_dir
    print(DISCLAIMER)
    print(f"real run: {portable(run_dir.as_posix())} | e5 dir: {portable(e5_dir.as_posix())} | baseline record: {portable(r1_record_path.as_posix())}")

    import numpy as np  # noqa: PLC0415
    import onnx  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import ultralytics  # noqa: PLC0415
    from ultralytics import YOLO, settings  # noqa: PLC0415
    from ultralytics.data.augment import LetterBox  # noqa: PLC0415
    from ultralytics.models.yolo.detect import DetectionValidator  # noqa: PLC0415
    from ultralytics.nn.modules.block import DFL  # noqa: PLC0415
    from ultralytics.utils.tal import dist2bbox, make_anchors  # noqa: PLC0415
    from yolov8_r1_traceable_export import make_wrapper_classes, preprocess_image, stats_of  # noqa: PLC0415

    settings.update({"datasets_dir": DATASETS.as_posix(), "sync": False, "runs_dir": (out / "ultralytics_runs").as_posix()})
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    _body_cls, head_cls, detector_cls = make_wrapper_classes(torch)
    evaluator = f"ultralytics {ultralytics.__version__} DetectionValidator (torch {torch.__version__})"

    # ---------------- 1. candidate artifact validation ----------------
    cand_sha = compute_sha256(cand_pt)
    chain_ok = cand_sha == execution_prev["artifact"]["checksum_sha256"]
    r1_sha = compute_sha256(r1_pt)
    trusted = policy.is_trusted(cand_sha) or args.allow_untrusted_candidate
    sv_pt = validate_pt(cand_pt, allow_full_unpickle=trusted, torch_module=torch).to_dict()
    sv_onnx = validate_onnx(cand_onnx, onnx_module=onnx).to_dict()
    head_meta = load_head_meta(r1_meta_path)
    expected_body = expected_body_output_shapes(input_shape, head_meta)
    candidate = {"pt": {"path": portable(cand_pt.as_posix()), "sha256": cand_sha, "size_bytes": cand_pt.stat().st_size, "format": "pt (torch.save GraphModule)", "trusted_for_full_unpickle": trusted,
                        "trust_basis": "trust list (configs/local_eval.yaml)" if policy.is_trusted(cand_sha) else ("--allow-untrusted-candidate" if trusted else "none"), "structural": sv_pt},
                 "onnx": {"path": portable(cand_onnx.as_posix()), "sha256": compute_sha256(cand_onnx), "size_bytes": cand_onnx.stat().st_size, "format": "onnx (SDK companion export)", "structural": sv_onnx,
                          "graph_outputs": sv_onnx.get("details", {}).get("outputs")},
                 "sha_matches_execution_record": chain_ok, "sdk_model_ids": {"uploaded": execution_prev["artifact"]["metadata"].get("sdk_model_id"), "compressed": sdk_meta.get("results", {}).get("compressed_model", {}).get("model_id")}}
    body = None
    if sv_pt["status"] == "PASS" and sv_pt["details"].get("object_kind") == "graph_module":
        body = torch.load(cand_pt.as_posix(), map_location="cpu", weights_only=False).eval()
        outs = body(torch.zeros(*input_shape))
        shapes = [list(o.shape) for o in outs]
        ok, msg = validate_output_schema(shapes, expected_body)
        candidate["pt"].update({"forward_output_shapes": shapes, "expected_output_shapes": expected_body, "schema_ok": ok, "schema_message": msg, "parameter_count_torch": count_torch_parameters(body)})
    artifact_status = "PASS" if (sv_pt["status"] == "PASS" and sv_onnx["status"] == "PASS" and candidate["pt"].get("schema_ok") and chain_ok) else "FAIL"
    print(f"[artifact] pt {sv_pt['status']} kind={sv_pt['details'].get('object_kind')} shapes={candidate['pt'].get('forward_output_shapes')} schema_ok={candidate['pt'].get('schema_ok')} | onnx {sv_onnx['status']} | chain sha ok={chain_ok} -> {artifact_status}")

    # ---------------- 2. params / FLOPs ----------------
    sdk_orig, sdk_comp = sdk_meta["results"]["original_model"], sdk_meta["results"]["compressed_model"]
    r1_body = torch.load(r1_pt.as_posix(), map_location="cpu", weights_only=False).eval()
    r1_params = count_torch_parameters(r1_body)
    cand_params = candidate["pt"].get("parameter_count_torch")
    exports = ROOT / "outputs" / "e5_1_exports" / out.name  # model binaries stay under the gitignored outputs/ tree, never under reports/
    exports.mkdir(parents=True, exist_ok=True)

    def export_body_onnx(module, path: Path) -> dict:
        try:
            torch.onnx.export(module, torch.zeros(*input_shape), path.as_posix(), opset_version=13, input_names=["images"], do_constant_folding=True, dynamo=False)
            m = onnx.load(path.as_posix())
            macs, cov, uncov, err = estimate_onnx_macs(m, onnx)
            return {"status": "PASS", "path": portable(path.as_posix()), "sha256": compute_sha256(path), "params_onnx": count_onnx_parameters(m), "macs": macs, "covered_ops": cov, "uncovered_ops": uncov, "error": err}
        except Exception as exc:  # noqa: BLE001
            return {"status": "FAIL", "error": f"{type(exc).__name__}: {str(exc)[:300]}"}

    r1_body_onnx = export_body_onnx(r1_body, exports / "r1_body.onnx")
    cand_body_onnx = export_body_onnx(body, exports / "candidate_body.onnx") if body is not None else {"status": "NOT_APPLICABLE"}
    sdk_onnx_model = onnx.load(cand_onnx.as_posix())
    sdk_macs, sdk_cov, sdk_uncov, sdk_err = estimate_onnx_macs(sdk_onnx_model, onnx)
    pct = lambda a, b: round((a - b) / a * 100.0, 2) if a else None  # noqa: E731
    model_metrics = {
        "params": {"baseline_full_model_impl": BASELINE_FULL_MODEL["params"], "baseline_body_note": "fx body excludes the 16 constant DFL weights (3,157,200 - 16 = 3,157,184)",
                   "baseline_sdk": sdk_orig["number_of_parameters"], "baseline_independent": r1_params, "candidate_sdk": sdk_comp["number_of_parameters"], "candidate_independent": cand_params,
                   "absolute_delta_sdk": sdk_orig["number_of_parameters"] - sdk_comp["number_of_parameters"], "reduction_percent_sdk": pct(sdk_orig["number_of_parameters"], sdk_comp["number_of_parameters"]),
                   "verification": {"baseline": verify("params", sdk_orig["number_of_parameters"], r1_params, policy.params_tolerance_percent, "torch parameter numel of the R1 body (trusted full unpickle)").to_dict(),
                                    "candidate": verify("params", sdk_comp["number_of_parameters"], cand_params, policy.params_tolerance_percent, "torch parameter numel of the compressed body (trusted full unpickle)").to_dict()}},
        "flops": {"baseline_full_model_impl_g": BASELINE_FULL_MODEL["flops_impl_g"], "baseline_sdk": sdk_orig["flops"], "candidate_sdk": sdk_comp["flops"],
                  "baseline_independent_2x_macs": 2 * r1_body_onnx["macs"] if r1_body_onnx.get("macs") else None, "candidate_independent_2x_macs": 2 * cand_body_onnx["macs"] if cand_body_onnx.get("macs") else None,
                  "candidate_sdk_onnx_2x_macs": 2 * sdk_macs if sdk_macs else None, "absolute_delta_sdk": sdk_orig["flops"] - sdk_comp["flops"], "reduction_percent_sdk": pct(sdk_orig["flops"], sdk_comp["flops"]),
                  "verification": {"baseline": verify_flops(sdk_orig["flops"], r1_body_onnx.get("macs"), policy.flops_tolerance_percent, "ONNX MAC estimate of the locally exported R1 body (Conv/Gemm/MatMul)").to_dict(),
                                   "candidate": verify_flops(sdk_comp["flops"], cand_body_onnx.get("macs"), policy.flops_tolerance_percent, "ONNX MAC estimate of the locally exported compressed body (Conv/Gemm/MatMul)").to_dict(),
                                   "candidate_sdk_onnx": verify_flops(sdk_comp["flops"], sdk_macs, policy.flops_tolerance_percent, "ONNX MAC estimate of the SDK companion sdk_output.onnx").to_dict()},
                  "coverage": {"r1_body": {k: r1_body_onnx.get(k) for k in ("covered_ops", "uncovered_ops", "error")}, "candidate_body": {k: cand_body_onnx.get(k) for k in ("covered_ops", "uncovered_ops", "error")},
                               "sdk_onnx": {"covered_ops": sdk_cov, "uncovered_ops": sdk_uncov, "error": sdk_err}},
                  "convention_note": "SDK 'flops' compared as 2 x MACs (observed convention in Phase 5-B/5-D and here; not claimed as an official documented NetsPresso definition)."},
        "onnx_exports": {"r1_body": r1_body_onnx, "candidate_body": cand_body_onnx},
    }
    print(f"[params] sdk {sdk_orig['number_of_parameters']:.0f} -> {sdk_comp['number_of_parameters']:.0f} ({model_metrics['params']['reduction_percent_sdk']} %) | independent {r1_params} -> {cand_params} | verify {model_metrics['params']['verification']['candidate']['verification_status']}")
    print(f"[flops] sdk {sdk_orig['flops']:.3e} -> {sdk_comp['flops']:.3e} ({model_metrics['flops']['reduction_percent_sdk']} %) | 2xMACs {model_metrics['flops']['baseline_independent_2x_macs']} -> {model_metrics['flops']['candidate_independent_2x_macs']} | verify {model_metrics['flops']['verification']['candidate']['verification_status']}")

    # ---------------- 3. head re-attach ----------------
    meta_raw = json.loads(r1_meta_path.read_text(encoding="utf-8"))
    names = YOLO(args.weights.as_posix()).names if args.weights.is_file() else {i: str(i) for i in range(meta_raw["nc"])}
    reattach: dict = {"status": "FAIL", "head_class": "R1DecodeHead (rebuilt from R1 head meta; upstream DFL/make_anchors/dist2bbox)", "expected": expected_decoded_output_shape(input_shape, head_meta)}
    detector = ref_detector = None
    if body is not None:
        try:
            detector = detector_cls(body, head_cls(meta_raw, DFL, make_anchors, dist2bbox).eval(), names, meta_raw["stride"]).eval()
            ref_detector = detector_cls(r1_body, head_cls(meta_raw, DFL, make_anchors, dist2bbox).eval(), names, meta_raw["stride"]).eval()
            y0, feats0 = detector(torch.zeros(*input_shape))
            ok, msg = validate_output_schema([list(y0.shape)], [reattach["expected"]])
            reattach.update({"status": "PASS" if ok else "FAIL", "decoded_output_shape": list(y0.shape), "body_output_shapes": [list(f.shape) for f in feats0], "message": msg,
                             "head_meta_sha256": compute_sha256(r1_meta_path), "head_meta": head_meta.to_dict()})
        except Exception as exc:  # noqa: BLE001
            reattach.update({"error": f"{type(exc).__name__}: {str(exc)[:300]}"})
            detector = None
    print(f"[reattach] {reattach['status']} decoded {reattach.get('decoded_output_shape')} expected {reattach['expected']}")

    # ---------------- 4. COCO128 smoke accuracy ----------------
    baseline_rec = r1["detection_evaluation"]["baseline"]
    baseline = metrics_from_record(baseline_rec)
    val_kwargs = dict(data=args.data, imgsz=args.imgsz, batch=8, device="cpu", conf=args.conf, iou=args.iou, plots=False, verbose=False, workers=0, rect=True, mode="val", task="detect",
                      half=False, project=(out / "ultralytics_runs").as_posix())
    candidate_acc = DetectionMetrics.not_available("head re-attach failed" if detector is None else "evaluation skipped (--skip-eval)", dataset=args.data, imgsz=args.imgsz, evaluator=evaluator, model_label="E5-1 compressed body + rebuilt head")
    if detector is not None and not args.skip_eval:
        try:
            validator = DetectionValidator(args={**val_kwargs, "model": "e5_1_compressed_body_plus_head", "name": "candidate"})
            validator(model=deepcopy(detector))
            box = validator.metrics.box
            candidate_acc = DetectionMetrics(status="MEASURED", map50_95=round(float(box.map), 5), map50=round(float(box.map50), 5), map75=round(float(box.map75), 5), precision=round(float(box.mp), 5),
                                             recall=round(float(box.mr), 5), dataset=args.data, dataset_version=baseline.dataset_version, split=baseline.split, images=128, instances=929, imgsz=args.imgsz,
                                             conf=args.conf, iou=args.iou, evaluator=evaluator, model_label="E5-1 NetsPresso-compressed body (PR_L2 ratio 0.5, no fine-tuning) + rebuilt head",
                                             extra={"speed_ms": {k: round(float(v), 3) for k, v in (getattr(validator.metrics, "speed", {}) or {}).items()}})
            print(f"[val candidate] mAP50-95={candidate_acc.map50_95} mAP50={candidate_acc.map50} mAP75={candidate_acc.map75} P={candidate_acc.precision} R={candidate_acc.recall}")
        except Exception as exc:  # noqa: BLE001
            candidate_acc = DetectionMetrics.not_available(f"validation failed: {type(exc).__name__}: {str(exc)[:300]}", dataset=args.data, imgsz=args.imgsz, evaluator=evaluator, model_label="E5-1 compressed body + rebuilt head")
            print(f"[val candidate] N/A {candidate_acc.reason}")
    comparison = compare_detection_accuracy(baseline, candidate_acc)
    gate_acc = config.quality_gate.profiles["release"]["accuracy"]
    accuracy = {"baseline": baseline.to_dict(), "baseline_source": portable(r1_record_path.as_posix()), "candidate": candidate_acc.to_dict(), "comparison": comparison.to_dict(),
                "threshold_pp": gate_acc.threshold, "threshold_basis": "PROJECT-DEFINED EXAMPLE THRESHOLD (configs/quality_gate.yaml release.accuracy); diagnostic use on COCO128 smoke only, not a release decision",
                "conditions": {"data": args.data, "imgsz": args.imgsz, "conf": args.conf, "iou": args.iou, "rect": True, "batch": 8, "device": "cpu", "half": False, "evaluator": evaluator}}
    print(f"[accuracy] {comparison.status}: baseline {comparison.baseline_metric} vs candidate {comparison.candidate_metric} drop_pp={comparison.drop_pp} rel={comparison.relative_delta_percent} %")

    # ---------------- 5/6. ONNX of body+head for ORT latency/memory + proxy ----------------
    class OnlyY(torch.nn.Module):
        def __init__(self, det):
            super().__init__()
            self.det = det

        def forward(self, x):
            return self.det(x)[0]

    def export_full(det, path: Path) -> dict:
        try:
            torch.onnx.export(OnlyY(det).eval(), torch.zeros(*input_shape), path.as_posix(), opset_version=13, input_names=["images"], output_names=["output0"], do_constant_folding=True, dynamo=False)
            return {"status": "PASS", "path": portable(path.as_posix()), "sha256": compute_sha256(path), "structural": validate_onnx(path, onnx_module=onnx).to_dict()}
        except Exception as exc:  # noqa: BLE001
            return {"status": "FAIL", "error": f"{type(exc).__name__}: {str(exc)[:300]}"}

    perf: dict = {"baseline": None, "candidate": None, "note": "local onnxruntime CPU, process-isolated per model (fresh interpreter), warm-up/iterations from configs/local_eval.yaml; development machine, not the nominal target device"}
    proxy_ort = {"status": "NOT_APPLICABLE", "reason": "body+head ONNX export unavailable"}
    base_run = cand_run = None
    if ref_detector is not None and detector is not None:
        base_full, cand_full = export_full(ref_detector, exports / "r1_body_plus_head.onnx"), export_full(detector, exports / "candidate_body_plus_head.onnx")
        perf["onnx_exports"] = {"baseline": base_full, "candidate": cand_full}
        if base_full["status"] == "PASS" and cand_full["status"] == "PASS":
            measure = dict(warmup=policy.warmup_iterations, iterations=args.iterations or policy.measurement_iterations, seed=policy.seed, threads=policy.intra_op_threads, provider=policy.execution_provider, input_shape=input_shape)
            base_run = measure_in_subprocess(exports / "r1_body_plus_head.onnx", **measure)
            cand_run = measure_in_subprocess(exports / "candidate_body_plus_head.onnx", **measure)
            for label, run, exp in (("baseline", base_run, base_full), ("candidate", cand_run, cand_full)):
                perf[label] = {"latency": run.latency.to_dict(), "memory": run.memory.to_dict(), "output_shapes": run.output_shapes, "onnx": exp["path"]}
                print(f"[ort {label}] median {run.latency.median_ms} ms p95 {run.latency.p95_ms} ms | RSS delta {run.memory.delta_mb} MB | outputs {run.output_shapes}")
            pr = compare_outputs(base_run.outputs, cand_run.outputs, baseline_shapes=base_run.output_shapes, optimized_shapes=cand_run.output_shapes, baseline_dtypes=base_run.output_dtypes,
                                 optimized_dtypes=cand_run.output_dtypes, min_cosine=policy.proxy_min_cosine_similarity)
            proxy_ort = {**pr.to_dict(), "num_outputs": len(base_run.outputs)}
            print(f"[proxy ort] {pr.status}: min cosine {pr.min_cosine_similarity} max|d| {pr.max_abs_diff} rel {pr.relative_diff}")
    # real-image proxy: decoded pre-NMS, R1 body+head vs compressed body+head
    real_images = {"images": 0, "per_image": [], "aggregate": {}}
    if ref_detector is not None and detector is not None:
        img_dir = DATASETS / "coco128" / "images" / "train2017"
        imgs = sorted(img_dir.glob("*.jpg"))[: args.images]
        rep = img_dir / "000000000009.jpg"
        if rep.is_file() and rep not in imgs:
            imgs.append(rep)
        for p in imgs:
            x, _ = preprocess_image(p, args.imgsz, LetterBox, np, torch)
            y_ref, f_ref = ref_detector(x)
            y_c, f_c = detector(x)
            real_images["per_image"].append({"image": p.name, "decoded": stats_of(y_ref, y_c), "decoded_boxes": stats_of(y_ref[:, :4], y_c[:, :4]), "decoded_scores": stats_of(y_ref[:, 4:], y_c[:, 4:]),
                                             "body_maps": [stats_of(a, b) for a, b in zip(f_ref, f_c, strict=True)]})
        real_images["images"] = len(imgs)
        for k in ("decoded", "decoded_boxes", "decoded_scores"):
            real_images["aggregate"][k] = {"min_cosine": min(r[k]["cosine"] for r in real_images["per_image"]), "max_abs_diff": max(r[k]["max_abs_diff"] for r in real_images["per_image"]),
                                           "max_relative_l2_diff": max(r[k]["relative_l2_diff"] for r in real_images["per_image"])}
        print(f"[proxy real images] decoded min cosine {real_images['aggregate']['decoded']['min_cosine']:.6f} rel {real_images['aggregate']['decoded']['max_relative_l2_diff']:.3e}")
    proxy = {"ort": proxy_ort, "real_images": real_images, "disclaimer": PROXY_DISCLAIMER + " The proxy never replaces the measured detection accuracy above."}

    # ---------------- 7. Result Model + gates + defects ----------------
    env_fp = environment_fingerprint(execution_provider=policy.execution_provider, threads=policy.intra_op_threads, packages=("ultralytics", "torch", "onnx", "onnxruntime", "numpy", "psutil"))
    environment = Environment(adapter="local_e5_1_validation", adapter_version="0.1.0", sdk_name="onnxruntime+ultralytics", sdk_version=env_fp["packages"].get("onnxruntime"), python_version=env_fp["python"],
                              platform=f"{env_fp['os']} {env_fp['os_release']} {env_fp['architecture']}", extra={"credit_consuming": False, "netspresso_api_calls": 0, "evaluation_type": "local_e5_1", **env_fp})

    def metrics_for(acc: DetectionMetrics, run, size_path: Path, extra: dict) -> Metrics:
        return Metrics(accuracy=acc.map50_95 if acc.status == "MEASURED" else None, latency_ms=run.latency.median_ms if run else None,
                       memory_mb=run.memory.delta_mb if run and run.memory.status == "PASS" else None, model_size_mb=round(size_path.stat().st_size / 1_000_000, 3), extra=extra)

    cand_extra = {"sdk_compressed_number_of_parameters": float(sdk_comp["number_of_parameters"]), "sdk_compressed_flops": float(sdk_comp["flops"]), "params_torch": float(cand_params or 0)}
    if cand_body_onnx.get("macs"):
        cand_extra["macs_estimate"] = float(cand_body_onnx["macs"])
    if cand_run is not None:
        cand_extra.update({"latency_p95_ms": cand_run.latency.p95_ms, "latency_min_ms": cand_run.latency.min_ms, "latency_max_ms": cand_run.latency.max_ms})
    if proxy_ort.get("status") not in (None, "NOT_APPLICABLE"):
        cand_extra.update({"proxy_min_cosine_similarity": proxy_ort.get("min_cosine_similarity") or 0.0, "proxy_max_abs_diff": proxy_ort.get("max_abs_diff") or 0.0,
                           "proxy_relative_diff": proxy_ort.get("relative_diff") or 0.0, "proxy_shape_match": 1.0 if proxy_ort.get("shape_match") else 0.0, "proxy_dtype_match": 1.0 if proxy_ort.get("dtype_match") else 0.0})
    base_extra = {"sdk_original_number_of_parameters": float(sdk_orig["number_of_parameters"]), "sdk_original_flops": float(sdk_orig["flops"]), "params_torch": float(r1_params)}
    if r1_body_onnx.get("macs"):
        base_extra["macs_estimate"] = float(r1_body_onnx["macs"])
    if base_run is not None:
        base_extra.update({"latency_p95_ms": base_run.latency.p95_ms, "latency_min_ms": base_run.latency.min_ms, "latency_max_ms": base_run.latency.max_ms})
    artifact = artifact_from_file(cand_pt, expected_sha256=None, source_model="yolov8n_fx_r1/model_fx.pt", source_model_sha256=r1_sha, operation=cfg.optimization,
                                  sdk_version=execution_prev["environment"].get("sdk_version"), sdk_model_id=candidate["sdk_model_ids"]["compressed"], structural_validation=sv_pt, companion_onnx=sv_onnx,
                                  configuration_key=cfg.key, evaluation="local_e5_1", generated_by="scripts/yolov8_e5_1_local_validation.py")
    artifact.path = portable(artifact.path or "")
    execution = ExecutionResult(
        operation="e5_1_local_validation", configuration=cfg, execution_status=ExecutionStatus.COMPLETED, environment=environment,
        baseline_metrics=metrics_for(baseline, base_run, r1_pt, base_extra), metrics=metrics_for(candidate_acc, cand_run, cand_pt, cand_extra), artifact=artifact,
        logs=[DISCLAIMER, f"source real run: {run_dir.name}", f"input artifact: yolov8n_fx_r1/model_fx.pt sha256={r1_sha}"],
        reproducibility=Reproducibility(level=ReproducibilityLevel.NOT_VERIFIED, runs=1, checksums=[cand_sha], notes="single authorized real run (execution_runs=1); a second run was NOT performed by design"),
        timestamp=utc_now_iso(), credit=CreditRecord(usage_type=CreditUsageType.NONE, estimated=0, actual=None, note="local validation step: 0 credits, no API call"))
    gates = {name: QualityGate(config.quality_gate, profile=name).evaluate(execution) for name in ("compression", "local_eval", "release")}
    defects_by_profile = {}
    for name, g in gates.items():
        d = classify(execution, g)
        defects_by_profile[name] = None if d is None else {"defect_id": f"E5-1-{name}-{d.category.value}", **d.to_dict()}
        print("\n" + render_text(g))
    drop = comparison.drop_pp
    if comparison.status == "MEASURED" and drop is not None and drop > (gate_acc.threshold or 1.0):
        rca = {"suspected_root_cause": f"structured pruning (SDK method {sdk_meta['compression_info'].get('method')}, ratio {sdk_meta['compression_info'].get('ratio')}) applied WITHOUT the fine-tuning step the official "
                                       "ModelZoo-YOLOv8 workflow prescribes after compression; accuracy collapse before retraining is the expected behaviour of pruning",
               "confidence": "MEDIUM", "verified": False,
               "evidence": f"params -{model_metrics['params']['reduction_percent_sdk']} %, FLOPs -{model_metrics['flops']['reduction_percent_sdk']} %, mAP50-95 drop {drop} pp on COCO128 (smoke); output proxy cosine "
                           f"{real_images['aggregate'].get('decoded', {}).get('min_cosine')}. Verification would require fine-tuning the candidate and re-evaluating (not done in this phase; 0 credits but training time)."}
    elif comparison.status == "MEASURED":
        rca = {"suspected_root_cause": None, "confidence": "N/A", "verified": None, "evidence": f"accuracy within the example threshold (drop {drop} pp)"}
    else:
        rca = {"suspected_root_cause": None, "confidence": "N/A", "verified": None, "evidence": f"accuracy comparison {comparison.status}: {comparison.reason}"}
    case = CaseResult(configuration=cfg, status=case_status_for(execution, gates["release"]), support=SupportState.SUPPORTED, selected=True, selection_reason="Phase 5-E E5-1 single real experiment",
                      execution=execution, gate=gates["release"], defect=classify(execution, gates["release"]))

    # ---------------- 8. registry candidate + traceability + recommendation ----------------
    registry = BaselineRegistry(ROOT / "reports" / "baselines" / "registry.json")
    entry = registry.register_candidate(BaselineEntry(
        artifact_id=make_artifact_id(cfg.optimization, cand_sha), model_name=cfg.model, source_artifact=f"yolov8n_fx_r1/model_fx.pt sha256={r1_sha}", sha256=cand_sha, file_size=cand_pt.stat().st_size,
        format="pt", sdk_version=execution_prev["environment"].get("sdk_version"), operation=cfg.optimization, configuration=cfg.to_dict(), input_shape=input_shape,
        compression_ratio=sdk_meta["compression_info"]["ratio"], environment_fingerprint=json.loads(portable(json.dumps(execution_prev["environment"]))), created_at=execution_prev["timestamp"], status="candidate",
        provenance={"run_dir": portable(run_dir.as_posix()), "adapter": "netspresso", "sdk_model_id_uploaded": candidate["sdk_model_ids"]["uploaded"], "sdk_model_id_compressed": candidate["sdk_model_ids"]["compressed"],
                    "companion_onnx_sha256": candidate["onnx"]["sha256"], "e5_1_record": portable((out / "e5_1_result.json").as_posix()), "registered_by": "scripts/yolov8_e5_1_local_validation.py"},
        notes=["candidate only (execution_runs=1): promotion requires Level 3/4 reproducibility evidence from a second authorized run; never becomes the expected checksum from this single run"]))
    registry.save()

    e5_credit = e5_exec.get("credit", {})
    chain = build_traceability_chain(
        source_model=f"yolov8n.pt sha256={r1['weights']['sha256']} (ultralytics checkpoint {r1['checkpoint_metadata'].get('version')})",
        r1_fx_artifact=f"outputs/models/yolov8n_fx_r1/model_fx.pt (patch {r1['patch_id']}, {portable(r1_record_path.as_posix())})", r1_sha256=r1_sha,
        netspresso_execution=f"{run_dir.name}: automatic_compression ratio {sdk_meta['compression_info']['ratio']} sdk {execution_prev['environment'].get('sdk_version')} status {execution_prev['execution_status']} "
                             f"model_id {candidate['sdk_model_ids']['uploaded']} -> compressed {candidate['sdk_model_ids']['compressed']}",
        candidate_artifact=candidate["pt"]["path"], candidate_sha256=cand_sha, local_validation=portable((out / "e5_1_result.json").as_posix()),
        accuracy_result=f"COCO128 mAP50-95 baseline {comparison.baseline_metric} -> candidate {comparison.candidate_metric} (drop_pp {comparison.drop_pp}, {comparison.status})",
        quality_gate={k: g.overall.value for k, g in gates.items()},
        credit_record=f"reports/credit_usage.json entry #{e5_exec.get('ledger_after', {}).get('operations')}: account {e5_credit.get('account_total_before')} -> {e5_credit.get('account_total_after')} "
                      f"({e5_credit.get('delta_classification', {}).get('status')})")
    statuses = {"execution_status": execution_prev["execution_status"], "artifact_status": artifact_status, "validation_status": "PASS" if (artifact_status == "PASS" and reattach["status"] == "PASS" and candidate_acc.status == "MEASURED") else "FAIL",
                "quality_gate_status": {k: g.overall.value for k, g in gates.items()}, "release_status": gates["release"].overall.value}
    rec = recommend(statuses, comparison, gate_acc.threshold or 1.0, model_metrics, perf)
    print(f"\n[recommendation] {rec['decision']}")

    result = {
        "experiment": {"phase": "5-E", "experiment_id": "E5-1", "timestamp": datetime.now(UTC).replace(microsecond=0).isoformat(), "model": cfg.model, "operation": cfg.optimization,
                       "compression_ratio": sdk_meta["compression_info"]["ratio"], "compression_method": sdk_meta["compression_info"].get("method"), "input_shape": input_shape,
                       "run_dir": portable(run_dir.as_posix()), "e5_dir": portable(out.as_posix()), "configuration": cfg.to_dict()},
        "disclaimer": DISCLAIMER, "credit_consuming": False, "netspresso_api_calls": 0,
        "input_provenance": {"source_artifact": "outputs/models/yolov8n_fx_r1/model_fx.pt", "source_sha256": r1_sha, "source_size_bytes": r1_pt.stat().st_size,
                             "generation_method": f"scripts/yolov8_r1_traceable_export.py: upstream ultralytics {r1['export']['source_code_version']}, patch {r1['patch_id']}",
                             "r1_validation_reference": portable(r1_record_path.as_posix()), "source_model": {"weights": "outputs/models/yolov8n.pt", "sha256": r1["weights"]["sha256"], "checkpoint": r1["checkpoint_metadata"]},
                             "input_gate": e5_exec.get("input", {}).get("identity"), "precheck_status": e5_exec.get("input", {}).get("precheck_status")},
        "netspresso_execution": {"sdk_version": execution_prev["environment"].get("sdk_version"), "status": execution_prev["execution_status"], "method": "NetsPresso.compressor_v2().automatic_compression",
                                 "execution_time_s": execution_prev.get("execution_time_s"), "timestamp": execution_prev["timestamp"], "sdk_model_id": candidate["sdk_model_ids"]["uploaded"],
                                 "compressed_model_id": candidate["sdk_model_ids"]["compressed"], "server_metadata": {k: sdk_meta.get(k) for k in ("status", "task_type", "is_retrainable", "model_info", "compression_info")},
                                 "sdk_reported": {"original": sdk_orig, "compressed": sdk_comp}, "credit": e5_credit, "ledger_after": e5_exec.get("ledger_after"), "operations_executed": e5_credit.get("service_operations_executed"),
                                 "execution_record": portable((run_dir / "execution_result.json").as_posix()), "e5_1_execution_record": portable((out / "e5_1_execution.json").as_posix()),
                                 "execution_stage_gate": json.loads((run_dir / "stage_gate.json").read_text(encoding="utf-8")) if (run_dir / "stage_gate.json").is_file() else None},
        "candidate_artifact": candidate, "model_metrics": model_metrics, "head_reattach": reattach, "accuracy": accuracy, "output_equivalence_proxy": proxy, "performance": perf,
        "quality_gate": {k: g.to_dict() for k, g in gates.items()}, "defects": {"by_profile": defects_by_profile, "root_cause_assessment": rca},
        "reproducibility": {"execution_runs": 1, "level": execution.reproducibility.level.value, "registry_entry": entry.to_dict(), "note": "candidate registered, NOT promoted; no second real run was performed (by design)"},
        "traceability": chain, "statuses": statuses, "recommendation": rec, "case_result": case.to_dict(), "environment": environment.to_dict(),
    }
    result = json.loads(portable(json.dumps(result, default=str)))
    dump(out / "e5_1_result.json", result)
    try:
        from framework.reporter import write_e5_1_html  # noqa: PLC0415

        write_e5_1_html(result, out / "e5_1_summary.html")
    except Exception as exc:  # noqa: BLE001
        print(f"[report] HTML skipped: {type(exc).__name__}: {exc}")
    assert not any(m == "netspresso" or m.startswith("netspresso.") for m in sys.modules), "SDK must never be imported here"
    print(f"ledger unchanged: {hashlib.sha256(LEDGER.read_bytes()).hexdigest() == ledger_before} | NetsPresso API calls: 0 | record: {portable((out / 'e5_1_result.json').as_posix())}")
    return 0 if statuses["validation_status"] == "PASS" else 1


def recommend(statuses: dict, comparison, threshold_pp: float, model_metrics: dict, perf: dict) -> dict:
    rationale = []
    if statuses["execution_status"] != "COMPLETED":
        return {"decision": "INVESTIGATE", "rationale": ["NetsPresso operation did not complete; investigate with 0 credits before any further credit-consuming operation"]}
    if statuses["artifact_status"] != "PASS":
        return {"decision": "NO-GO", "rationale": ["candidate artifact invalid (structure/schema/chain); no further credits on this artifact"]}
    if comparison.status != "MEASURED":
        return {"decision": "INVESTIGATE", "rationale": [f"accuracy not measurable ({comparison.status}: {comparison.reason}); resolve locally first"]}
    drop = comparison.drop_pp
    rationale.append(f"COCO128 smoke mAP50-95 {comparison.baseline_metric} -> {comparison.candidate_metric} (drop {drop} pp vs example threshold {threshold_pp} pp)")
    rationale.append(f"SDK-reported reduction: params -{model_metrics['params']['reduction_percent_sdk']} %, FLOPs -{model_metrics['flops']['reduction_percent_sdk']} %")
    if perf.get("baseline") and perf.get("candidate"):
        rationale.append(f"local ORT median latency {perf['baseline']['latency']['median_ms']} -> {perf['candidate']['latency']['median_ms']} ms (development machine)")
    if drop <= threshold_pp:
        rationale.append("accuracy within the example threshold on smoke data; E5-2 (local full pipeline) and a decision on E5-3 (convert+profile, 75 credits) may be justified after release-data accuracy")
        return {"decision": "PROCEED", "rationale": rationale}
    rationale.append("substantial accuracy regression: the official YOLOv8 workflow fine-tunes after compression; conversion/profiling of an un-retrained candidate is premature")
    rationale.append("options: (a) fine-tune the candidate locally (0 credits, CPU time) and re-evaluate; (b) a lower compression ratio would cost another 25 credits and still needs fine-tuning; "
                     "(c) keep YOLOv8n - the model is not the problem, the missing retraining step is")
    return {"decision": "HOLD", "rationale": rationale}


if __name__ == "__main__":
    raise SystemExit(main())

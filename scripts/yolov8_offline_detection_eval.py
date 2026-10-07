#!/usr/bin/env python
"""OFFLINE detection evaluation path (head re-attach -> val -> mAP) with 0 NetsPresso credits.

Runs in the Nota fork environment (``.venv-yolo-nota``: ultralytics_nota 8.0.108, torch 2.0.1):

    .venv-yolo-nota/Scripts/python scripts/yolov8_offline_detection_eval.py
    .venv-yolo-nota/Scripts/python scripts/yolov8_offline_detection_eval.py --body <future compressed model_fx.pt> --label "netspresso compressed body"

Default input is the **identity fixture** = the un-compressed baseline ``model_fx.pt``
produced by ``export_netspresso()``. It exercises the exact path a future NetsPresso
compressed body will take:

    body (GraphModule) + netspresso_head_meta.json
      -> precheck (framework contract)             TC-Y8-CG-002/003/004/014
      -> DetectionModel_netspresso (fork re-attach) TC-Y8-CG-002
      -> decoded output schema check               TC-Y8-CG-004
      -> DetectionValidator on COCO128             TC-Y8-CG-005/006/009
      -> DetectionMetrics / AccuracyComparison     TC-Y8-CG-007/008
      -> accuracy criterion (release threshold)    gate semantics

Fixture results are NEVER reported as compressed-model results. No netspresso SDK import.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from framework.config import CriterionConfig, load_config  # noqa: E402
from framework.evaluation.detection_accuracy import DetectionMetrics, compare_detection_accuracy  # noqa: E402
from framework.evaluation.detection_head import (  # noqa: E402
    expected_decoded_output_shape,
    load_head_meta,
    precheck_fx_bundle,
    validate_output_schema,
)
from framework.evaluation.environment import environment_fingerprint  # noqa: E402
from framework.validation.accuracy import validate_accuracy  # noqa: E402
from framework.validation.artifact import compute_sha256  # noqa: E402

LEDGER = ROOT / "reports" / "credit_usage.json"
DATASETS = ROOT / "datasets"
DISCLAIMER = ("Offline detection evaluation path validated locally. No NetsPresso API call was made. "
              "Fixture results are NOT NetsPresso compression results.")


def portable(text: str) -> str:
    root_fwd = ROOT.as_posix() + "/"
    root_back = str(ROOT) + "\\"
    return text.replace(root_back.replace("\\", "\\\\"), "").replace(root_back, "").replace(root_fwd, "")


def metrics_from_validator(m, *, dataset: str, split: str, imgsz: int, conf: float, iou: float, evaluator: str, label: str, images: int | None, instances: int | None) -> DetectionMetrics:
    box = m.box
    return DetectionMetrics(
        status="MEASURED", map50_95=round(float(box.map), 5), map50=round(float(box.map50), 5), map75=round(float(box.map75), 5),
        precision=round(float(box.mp), 5), recall=round(float(box.mr), 5), dataset=dataset, dataset_version="ultralytics coco128.zip (COCO 2017 subset)" if "128" in dataset else None,
        split=split, images=images, instances=instances, imgsz=imgsz, conf=conf, iou=iou, evaluator=evaluator, model_label=label,
        extra={"speed_ms": {k: round(float(v), 3) for k, v in (getattr(m, "speed", {}) or {}).items()}},
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--body", type=Path, default=ROOT / "outputs" / "models" / "yolov8n_fx" / "model_fx.pt")
    ap.add_argument("--head-meta", type=Path, default=ROOT / "outputs" / "models" / "yolov8n_fx" / "netspresso_head_meta.json")
    ap.add_argument("--label", default="identity fixture (pre-compression baseline fx body)")
    ap.add_argument("--baseline-weights", type=Path, default=ROOT / "outputs" / "models" / "yolov8n.pt", help="evaluated with the SAME fork evaluator for a like-for-like baseline")
    ap.add_argument("--meta-config", default="yolov8n.yaml", help="ultralytics model yaml used by YOLO_netspresso for default args")
    ap.add_argument("--data", default="coco128.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--allow-untrusted-body", action="store_true", help="allow full unpickle of a body whose SHA is not in the trust list (locally generated artifacts only)")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--configs", type=Path, default=ROOT / "configs")
    args = ap.parse_args(argv)
    if not args.body.is_file() or not args.head_meta.is_file():
        print(f"ABORT: body/head-meta not found: {args.body} / {args.head_meta}")
        return 2

    ledger_before = hashlib.sha256(LEDGER.read_bytes()).hexdigest()
    config = load_config(args.configs)
    policy = config.local_eval
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or (ROOT / "reports" / "yolov8_baseline" / f"{stamp}_offline_eval")
    out.mkdir(parents=True, exist_ok=True)
    print(DISCLAIMER)

    import torch  # noqa: PLC0415
    import ultralytics  # noqa: PLC0415
    import ultralytics.yolo.data.utils as _data_utils  # noqa: PLC0415
    import ultralytics.yolo.utils as _utils  # noqa: PLC0415
    from ultralytics import YOLO, YOLO_netspresso  # noqa: PLC0415
    from ultralytics.yolo.utils import SETTINGS  # noqa: PLC0415

    # The fork resolves dataset paths from a module constant computed at import time; point it at the repo's
    # gitignored datasets/ so no second copy is downloaded outside the repository.
    SETTINGS["datasets_dir"] = DATASETS.as_posix()
    _utils.DATASETS_DIR = DATASETS
    _data_utils.DATASETS_DIR = DATASETS
    evaluator = f"ultralytics_nota {ultralytics.__version__} DetectionValidator (torch {torch.__version__})"
    body_sha = compute_sha256(args.body)
    trusted = policy.is_trusted(body_sha) or args.allow_untrusted_body
    input_shape = [1, 3, args.imgsz, args.imgsz]
    record: dict = {"credit_consuming": False, "netspresso_api_calls": 0, "disclaimer": DISCLAIMER, "fixture_label": args.label,
                    "body": {"path": portable(args.body.as_posix()), "sha256": body_sha, "size_bytes": args.body.stat().st_size, "full_unpickle_allowed": trusted,
                             "trust_basis": "trust list" if policy.is_trusted(body_sha) else ("--allow-untrusted-body (locally generated)" if trusted else "none")},
                    "head_meta": {"path": portable(args.head_meta.as_posix()), "sha256": compute_sha256(args.head_meta)}, "evaluator": evaluator}

    # ---- 1. framework precheck (format, GraphModule, head meta, input shape, body output schema) ----
    def body_forward(shape):
        gm = torch.load(args.body.as_posix(), map_location="cpu")
        gm.train()
        with torch.no_grad():
            outs = gm(torch.zeros(*shape))
        return [list(o.shape) for o in (outs if isinstance(outs, (list, tuple)) else [outs])]

    pre = precheck_fx_bundle(args.body, args.head_meta, input_shape=input_shape, expected_nc=80, allow_full_unpickle=trusted, torch_module=torch, forward=body_forward)
    record["precheck"] = pre.to_dict()
    print(f"[precheck] {pre.status}: {pre.message}")
    if pre.status != "READY":
        _finish(out, record, ledger_before)
        return 3
    head = load_head_meta(args.head_meta)

    # ---- 2. re-attach head via the fork (DetectionModel_netspresso) ----
    try:
        model = YOLO_netspresso(compressed_model=args.body.as_posix(), head_meta=args.head_meta.as_posix(), task="detect_retraining", meta_config=args.meta_config)
        net = model.model.eval()
        with torch.no_grad():
            y = net(torch.zeros(*input_shape))
        decoded = y[0] if isinstance(y, (list, tuple)) else y
        ok, msg = validate_output_schema([list(decoded.shape)], [expected_decoded_output_shape(input_shape, head)])
        record["reattach"] = {"status": "PASS" if ok else "FAIL", "class": type(net).__name__, "decoded_output_shape": list(decoded.shape),
                              "expected": expected_decoded_output_shape(input_shape, head), "message": msg, "names_count": len(net.names), "stride": [float(s) for s in net.stride]}
        print(f"[reattach] {record['reattach']['status']}: {type(net).__name__} decoded output {list(decoded.shape)} ({msg})")
        if not ok:
            _finish(out, record, ledger_before)
            return 4
    except Exception as exc:  # noqa: BLE001
        record["reattach"] = {"status": "FAIL", "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        print(f"[reattach] FAIL {record['reattach']['error']}")
        _finish(out, record, ledger_before)
        return 4

    # ---- 3. COCO128 evaluation of the fixture ----
    val_kwargs = dict(data=args.data, imgsz=args.imgsz, batch=8, device="cpu", conf=args.conf, iou=args.iou, plots=False, verbose=False, workers=0,
                      project=(out / "ultralytics_runs").as_posix(), name="fixture")
    try:
        m_fix = model.val(**val_kwargs)
        n_img, n_inst = _dataset_counts(model)
        fixture_metrics = metrics_from_validator(m_fix, dataset=args.data, split="val(train2017 subset)" if "128" in args.data else "val", imgsz=args.imgsz,
                                                 conf=args.conf, iou=args.iou, evaluator=evaluator, label=args.label, images=n_img, instances=n_inst)
        print(f"[val fixture] mAP50-95={fixture_metrics.map50_95} mAP50={fixture_metrics.map50} P={fixture_metrics.precision} R={fixture_metrics.recall}")
    except Exception as exc:  # noqa: BLE001
        fixture_metrics = DetectionMetrics.not_available(f"validation failed: {type(exc).__name__}: {str(exc)[:200]}", dataset=args.data, imgsz=args.imgsz, evaluator=evaluator, model_label=args.label)
        print(f"[val fixture] N/A {fixture_metrics.reason}")

    # ---- 4. like-for-like baseline with the SAME evaluator ----
    baseline_metrics = DetectionMetrics.not_available("baseline evaluation skipped", dataset=args.data, imgsz=args.imgsz, evaluator=evaluator, model_label="yolov8n baseline (weights)")
    if not args.skip_baseline and args.baseline_weights.is_file():
        try:
            base = YOLO(args.baseline_weights.as_posix())
            m_base = base.val(**{**val_kwargs, "name": "baseline"})
            n_img, n_inst = _dataset_counts(base)
            baseline_metrics = metrics_from_validator(m_base, dataset=args.data, split="val(train2017 subset)" if "128" in args.data else "val", imgsz=args.imgsz,
                                                      conf=args.conf, iou=args.iou, evaluator=evaluator, label="yolov8n baseline (official weights, same evaluator)", images=n_img, instances=n_inst)
            print(f"[val baseline] mAP50-95={baseline_metrics.map50_95} mAP50={baseline_metrics.map50} P={baseline_metrics.precision} R={baseline_metrics.recall}")
        except Exception as exc:  # noqa: BLE001
            baseline_metrics = DetectionMetrics.not_available(f"validation failed: {type(exc).__name__}: {str(exc)[:200]}", dataset=args.data, imgsz=args.imgsz, evaluator=evaluator, model_label="yolov8n baseline (weights)")
            print(f"[val baseline] N/A {baseline_metrics.reason}")

    # ---- 5. comparison + accuracy criterion semantics ----
    comparison = compare_detection_accuracy(baseline_metrics, fixture_metrics)
    gate_cfg = config.quality_gate.profiles["release"].get("accuracy") or CriterionConfig("accuracy", 1.0, True)
    criterion = validate_accuracy(baseline_metrics.to_framework_metrics() if baseline_metrics.status == "MEASURED" else None,
                                  fixture_metrics.to_framework_metrics() if fixture_metrics.status == "MEASURED" else None, gate_cfg)
    na_demo = validate_accuracy(baseline_metrics.to_framework_metrics() if baseline_metrics.status == "MEASURED" else None, None, gate_cfg)
    # cross-evaluator reference: the upstream (ultralytics 8.4.x) baseline record, if present. Different evaluator ->
    # the framework must classify it NOT_COMPARABLE instead of producing a delta.
    cross_ref = None
    runs = sorted(p for p in (ROOT / "reports" / "yolov8_baseline").glob("2*Z") if (p / "baseline.json").is_file())
    if runs:
        bj = json.loads((runs[-1] / "baseline.json").read_text(encoding="utf-8"))
        acc = bj.get("accuracy", {})
        if acc.get("status") == "MEASURED":
            upstream = DetectionMetrics(status="MEASURED", map50_95=acc["metrics"]["mAP50-95"], map50=acc["metrics"]["mAP50"], map75=acc["metrics"].get("mAP75"),
                                        precision=acc["metrics"]["precision"], recall=acc["metrics"]["recall"], dataset=acc["dataset_yaml"], split="val(train2017 subset)",
                                        imgsz=acc["imgsz"], conf=args.conf, iou=args.iou,
                                        evaluator=f"ultralytics {bj['model']['implementation']['ultralytics_version']} (torch {bj['environment']['packages'].get('torch')})",
                                        model_label="yolov8n baseline (upstream evaluator, reports/yolov8_baseline)")
            cross_ref = {"upstream_baseline": upstream.to_dict(), "comparison_vs_fork_baseline": compare_detection_accuracy(upstream, baseline_metrics).to_dict(),
                         "note": "same weights, same images; evaluator differs -> NOT_COMPARABLE by policy. A large gap is an evaluator-drift finding, not a model result."}
    record.update({
        "cross_evaluator_reference": cross_ref,
        "fixture_accuracy": fixture_metrics.to_dict(),
        "baseline_accuracy_same_evaluator": baseline_metrics.to_dict(),
        "comparison": comparison.to_dict(),
        "accuracy_criterion_demo": {"threshold_pp": gate_cfg.threshold, "threshold_basis": "PROJECT-DEFINED EXAMPLE THRESHOLD (configs/quality_gate.yaml release.accuracy)",
                                    "fixture_vs_baseline": criterion.to_dict(), "missing_candidate": na_demo.to_dict(),
                                    "note": "identity fixture is expected to match its baseline; this validates the pipeline, not a compression result"},
        "evaluation_conditions": {"data": args.data, "imgsz": args.imgsz, "conf": args.conf, "iou": args.iou, "device": "cpu", "batch": 8, "rect": True, "half": False},
        "environment": environment_fingerprint(packages=("ultralytics", "torch", "torchvision", "numpy", "onnx", "psutil")),
    })
    print(f"[compare] {comparison.status}: baseline {comparison.baseline_metric} vs fixture {comparison.candidate_metric} drop_pp={comparison.drop_pp} | criterion {criterion.status.value}: {criterion.message}")
    _finish(out, record, ledger_before)
    return 0


def _dataset_counts(yolo_model) -> tuple[int | None, int | None]:
    try:
        stats = yolo_model.metrics  # DetMetrics
        return int(getattr(stats, "nt_per_class", []).sum()) if hasattr(stats, "nt_per_class") else None, None
    except Exception:  # noqa: BLE001
        return None, None


def _finish(out: Path, record: dict, ledger_before: str) -> None:
    assert not any(m == "netspresso" or m.startswith("netspresso.") for m in sys.modules), "SDK must never be imported here"
    record["timestamp"] = datetime.now(UTC).replace(microsecond=0).isoformat()
    record["ledger_unchanged"] = hashlib.sha256(LEDGER.read_bytes()).hexdigest() == ledger_before
    (out / "offline_detection_eval.json").write_text(portable(json.dumps(record, indent=2, ensure_ascii=False, default=str)) + "\n", encoding="utf-8")
    print(f"ledger unchanged: {record['ledger_unchanged']} | NetsPresso API calls: 0 | record: {portable((out / 'offline_detection_eval.json').as_posix())}")


if __name__ == "__main__":
    raise SystemExit(main())

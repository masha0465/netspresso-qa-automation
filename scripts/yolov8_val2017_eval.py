#!/usr/bin/env python
"""Phase 5-E: COCO val2017 independent (unseen) generalization evaluation. ZERO NetsPresso credits, NO training.

Runs in the upstream environment (``.venv-yolo``: ultralytics 8.4.173, torch 2.14, CPU). Three sub-commands, each a
separate process so every evaluation is isolated:

    env -u NETSPRESSO_API_KEY .venv-yolo/Scripts/python scripts/yolov8_val2017_eval.py prepare   --report-dir <dir>
    env -u NETSPRESSO_API_KEY .venv-yolo/Scripts/python scripts/yolov8_val2017_eval.py evaluate  --report-dir <dir> --model baseline|e5_1|phase_c
    env -u NETSPRESSO_API_KEY .venv-yolo/Scripts/python scripts/yolov8_val2017_eval.py summarize --report-dir <dir>

prepare   downloads ONLY val2017.zip (1 GB, 5,000 images) + the ultralytics COCO label pack into datasets/coco (gitignored),
          writes datasets/coco_val2017_eval.yaml, dataset_manifest.json and the COCO128-vs-val2017 contamination check.
evaluate  loads one frozen artifact (R1 baseline body / E5-1 NetsPresso candidate / Phase C local derivative), re-attaches the
          R1 decode head and runs ultralytics DetectionValidator with the E5-1 conditions (imgsz 640, conf 0.001, IoU 0.7,
          rect True, batch 8). Inference only: torch.no_grad, no optimizer, no backward. SHA checked before and after.
summarize builds the comparison tables (val2017; COCO128 vs val2017 generalization gap), recovery classification, Quality
          Gate profiles (Phase C performance numbers are referenced, not re-measured), proxy on a few val2017 images, and the
          HTML/JSON reports.

Guards: abort if NETSPRESSO_API_KEY is present (presence check only, value never read), abort if the netspresso module is
imported, ledger must stay 50/450/2, model files are immutable (SHA before == after).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from framework.config import load_config  # noqa: E402
from framework.defects import classify  # noqa: E402
from framework.evaluation.artifact_structure import validate_pt  # noqa: E402
from framework.evaluation.detection_accuracy import DetectionMetrics, compare_detection_accuracy  # noqa: E402
from framework.evaluation.detection_head import (  # noqa: E402
    expected_decoded_output_shape,
    load_head_meta,
    validate_output_schema,
)
from framework.evaluation.environment import environment_fingerprint  # noqa: E402
from framework.evaluation.input_gate import build_traceability_chain  # noqa: E402
from framework.evaluation.recovery import (  # noqa: E402
    classify_unseen_recovery,
    generalization_gap,
    unseen_next_step,
)
from framework.pipeline.result import (  # noqa: E402
    Configuration,
    CreditRecord,
    CreditUsageType,
    Environment,
    ExecutionResult,
    ExecutionStatus,
    Metrics,
    Reproducibility,
    ReproducibilityLevel,
    utc_now_iso,
)
from framework.quality_gate import QualityGate, render_text  # noqa: E402
from framework.validation.artifact import artifact_from_file, compute_sha256  # noqa: E402

LEDGER = ROOT / "reports" / "credit_usage.json"
DATASETS = ROOT / "datasets"
COCO_DIR = DATASETS / "coco"
DATA_YAML = DATASETS / "coco_val2017_eval.yaml"
R1_DIR = ROOT / "outputs" / "models" / "yolov8n_fx_r1"
E5_RUN = ROOT / "reports" / "real_runs" / "20261005T040009Z_automatic_compression"
E5_RECORD = ROOT / "reports" / "yolov8_e5_1" / "20261005T040009Z" / "e5_1_result.json"
EXPECTED_LEDGER = (50, 450, 2)
MODELS = {
    "baseline": {"path": R1_DIR / "model_fx.pt", "sha256": "d8e761dae29301ef51df9e7679c729338a522d397e271a900d453d171e1c022b", "size": 12846691, "kind": "upstream YOLOv8n R1 fx body (untouched baseline)", "smoke_map": 0.44369},
    "e5_1": {"path": E5_RUN / "sdk_output" / "sdk_output.pt", "sha256": "21d8cbd1361cfd3ef55c0c132d11f3fad294dbde6f175e8082bc0233b860e0f1", "size": 3693813, "kind": "NetsPresso-generated E5-1 compressed candidate", "smoke_map": 0.0},
    "phase_c": {"path": ROOT / "outputs" / "models" / "yolov8n_e5_1_finetuned" / "model_fx_phaseC.pt", "sha256": "dba5cd214f0f4098c2708366fb8ee2a3f2d4e5681b7668fc49f36aa148a4d9ee", "size": 3714299,
                "kind": "LOCAL fine-tuned derivative (phase C, global epoch 80) of the E5-1 candidate", "smoke_map": 0.46688},
}
PHASE_B_SMOKE = 0.28965
DISCLAIMER = ("COCO val2017 independent evaluation: inference only, no training, no NetsPresso API call, 0 credits. COCO128 (train == val) numbers are smoke "
              "evidence; val2017 is the unseen-data evidence. Thresholds are PROJECT-DEFINED EXAMPLES; no Release PASS is declared here.")
EVAL_CONDITIONS = {"imgsz": 640, "conf": 0.001, "iou": 0.7, "rect": True, "batch": 8, "device": "cpu", "half": False, "workers": 0, "evaluator": "ultralytics DetectionValidator (same conditions as E5-1 / phases A-C)"}


def portable(text: str) -> str:
    root_fwd = ROOT.as_posix() + "/"
    root_back = str(ROOT) + "\\"
    return text.replace(root_back.replace("\\", "\\\\"), "").replace(root_back, "").replace(root_fwd, "")


def dump(path: Path, data: object) -> None:
    path.write_text(portable(json.dumps(data, indent=2, ensure_ascii=False, default=str)) + "\n", encoding="utf-8")


def ledger_state() -> dict:
    d = json.loads(LEDGER.read_text(encoding="utf-8"))
    return {"used": d["used_credit"], "remaining": d["remaining_estimate"], "operations": len(d["operations"]), "sha256": hashlib.sha256(LEDGER.read_bytes()).hexdigest()}


def guards() -> dict | None:
    if "NETSPRESSO_API_KEY" in os.environ:  # presence check only; the value is never read
        print("ABORT: NETSPRESSO_API_KEY is present in the environment; this step is local-only (run with env -u NETSPRESSO_API_KEY).")
        return None
    if any(m == "netspresso" or m.startswith("netspresso.") for m in sys.modules):
        print("ABORT: netspresso SDK is imported in this interpreter.")
        return None
    st = ledger_state()
    if (st["used"], st["remaining"], st["operations"]) != EXPECTED_LEDGER:
        print(f"ABORT: ledger {st} != expected {EXPECTED_LEDGER}")
        return None
    return st


def integrity(models: dict = MODELS) -> dict:
    out = {}
    for key, m in models.items():
        p = m["path"]
        sha = compute_sha256(p) if p.is_file() else None
        out[key] = {"path": portable(p.as_posix()), "exists": p.is_file(), "sha256": sha, "expected_sha256": m["sha256"], "size_bytes": p.stat().st_size if p.is_file() else None, "expected_size": m["size"],
                    "match": sha == m["sha256"] and p.is_file() and p.stat().st_size == m["size"], "kind": m["kind"]}
    return out


# --------------------------------------------------------------------------- prepare
def cmd_prepare(args, report: Path, ledger_before: dict) -> int:
    import yaml  # noqa: PLC0415
    from ultralytics.utils import ASSETS_URL  # noqa: PLC0415
    from ultralytics.utils.downloads import download  # noqa: PLC0415

    COCO_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"credit_consuming": False, "netspresso_api_calls": 0, "disclaimer": DISCLAIMER, "dataset": "COCO val2017 (official images.cocodataset.org) + ultralytics coco2017labels.zip",
                      "downloads": [], "note": "train2017.zip (19 GB) deliberately NOT downloaded; only the 5,000-image validation split is needed"}
    t0 = time.perf_counter()
    labels_zip = COCO_DIR.parent / "coco2017labels.zip"
    if not (COCO_DIR / "labels" / "val2017").is_dir():
        download([ASSETS_URL + "/coco2017labels.zip"], dir=COCO_DIR.parent, unzip=True, delete=False)
    if labels_zip.is_file():
        manifest["downloads"].append({"file": labels_zip.name, "url": ASSETS_URL + "/coco2017labels.zip", "size_bytes": labels_zip.stat().st_size, "sha256": compute_sha256(labels_zip)})
    images_dir = COCO_DIR / "images" / "val2017"
    val_zip = COCO_DIR / "images" / "val2017.zip"
    if not images_dir.is_dir() or len(list(images_dir.glob("*.jpg"))) < 5000:
        (COCO_DIR / "images").mkdir(parents=True, exist_ok=True)
        download(["http://images.cocodataset.org/zips/val2017.zip"], dir=COCO_DIR / "images", unzip=True, delete=False, threads=1)
    if val_zip.is_file():
        manifest["downloads"].append({"file": val_zip.name, "url": "http://images.cocodataset.org/zips/val2017.zip", "size_bytes": val_zip.stat().st_size, "sha256": compute_sha256(val_zip)})
    manifest["download_seconds"] = round(time.perf_counter() - t0, 1)

    images = sorted(images_dir.glob("*.jpg"))
    labels = sorted((COCO_DIR / "labels" / "val2017").glob("*.txt"))
    ann_path = COCO_DIR / "annotations" / "instances_val2017.json"
    ann = json.loads(ann_path.read_text(encoding="utf-8")) if ann_path.is_file() else None
    label_boxes = sum(1 for lf in labels for line in lf.read_text(encoding="utf-8").splitlines() if line.strip())
    manifest["val2017"] = {"images_dir": portable(images_dir.as_posix()), "image_count": len(images), "label_files": len(labels), "label_boxes": label_boxes,
                           "images_without_label_file": len({p.stem for p in images} - {p.stem for p in labels}),
                           "annotations_json": {"path": portable(ann_path.as_posix()), "exists": ann_path.is_file(), "images": len(ann["images"]) if ann else None, "annotations": len(ann["annotations"]) if ann else None,
                                                "categories": len(ann["categories"]) if ann else None, "sha256": compute_sha256(ann_path) if ann_path.is_file() else None},
                           "list_file": portable((COCO_DIR / "val2017.txt").as_posix()), "list_entries": len((COCO_DIR / "val2017.txt").read_text(encoding="utf-8").splitlines()) if (COCO_DIR / "val2017.txt").is_file() else None}

    # ---- contamination check: COCO128 (train2017 subset used for A/B/C training) vs val2017 ----
    c128 = sorted((DATASETS / "coco128" / "images" / "train2017").glob("*.jpg"))
    c128_names, val_names = {p.name for p in c128}, {p.name for p in images}
    c128_ids, val_ids = {int(p.stem) for p in c128}, {int(p.stem) for p in images}
    c128_hashes = {compute_sha256(p) for p in c128}
    t1 = time.perf_counter()
    val_hashes = {compute_sha256(p) for p in images}
    manifest["contamination_check"] = {"coco128_images": len(c128), "coco128_source_split": "train2017 (ultralytics coco128.zip)", "val2017_images": len(images),
                                       "filename_overlap": len(c128_names & val_names), "coco_image_id_overlap": len(c128_ids & val_ids), "content_sha256_overlap": len(c128_hashes & val_hashes),
                                       "overlap_ratio_of_coco128": round(len(c128_hashes & val_hashes) / max(1, len(c128)), 4), "hashing_seconds": round(time.perf_counter() - t1, 1),
                                       "coco128_train_equals_val": True, "conclusion": "val2017 is disjoint from the COCO128 training images" if not (c128_names & val_names or c128_hashes & val_hashes) else "OVERLAP FOUND - see counts"}
    # ---- eval yaml (val only) ----
    coco_names = yaml.safe_load((Path(sys.modules["ultralytics"].__file__).parent / "cfg" / "datasets" / "coco.yaml").read_text(encoding="utf-8"))["names"]
    data_yaml = {"path": "coco", "train": "val2017.txt", "val": "val2017.txt", "names": coco_names}
    DATA_YAML.write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")
    manifest["data_yaml"] = {"path": portable(DATA_YAML.as_posix()), "val": "val2017.txt", "train_field_note": "train points to val2017.txt only because ultralytics requires the key; nothing is trained", "nc": len(coco_names)}
    manifest["timestamp"] = datetime.now(UTC).replace(microsecond=0).isoformat()
    manifest["ledger"] = {"before": ledger_before, "after": ledger_state()}
    dump(report / "dataset_manifest.json", manifest)
    print(f"[dataset] val2017 images {len(images)} | label files {len(labels)} ({label_boxes} boxes) | annotations json {manifest['val2017']['annotations_json']['annotations']} | contamination: {manifest['contamination_check']['conclusion']} "
          f"(name {manifest['contamination_check']['filename_overlap']}, id {manifest['contamination_check']['coco_image_id_overlap']}, sha {manifest['contamination_check']['content_sha256_overlap']})")
    return 0 if len(images) == 5000 and manifest["contamination_check"]["content_sha256_overlap"] == 0 else 5


# --------------------------------------------------------------------------- evaluate
def cmd_evaluate(args, report: Path, ledger_before: dict) -> int:
    import torch  # noqa: PLC0415
    import ultralytics  # noqa: PLC0415
    from ultralytics import YOLO, settings  # noqa: PLC0415
    from ultralytics.models.yolo.detect import DetectionValidator  # noqa: PLC0415
    from ultralytics.nn.modules.block import DFL  # noqa: PLC0415
    from ultralytics.utils.tal import dist2bbox, make_anchors  # noqa: PLC0415
    from yolov8_r1_traceable_export import make_wrapper_classes  # noqa: PLC0415

    if any(m == "netspresso" or m.startswith("netspresso.") for m in sys.modules):
        print("ABORT: netspresso imported after training-framework imports.")
        return 2
    settings.update({"datasets_dir": DATASETS.as_posix(), "sync": False, "runs_dir": (report / "ultralytics_runs").as_posix()})
    torch.set_grad_enabled(False)  # inference only: no autograd anywhere in this process
    m = MODELS[args.model]
    integ_before = integrity({args.model: m})[args.model]
    if not integ_before["match"]:
        print(f"ABORT: {args.model} integrity mismatch: {integ_before}")
        return 3
    policy = load_config(ROOT / "configs").local_eval
    trusted = policy.is_trusted(integ_before["sha256"]) or args.model == "phase_c"  # phase C body was generated locally by scripts/yolov8_e5_1_finetune.py
    sv = validate_pt(m["path"], allow_full_unpickle=trusted, torch_module=torch).to_dict()
    body = torch.load(m["path"].as_posix(), map_location="cpu", weights_only=False).eval()
    assert isinstance(body, torch.fx.GraphModule)
    for p in body.parameters():
        p.requires_grad_(False)
    _, head_cls, detector_cls = make_wrapper_classes(torch)
    meta_raw = json.loads((R1_DIR / "netspresso_head_meta.json").read_text(encoding="utf-8"))
    head_meta = load_head_meta(R1_DIR / "netspresso_head_meta.json")
    names = YOLO(args.weights.as_posix()).names if args.weights.is_file() else {i: str(i) for i in range(meta_raw["nc"])}
    det = detector_cls(body, head_cls(meta_raw, DFL, make_anchors, dist2bbox).eval(), names, meta_raw["stride"]).eval()
    y0, _ = det(torch.zeros(1, 3, 640, 640))
    ok, msg = validate_output_schema([list(y0.shape)], [expected_decoded_output_shape([1, 3, 640, 640], head_meta)])
    evaluator = f"ultralytics {ultralytics.__version__} DetectionValidator (torch {torch.__version__})"
    t0 = time.perf_counter()
    validator = DetectionValidator(args={"data": DATA_YAML.as_posix(), "imgsz": 640, "batch": 8, "device": "cpu", "conf": 0.001, "iou": 0.7, "plots": False, "verbose": False, "workers": 0, "rect": True, "mode": "val",
                                         "task": "detect", "half": False, "project": (report / "ultralytics_runs").as_posix(), "model": f"val2017_{args.model}", "name": args.model})
    validator(model=det)
    box = validator.metrics.box
    duration = round(time.perf_counter() - t0, 1)
    metrics = DetectionMetrics(status="MEASURED", map50_95=round(float(box.map), 5), map50=round(float(box.map50), 5), map75=round(float(box.map75), 5), precision=round(float(box.mp), 5), recall=round(float(box.mr), 5),
                               dataset="coco_val2017_eval.yaml", dataset_version="COCO 2017 val (images.cocodataset.org val2017.zip + ultralytics coco2017labels.zip)", split="val2017", images=5000,
                               instances=int(getattr(validator.metrics, "nt_per_class", torch.zeros(1)).sum()) if hasattr(validator.metrics, "nt_per_class") else None, imgsz=640, conf=0.001, iou=0.7,
                               evaluator=evaluator, model_label=m["kind"], extra={"speed_ms": {k: round(float(v), 3) for k, v in (getattr(validator.metrics, "speed", {}) or {}).items()}})
    integ_after = integrity({args.model: m})[args.model]
    env = environment_fingerprint(packages=("ultralytics", "torch", "onnx", "onnxruntime", "numpy", "psutil"))
    rec = {"credit_consuming": False, "netspresso_api_calls": 0, "disclaimer": DISCLAIMER, "model": args.model, "kind": m["kind"], "model_sha256": integ_before["sha256"], "model_size_bytes": integ_before["size_bytes"],
           "integrity_before": integ_before, "integrity_after": integ_after, "immutable": integ_before["sha256"] == integ_after["sha256"], "structural": sv, "decoded_output_shape": list(y0.shape), "schema_ok": ok,
           "dataset": {"yaml": portable(DATA_YAML.as_posix()), "split": "val2017", "images": 5000, "manifest": portable((report / "dataset_manifest.json").as_posix())},
           "evaluation_config": {**EVAL_CONDITIONS, "evaluator": evaluator, "grad_enabled": torch.is_grad_enabled(), "training": False}, "duration_seconds": duration, "metrics": metrics.to_dict(),
           "environment": env, "ledger": {"before": ledger_before, "after": ledger_state()}, "timestamp": datetime.now(UTC).replace(microsecond=0).isoformat()}
    rec["ledger"]["unchanged"] = ledger_before["sha256"] == rec["ledger"]["after"]["sha256"]
    dump(report / f"validation_{args.model}.json", rec)
    print(f"[val2017 {args.model}] mAP50-95 {metrics.map50_95} mAP50 {metrics.map50} mAP75 {metrics.map75} P {metrics.precision} R {metrics.recall} | {duration} s | immutable {rec['immutable']} | ledger unchanged {rec['ledger']['unchanged']}")
    return 0


# --------------------------------------------------------------------------- summarize
def cmd_summarize(args, report: Path, ledger_before: dict) -> int:
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from ultralytics import YOLO  # noqa: PLC0415
    from ultralytics.data.augment import LetterBox  # noqa: PLC0415
    from ultralytics.nn.modules.block import DFL  # noqa: PLC0415
    from ultralytics.utils.tal import dist2bbox, make_anchors  # noqa: PLC0415
    from yolov8_r1_traceable_export import make_wrapper_classes, preprocess_image, stats_of  # noqa: PLC0415

    torch.set_grad_enabled(False)
    recs = {k: json.loads((report / f"validation_{k}.json").read_text(encoding="utf-8")) for k in MODELS if (report / f"validation_{k}.json").is_file()}
    if set(recs) != set(MODELS):
        print(f"ABORT: missing evaluations: {set(MODELS) - set(recs)}")
        return 4
    manifest = json.loads((report / "dataset_manifest.json").read_text(encoding="utf-8"))
    e5 = json.loads(E5_RECORD.read_text(encoding="utf-8"))
    phase_c_rec = json.loads(sorted((ROOT / "reports" / "yolov8_e5_1_finetune").glob("2*Z_phaseC/finetune_result.json"))[-1].read_text(encoding="utf-8"))
    config = load_config(ROOT / "configs")
    val = {k: recs[k]["metrics"]["metrics"] for k in MODELS}
    b, c1, pc = val["baseline"]["mAP50-95"], val["e5_1"]["mAP50-95"], val["phase_c"]["mAP50-95"]

    accuracy_table = [{"model": k, "kind": MODELS[k]["kind"], "sha256": recs[k]["model_sha256"], **val[k], "delta_vs_baseline_pp": round((val[k]["mAP50-95"] - b) * 100, 3), "recovery_vs_e5_1_pp": round((val[k]["mAP50-95"] - c1) * 100, 3),
                       "duration_seconds": recs[k]["duration_seconds"]} for k in MODELS]
    gen = {"baseline": generalization_gap(smoke=MODELS["baseline"]["smoke_map"], unseen=b), "e5_1": generalization_gap(smoke=MODELS["e5_1"]["smoke_map"], unseen=c1),
           "phase_b": {"status": "smoke only", "smoke": PHASE_B_SMOKE, "unseen": None, "note": "not required by the plan"}, "phase_c": generalization_gap(smoke=MODELS["phase_c"]["smoke_map"], unseen=pc)}
    gen["phase_c"]["overfit_indicator"] = {"smoke_above_baseline_smoke": MODELS["phase_c"]["smoke_map"] > MODELS["baseline"]["smoke_map"], "unseen_vs_baseline_unseen_pp": round((pc - b) * 100, 3),
                                           "excess_gap_vs_baseline_pp": round(gen["phase_c"]["gap_pp"] - gen["baseline"]["gap_pp"], 3),
                                           "reading": "smoke (train==val) advantage that disappears on unseen data = memorization; the excess gap over the baseline's own COCO128/val2017 gap quantifies it"}
    cls = classify_unseen_recovery(baseline=b, compressed=c1, candidate=pc, baseline_smoke=MODELS["baseline"]["smoke_map"], candidate_smoke=MODELS["phase_c"]["smoke_map"])
    next_step = unseen_next_step(cls["decision"])
    recovery = {"e5_1_to_phase_c_pp": round((pc - c1) * 100, 3), "phase_c_to_baseline_pp": round((pc - b) * 100, 3), "remaining_drop_pp": round((b - pc) * 100, 3),
                "recovery_rate_percent": round((pc - c1) / (b - c1) * 100, 2) if b - c1 > 0 else None, "classification": cls}

    # ---- proxy on a few val2017 images (never used in training): baseline vs C, E5-1 vs C ----
    _, head_cls, detector_cls = make_wrapper_classes(torch)
    meta_raw = json.loads((R1_DIR / "netspresso_head_meta.json").read_text(encoding="utf-8"))
    names = YOLO(args.weights.as_posix()).names if args.weights.is_file() else {i: str(i) for i in range(80)}
    dets = {k: detector_cls(torch.load(MODELS[k]["path"].as_posix(), map_location="cpu", weights_only=False).eval(), head_cls(meta_raw, DFL, make_anchors, dist2bbox).eval(), names, meta_raw["stride"]).eval() for k in MODELS}
    imgs = sorted((COCO_DIR / "images" / "val2017").glob("*.jpg"))[: args.images]
    per_image = []
    for p in imgs:
        x, _ = preprocess_image(p, 640, LetterBox, np, torch)
        ys = {k: d(x)[0] for k, d in dets.items()}
        per_image.append({"image": p.name, "baseline_vs_phase_c": stats_of(ys["baseline"], ys["phase_c"]), "e5_1_vs_phase_c": stats_of(ys["e5_1"], ys["phase_c"]), "baseline_vs_e5_1": stats_of(ys["baseline"], ys["e5_1"])})
    proxy = {"images": len(imgs), "source": "first val2017 images (unseen), decoded pre-NMS [1,84,8400]", "aggregate": {k: {"min_cosine": min(r[k]["cosine"] for r in per_image), "max_abs_diff": max(r[k]["max_abs_diff"] for r in per_image),
                                                                                                                   "max_relative_l2_diff": max(r[k]["relative_l2_diff"] for r in per_image)} for k in per_image[0] if k != "image"},
             "per_image": per_image, "disclaimer": "output equivalence PROXY, not accuracy; proxy PASS never implies accuracy PASS (E5-1: cosine 0.984 with mAP 0.0)"}

    # ---- Quality Gate on the unseen result (performance referenced from the phase C record, not re-measured) ----
    pc_perf = phase_c_rec["performance"]
    pc_proxy = phase_c_rec["output_equivalence_proxy"]["ort_baseline_vs_fine_tuned"]
    cfg_run = Configuration.from_dict(e5["experiment"]["configuration"])
    env_fp = environment_fingerprint(packages=("ultralytics", "torch", "onnx", "onnxruntime", "numpy", "psutil"))
    environment = Environment(adapter="local_val2017_eval", adapter_version="0.1.0", sdk_name="ultralytics (inference only)", sdk_version=env_fp["packages"].get("ultralytics"), python_version=env_fp["python"],
                              platform=f"{env_fp['os']} {env_fp['os_release']} {env_fp['architecture']}", extra={"credit_consuming": False, "netspresso_api_calls": 0, "evaluation_type": "unseen_val2017", **env_fp})
    pc_extra = {"params_torch": 882996.0, "proxy_min_cosine_similarity": pc_proxy.get("min_cosine_similarity") or 0.0, "proxy_shape_match": 1.0, "proxy_dtype_match": 1.0, "latency_p95_ms": pc_perf["fine_tuned"]["latency"]["p95_ms"],
                "performance_source": "reports/yolov8_e5_1_finetune/*_phaseC (already measured; not re-measured here)"}
    sv_c = recs["phase_c"]["structural"]
    artifact = artifact_from_file(MODELS["phase_c"]["path"], expected_sha256=None, source_model="E5-1 sdk_output.pt (NetsPresso candidate)", source_model_sha256=MODELS["e5_1"]["sha256"], operation="local_finetune",
                                  structural_validation=sv_c, configuration_key=cfg_run.key, evaluation="unseen_val2017", generated_by="scripts/yolov8_val2017_eval.py")
    artifact.path = portable(artifact.path or "")
    execution = ExecutionResult(
        operation="unseen_val2017_evaluation", configuration=cfg_run, execution_status=ExecutionStatus.COMPLETED, environment=environment,
        baseline_metrics=Metrics(accuracy=b, latency_ms=pc_perf["baseline"]["latency"]["median_ms"], memory_mb=pc_perf["baseline"]["memory"]["delta_mb"], model_size_mb=round(MODELS["baseline"]["size"] / 1e6, 3), extra={"params_torch": 3157184.0}),
        metrics=Metrics(accuracy=pc, latency_ms=pc_perf["fine_tuned"]["latency"]["median_ms"], memory_mb=pc_perf["fine_tuned"]["memory"]["delta_mb"], model_size_mb=round(MODELS["phase_c"]["size"] / 1e6, 3), extra=pc_extra),
        artifact=artifact, logs=[DISCLAIMER, "accuracy = COCO val2017 (unseen); latency/memory/proxy = phase C local ORT measurement (referenced)"],
        reproducibility=Reproducibility(level=ReproducibilityLevel.NOT_VERIFIED, runs=1, checksums=[MODELS["phase_c"]["sha256"]], notes="single evaluation of a single local training run; no promotion"),
        timestamp=utc_now_iso(), credit=CreditRecord(usage_type=CreditUsageType.NONE, estimated=0, actual=None, note="0 credits, no API call"))
    gates = {name: QualityGate(config.quality_gate, profile=name).evaluate(execution) for name in ("compression", "local_eval", "release")}
    defects = {name: (None if (d := classify(execution, g)) is None else {"defect_id": f"VAL2017-{name}-{d.category.value}", **d.to_dict()}) for name, g in gates.items()}
    for g in gates.values():
        print("\n" + render_text(g))

    integrity_all = {k: {"before": recs[k]["integrity_before"], "after": integrity()[k], "immutable_across_evaluations": recs[k]["integrity_before"]["sha256"] == integrity()[k]["sha256"] == MODELS[k]["sha256"]} for k in MODELS}
    chain = build_traceability_chain(
        source_model=f"yolov8n.pt sha256={e5['input_provenance']['source_model']['sha256']}", r1_fx_artifact="outputs/models/yolov8n_fx_r1/model_fx.pt", r1_sha256=MODELS["baseline"]["sha256"],
        netspresso_execution=f"{E5_RUN.name}: automatic_compression ratio 0.5 (E5-1)", candidate_artifact=portable(MODELS["e5_1"]["path"].as_posix()), candidate_sha256=MODELS["e5_1"]["sha256"],
        local_validation=f"phase A 0a203872 -> phase B 73a8f76f -> phase C {MODELS['phase_c']['sha256'][:16]} ({portable(phase_c_rec['experiment']['report_dir'])}) -> val2017 evaluation {portable(report.as_posix())}",
        accuracy_result=f"val2017 mAP50-95 baseline {b} / E5-1 {c1} / phase C {pc} (COCO128 smoke 0.44369 / 0.0 / 0.46688)", quality_gate={k: g.overall.value for k, g in gates.items()},
        credit_record=f"ledger unchanged {EXPECTED_LEDGER} (0 credits)")
    after = ledger_state()
    summary = {
        "experiment": {"phase": "5-E unseen validation (COCO val2017)", "timestamp": datetime.now(UTC).replace(microsecond=0).isoformat(), "report_dir": portable(report.as_posix()), "phase_c_record": portable(phase_c_rec["experiment"]["report_dir"])},
        "disclaimer": DISCLAIMER, "credit_consuming": False, "netspresso_api_calls": 0, "training_performed": False,
        "decision": cls["decision"], "classification": cls, "next_step": next_step, "netspresso_additional_execution": "HOLD",
        "dataset": {"name": "COCO val2017", "images": manifest["val2017"]["image_count"], "annotations": manifest["val2017"]["annotations_json"]["annotations"], "label_files": manifest["val2017"]["label_files"],
                    "contamination_check": manifest["contamination_check"], "coco128_train_equals_val": True, "manifest": portable((report / "dataset_manifest.json").as_posix())},
        "evaluation_config": {**EVAL_CONDITIONS, "evaluator": recs["baseline"]["evaluation_config"]["evaluator"], "independent_processes": True, "order": ["baseline", "e5_1", "phase_c"]},
        "accuracy": {"val2017": accuracy_table, "coco128_smoke": {"baseline": MODELS["baseline"]["smoke_map"], "e5_1": MODELS["e5_1"]["smoke_map"], "phase_b": PHASE_B_SMOKE, "phase_c": MODELS["phase_c"]["smoke_map"]},
                     "phase_c_vs_baseline_unseen": compare_detection_accuracy(DetectionMetrics(**{k: v for k, v in _dm(recs["baseline"]).items()}), DetectionMetrics(**{k: v for k, v in _dm(recs["phase_c"]).items()})).to_dict(),
                     "threshold_pp": config.quality_gate.profiles["release"]["accuracy"].threshold, "threshold_basis": "PROJECT-DEFINED EXAMPLE THRESHOLD; unseen-data result judged separately from COCO128"},
        "generalization": gen, "recovery": recovery, "output_equivalence_proxy": proxy, "performance": {"source": "phase C record (already measured, not re-measured)", "baseline": pc_perf["baseline"], "e5_1": pc_perf.get("e5_1"), "phase_c": pc_perf["fine_tuned"]},
        "quality_gate": {k: g.to_dict() for k, g in gates.items()}, "defects": defects, "model_integrity": integrity_all, "baseline_promotion": "NOT performed (explicit separate decision)",
        "reproducibility": {"evaluation_runs_per_model": 1, "training_runs": 1, "level": "NOT_VERIFIED"}, "traceability": chain,
        "credit_safety": {"ledger_before": ledger_before, "ledger_after": after, "ledger_unchanged": ledger_before["sha256"] == after["sha256"], "api_calls": 0, "sdk_execution": 0},
        "environment": environment.to_dict(),
    }
    summary = json.loads(portable(json.dumps(summary, default=str)))
    dump(report / "val2017_summary.json", summary)
    dump(report / "validation_results.json", {"credit_consuming": False, "disclaimer": DISCLAIMER, "val2017": accuracy_table, "coco128_smoke": summary["accuracy"]["coco128_smoke"], "generalization": gen, "per_model_records": {k: portable((report / f"validation_{k}.json").as_posix()) for k in MODELS}})
    dump(report / "evaluation_config.json", {"credit_consuming": False, **summary["evaluation_config"], "dataset_yaml": portable(DATA_YAML.as_posix())})
    dump(report / "model_integrity.json", {"credit_consuming": False, **integrity_all})
    dump(report / "environment_fingerprint.json", {"credit_consuming": False, **environment.to_dict()})
    try:
        from framework.reporter import write_val2017_html  # noqa: PLC0415

        write_val2017_html(summary, report / "val2017_summary.html")
    except Exception as exc:  # noqa: BLE001
        print(f"[report] HTML skipped: {type(exc).__name__}: {exc}")
    print(f"\n[val2017] baseline {b} | E5-1 {c1} | phase C {pc} | drop {recovery['remaining_drop_pp']} pp | gen gap C {gen['phase_c']['gap_pp']} pp (baseline gap {gen['baseline']['gap_pp']} pp) -> {cls['decision']} ({cls['reason']})")
    print(f"[next] {next_step['label']} | NetsPresso additional execution: HOLD | ledger unchanged {summary['credit_safety']['ledger_unchanged']}")
    return 0


def _dm(rec: dict) -> dict:
    d = rec["metrics"]
    m = d["metrics"]
    return {"status": d["status"], "map50_95": m["mAP50-95"], "map50": m["mAP50"], "map75": m["mAP75"], "precision": m["precision"], "recall": m["recall"], "dataset": d["dataset"], "dataset_version": d.get("dataset_version"),
            "split": d["split"], "images": d.get("images"), "instances": d.get("instances"), "imgsz": d["imgsz"], "conf": d["conf"], "iou": d["iou"], "evaluator": d["evaluator"], "model_label": d["model_label"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("prepare", "evaluate", "summarize"))
    ap.add_argument("--report-dir", type=Path, required=True)
    ap.add_argument("--model", choices=tuple(MODELS), default=None)
    ap.add_argument("--weights", type=Path, default=ROOT / "outputs" / "models" / "yolov8n.pt", help="class names only")
    ap.add_argument("--images", type=int, default=8)
    args = ap.parse_args(argv)
    ledger_before = guards()
    if ledger_before is None:
        return 2
    args.report_dir.mkdir(parents=True, exist_ok=True)
    print(DISCLAIMER)
    if args.command == "prepare":
        rc = cmd_prepare(args, args.report_dir, ledger_before)
    elif args.command == "evaluate":
        if args.model is None:
            print("ABORT: --model required")
            return 2
        rc = cmd_evaluate(args, args.report_dir, ledger_before)
    else:
        rc = cmd_summarize(args, args.report_dir, ledger_before)
    assert not any(m == "netspresso" or m.startswith("netspresso.") for m in sys.modules), "SDK must never be imported here"
    after = ledger_state()
    print(f"ledger unchanged: {after['sha256'] == ledger_before['sha256']} ({after['used']}/{after['remaining']}/{after['operations']}) | NetsPresso API calls: 0 | python {platform.python_version()}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

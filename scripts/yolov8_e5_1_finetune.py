#!/usr/bin/env python
"""Phase 5-E fine-tuning recovery: LOCAL, ZERO-CREDIT fine-tuning of the E5-1 NetsPresso-compressed YOLOv8n body.

Runs in the upstream environment (``.venv-yolo``: ultralytics 8.4.173, torch 2.14, CPU):

    env -u NETSPRESSO_API_KEY .venv-yolo/Scripts/python scripts/yolov8_e5_1_finetune.py --phase A --epochs 10
    env -u NETSPRESSO_API_KEY .venv-yolo/Scripts/python scripts/yolov8_e5_1_finetune.py --phase B --epochs 20 --init-body outputs/models/yolov8n_e5_1_finetuned/model_fx_phaseA.pt

What it does (no NetsPresso import, no key, no network):
    0. verifies the ledger (50 / 450 / 2), the E5-1 record and the candidate SHA (21d8cbd1...); old fork artifact forbidden
    1. loads the compressed fx body (GraphModule), re-attaches the R1 decode head, checks [1,84,8400] BEFORE training
    2. builds a tiny deterministic training loop around unmodified upstream pieces: build_yolo_dataset / build_dataloader /
       v8DetectionLoss (the body already contains the Detect cv2/cv3 branches, so the loss head only re-shapes the maps)
       Phase A: backbone+neck frozen (params + BN statistics), detect branches cv2*/cv3* trained
       Phase B: everything trainable (starting from a Phase A body), lower learning rate
    3. saves a NEW local artifact outputs/models/yolov8n_e5_1_finetuned/model_fx_phase<X>.pt (E5-1 files are never touched)
    4. validates structure / shapes / params / FLOPs (architecture must be unchanged)
    5. three-way COCO128 smoke accuracy (baseline / compressed / fine-tuned) with the E5-1 conditions (rect=True, conf 0.001, IoU 0.7)
    6. output-equivalence proxy, isolated-process ORT latency/memory, Quality Gate profiles, defects, registry candidate,
       traceability chain, next-step recommendation -> reports/yolov8_e5_1_finetune/<stamp>/

COCO128 train and val are the SAME 128 images: every accuracy number here is smoke/diagnostic, never release evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

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
from framework.evaluation.input_gate import build_traceability_chain, verify_input_artifact  # noqa: E402
from framework.evaluation.local_ort import (  # noqa: E402
    PROXY_DISCLAIMER,
    compare_outputs,
    measure_in_subprocess,
)
from framework.evaluation.model_stats import (  # noqa: E402
    count_onnx_parameters,
    count_torch_parameters,
    estimate_onnx_macs,
)
from framework.evaluation.recovery import (  # noqa: E402
    architecture_unchanged,
    classify_recovery,
    continuation_metrics,
    epoch_trend,
    milestones_reached,
    phase_c_decision,
    phase_c_next_step,
    plateau_detected,
    recommend_next_step,
    three_way_comparison,
    validate_finetune_config,
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
E5_RUN = ROOT / "reports" / "real_runs" / "20261005T040009Z_automatic_compression"
E5_RECORD = ROOT / "reports" / "yolov8_e5_1" / "20261005T040009Z" / "e5_1_result.json"
CANDIDATE_SHA = "21d8cbd1361cfd3ef55c0c132d11f3fad294dbde6f175e8082bc0233b860e0f1"
R1_SHA = "d8e761dae29301ef51df9e7679c729338a522d397e271a900d453d171e1c022b"
OLD_FORK_ARTIFACT = "yolov8n_fx/model_fx.pt"
EXPECTED_LEDGER = (50, 450, 2)
DISCLAIMER = ("Local fine-tuning recovery experiment. No NetsPresso API call was made. The fine-tuned artifact is a LOCAL derivative of the "
              "E5-1 NetsPresso candidate, not a NetsPresso output. COCO128 train == val (128 images): smoke/diagnostic only, never release evidence.")
REG_MAX = 16
HYP = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)  # ultralytics default loss gains


def portable(text: str) -> str:
    root_fwd = ROOT.as_posix() + "/"
    root_back = str(ROOT) + "\\"
    return text.replace(root_back.replace("\\", "\\\\"), "").replace(root_back, "").replace(root_fwd, "")


def dump(path: Path, data: object) -> None:
    path.write_text(portable(json.dumps(data, indent=2, ensure_ascii=False, default=str)) + "\n", encoding="utf-8")


def ledger_state() -> tuple[int, int, int, str]:
    d = json.loads(LEDGER.read_text(encoding="utf-8"))
    return d["used_credit"], d["remaining_estimate"], len(d["operations"]), hashlib.sha256(LEDGER.read_bytes()).hexdigest()


def metrics_from_record(d: dict) -> DetectionMetrics:
    m = d.get("metrics") or {}
    return DetectionMetrics(status=d["status"], reason=d.get("reason"), map50_95=m.get("mAP50-95"), map50=m.get("mAP50"), map75=m.get("mAP75"), precision=m.get("precision"),
                            recall=m.get("recall"), dataset=d.get("dataset"), dataset_version=d.get("dataset_version"), split=d.get("split"), images=d.get("images"),
                            instances=d.get("instances"), imgsz=d.get("imgsz"), conf=d.get("conf"), iou=d.get("iou"), evaluator=d.get("evaluator"), model_label=d.get("model_label"))


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=("A", "B", "C"), default="A", help="A: detect branches only (backbone frozen); B: all layers (from --init-body); C: continuation of B (all layers, global epochs)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--epoch-offset", type=int, default=0, help="epochs already trained by previous phases (phase C: 40) -> global_epoch = offset + epoch")
    ap.add_argument("--prev-record", "--phase-a-record", dest="prev_record", type=Path, default=None, help="finetune_result.json of the phase this run continues from")
    ap.add_argument("--plateau-delta", type=float, default=0.005, help="early-stop: improvement per validation point below this ...")
    ap.add_argument("--plateau-points", type=int, default=3, help="... for this many consecutive validation points")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lr", type=float, default=None, help="default 1e-3 (A) / 2e-4 (B)")
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-interval", type=int, default=2)
    ap.add_argument("--candidate", type=Path, default=E5_RUN / "sdk_output" / "sdk_output.pt")
    ap.add_argument("--init-body", type=Path, default=None, help="start from a previously fine-tuned body (phase B/C)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "models" / "yolov8n_e5_1_finetuned")
    ap.add_argument("--report-dir", type=Path, default=None)
    ap.add_argument("--weights", type=Path, default=ROOT / "outputs" / "models" / "yolov8n.pt", help="class names only")
    ap.add_argument("--data", default="coco128.yaml")
    ap.add_argument("--images", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--skip-ort", action="store_true")
    args = ap.parse_args(argv)
    if args.lr is None:
        args.lr = 1e-3 if args.phase in ("A", "C") else 2e-4

    # ---------------- 0. zero-credit preconditions + E5-1 record + candidate identity ----------------
    if "NETSPRESSO_API_KEY" in os.environ:  # presence check only; the value is never read
        print("ABORT: NETSPRESSO_API_KEY is present in the environment. This phase is local-only: re-run with `env -u NETSPRESSO_API_KEY`. No training performed.")
        return 2
    if any(m == "netspresso" or m.startswith("netspresso.") for m in sys.modules):
        print("ABORT: netspresso SDK is imported in this interpreter; local-only phase refuses to run.")
        return 2
    used, remaining, ops, ledger_sha = ledger_state()
    if (used, remaining, ops) != EXPECTED_LEDGER:
        print(f"ABORT: ledger {used}/{remaining}/{ops} != expected {EXPECTED_LEDGER}")
        return 2
    e5 = json.loads(E5_RECORD.read_text(encoding="utf-8"))
    e5_checks = {"operation": e5["experiment"]["operation"] == "automatic_compression", "ratio": e5["experiment"]["compression_ratio"] == 0.5,
                 "status": e5["statuses"]["execution_status"] == "COMPLETED", "candidate_sha": e5["candidate_artifact"]["pt"]["sha256"] == CANDIDATE_SHA,
                 "params": e5["model_metrics"]["params"]["candidate_sdk"] == 882996, "map50_95_zero": e5["accuracy"]["candidate"]["metrics"]["mAP50-95"] == 0.0,
                 "source_sha": e5["input_provenance"]["source_sha256"] == R1_SHA}
    e5_files_sha_before = {p.name: compute_sha256(p) for p in (E5_RECORD, E5_RUN / "execution_result.json", args.candidate)}
    identity = verify_input_artifact(args.candidate, expected_sha256=CANDIDATE_SHA, expected_size_bytes=3693813, forbidden_paths=(OLD_FORK_ARTIFACT,))
    identity["details"]["path"] = portable(identity["details"]["path"])
    print(f"[precondition] ledger {used}/{remaining}/{ops} OK | E5-1 record checks {all(e5_checks.values())} {e5_checks} | candidate identity {identity['status']}")
    if not all(e5_checks.values()) or identity["status"] != "PASS":
        print("BLOCKED: E5-1 record or candidate identity mismatch. No training performed.")
        return 3
    if args.init_body is not None and not args.init_body.is_file():
        print(f"BLOCKED: --init-body not found: {args.init_body}")
        return 3

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report = args.report_dir or (ROOT / "reports" / "yolov8_e5_1_finetune" / f"{stamp}_phase{args.phase}")
    report.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(DISCLAIMER)

    import numpy as np  # noqa: PLC0415
    import onnx  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import ultralytics  # noqa: PLC0415
    from ultralytics import YOLO, settings  # noqa: PLC0415
    from ultralytics.cfg import get_cfg  # noqa: PLC0415
    from ultralytics.data.augment import LetterBox  # noqa: PLC0415
    from ultralytics.data.build import build_dataloader, build_yolo_dataset  # noqa: PLC0415
    from ultralytics.data.utils import check_det_dataset  # noqa: PLC0415
    from ultralytics.models.yolo.detect import DetectionValidator  # noqa: PLC0415
    from ultralytics.nn.modules.block import DFL  # noqa: PLC0415
    from ultralytics.utils.loss import v8DetectionLoss  # noqa: PLC0415
    from ultralytics.utils.tal import dist2bbox, make_anchors  # noqa: PLC0415
    from ultralytics.utils.torch_utils import init_seeds  # noqa: PLC0415
    from yolov8_r1_traceable_export import make_wrapper_classes, preprocess_image, stats_of  # noqa: PLC0415

    if any(m == "netspresso" or m.startswith("netspresso.") for m in sys.modules):
        print("ABORT: netspresso SDK became importable/imported after the training imports; refusing to continue.")
        return 2
    settings.update({"datasets_dir": DATASETS.as_posix(), "sync": False, "runs_dir": (report / "ultralytics_runs").as_posix()})
    init_seeds(args.seed, deterministic=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    _body_cls, head_cls, detector_cls = make_wrapper_classes(torch)
    evaluator = f"ultralytics {ultralytics.__version__} DetectionValidator (torch {torch.__version__})"
    input_shape = [1, 3, args.imgsz, args.imgsz]
    config = load_config(ROOT / "configs")
    policy = config.local_eval
    head_meta_path = R1_DIR / "netspresso_head_meta.json"
    head_meta = load_head_meta(head_meta_path)
    meta_raw = json.loads(head_meta_path.read_text(encoding="utf-8"))
    names = YOLO(args.weights.as_posix()).names if args.weights.is_file() else {i: str(i) for i in range(meta_raw["nc"])}

    # ---------------- 1. load candidate / init body, pre-training shape checks ----------------
    start_path = args.init_body or args.candidate
    start_sha = compute_sha256(start_path)
    trusted = policy.is_trusted(start_sha) or args.init_body is not None  # phase-A outputs are generated locally by this script
    body = torch.load(start_path.as_posix(), map_location="cpu", weights_only=False)
    assert isinstance(body, torch.fx.GraphModule), f"start body is not a GraphModule: {type(body).__name__}"
    body.eval()
    with torch.no_grad():
        pre_maps = body(torch.zeros(*input_shape))
        pre_det = detector_cls(body, head_cls(meta_raw, DFL, make_anchors, dist2bbox).eval(), names, meta_raw["stride"]).eval()
        y_pre, _ = pre_det(torch.zeros(*input_shape))
    ok_maps, _ = validate_output_schema([list(m.shape) for m in pre_maps], expected_body_output_shapes(input_shape, head_meta))
    ok_dec, _ = validate_output_schema([list(y_pre.shape)], [expected_decoded_output_shape(input_shape, head_meta)])
    params_start = count_torch_parameters(body)
    start_body = deepcopy(body).eval()  # frozen copy of the starting weights (previous phase) for proxy/perf comparison
    prev_record = json.loads(args.prev_record.read_text(encoding="utf-8")) if args.prev_record and args.prev_record.is_file() else None
    prev_best = prev_record["accuracy"]["fine_tuned"]["metrics"]["mAP50-95"] if prev_record else None
    if prev_record is not None and prev_record["artifact_validation"]["pt"]["sha256"] != start_sha:
        print(f"BLOCKED: --init-body sha {start_sha[:16]} does not match the previous record's artifact sha {prev_record['artifact_validation']['pt']['sha256'][:16]}")
        return 4
    print(f"[pre-train] start body {portable(start_path.as_posix())} sha {start_sha[:16]} params {params_start} | maps ok={ok_maps} decoded {list(y_pre.shape)} ok={ok_dec}")
    if not (ok_maps and ok_dec):
        print("BLOCKED: candidate does not produce the expected shapes before training.")
        return 4

    # ---------------- 2. trainability analysis + freeze plan ----------------
    all_params = list(body.named_parameters())
    sdk_frozen = [n for n, p in all_params if not p.requires_grad]
    detect_branch = [n for n, _ in all_params if n.startswith(("cv2", "cv3"))]
    bn_modules = [n for n, m in body.named_modules() if isinstance(m, torch.nn.BatchNorm2d)]
    for _, p in all_params:
        p.requires_grad_(False)
    trainable_names = detect_branch if args.phase == "A" else [n for n, _ in all_params]
    for n, p in all_params:
        if n in trainable_names:
            p.requires_grad_(True)

    def set_modes():
        """Phase A: frozen backbone/neck stay in eval (BN statistics untouched); detect branches train. Phase B: all train."""
        if args.phase == "A":
            body.eval()
            for n, m in body.named_modules():
                if n.startswith(("cv2", "cv3")):
                    m.train()
        else:
            body.train()

    trainability = {"total_parameter_tensors": len(all_params), "total_params": params_start,
                    "start_requires_grad_false_tensors": len(sdk_frozen), "start_requires_grad_false_names": sdk_frozen if args.phase == "A" else "<all tensors: previous phase saved the body with grads disabled; not SDK state>",
                    "detect_branch_tensors": len(detect_branch), "detect_branch_params": sum(p.numel() for n, p in all_params if n in detect_branch),
                    "batchnorm_modules": len(bn_modules), "trainable_tensors": len(trainable_names), "trainable_params": sum(p.numel() for n, p in all_params if n in trainable_names),
                    "frozen_tensors": len(all_params) - len(trainable_names),
                    "notes": ["detect branches cv2*/cv3* live INSIDE the compressed GraphModule (fork export contract); the decode head (DFL/anchors/sigmoid) has no trainable weights",
                              "phase A keeps backbone/neck BatchNorm in eval mode so pruned-channel statistics are not drifted by 128-image batches",
                              ("11 tensors came back from the SDK with requires_grad=False; phases B/C re-enable them explicitly" if args.phase == "A"
                               else "requires_grad flags of the start body reflect how the previous phase saved it (all False), not the SDK; all re-enabled here")]}
    print(f"[trainability] trainable {trainability['trainable_tensors']}/{trainability['total_parameter_tensors']} tensors ({trainability['trainable_params']} params) | BN modules {len(bn_modules)} | start requires_grad=False tensors {len(sdk_frozen)}")

    # ---------------- 3. training components (upstream pieces, unmodified) ----------------
    class TrainHead(torch.nn.Module):
        """Re-shapes the body's per-level maps into the dict upstream v8DetectionLoss expects (boxes / scores / feats)."""

        def __init__(self):
            super().__init__()
            self.nc, self.nl, self.reg_max = int(meta_raw["nc"]), int(meta_raw["nl"]), REG_MAX
            self.no = self.nc + 4 * self.reg_max
            self.register_buffer("stride", torch.tensor([float(s) for s in meta_raw["stride"]]))

        def forward(self, feats):
            bs = feats[0].shape[0]
            boxes = torch.cat([f[:, : 4 * self.reg_max].reshape(bs, 4 * self.reg_max, -1) for f in feats], -1)
            scores = torch.cat([f[:, 4 * self.reg_max :].reshape(bs, self.nc, -1) for f in feats], -1)
            return {"boxes": boxes, "scores": scores, "feats": list(feats)}

    class TrainModel(torch.nn.Module):
        def __init__(self, body_, head_):
            super().__init__()
            self.model = torch.nn.Sequential(body_, head_)
            self.args = HYP

        def forward(self, x):
            return self.model[1](self.model[0](x))

    train_model = TrainModel(body, TrainHead())
    criterion = v8DetectionLoss(train_model)
    cfg = get_cfg(overrides={"imgsz": args.imgsz, "seed": args.seed, "task": "detect", "mode": "train", "batch": args.batch, "workers": 0})
    data = check_det_dataset(args.data)
    dataset = build_yolo_dataset(cfg, data["train"], args.batch, data, mode="train", rect=False, stride=32)
    loader = build_dataloader(dataset, args.batch, workers=0, shuffle=True)
    aug_keys = ("mosaic", "mixup", "cutmix", "copy_paste", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr", "erasing", "auto_augment")
    augmentation = {k: getattr(cfg, k, None) for k in aug_keys}
    decay_params = [p for n, p in all_params if n in trainable_names and p.ndim > 1]
    no_decay_params = [p for n, p in all_params if n in trainable_names and p.ndim <= 1]
    optimizer = torch.optim.AdamW([{"params": decay_params, "weight_decay": args.weight_decay}, {"params": no_decay_params, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.999))
    iters_per_epoch = len(loader)
    total_iters = args.epochs * iters_per_epoch
    warmup_iters = max(iters_per_epoch, 1)

    def lr_at(it: int) -> float:
        if it < warmup_iters:
            return args.lr * (it + 1) / warmup_iters
        prog = (it - warmup_iters) / max(1, total_iters - warmup_iters)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))  # cosine to 10 %

    finetune_config = {
        "phase": args.phase, "experiment": "E5-1 local fine-tuning recovery", "seed": args.seed, "deterministic": True, "epochs": args.epochs, "batch": args.batch, "imgsz": args.imgsz,
        "optimizer": "AdamW(betas=0.9/0.999)", "lr": args.lr, "lr_schedule": "linear warm-up 1 epoch, cosine decay to 10 %", "weight_decay": args.weight_decay,
        "weight_decay_applies_to": "conv weights only (ndim>1); BN/bias excluded", "grad_clip_norm": 10.0,
        "loss": "ultralytics v8DetectionLoss (box 7.5 / cls 0.5 / dfl 1.5, TaskAlignedAssigner topk 10)",
        "freeze": {"phase": args.phase, "trainable_prefixes": ["cv2", "cv3"] if args.phase == "A" else ["<all>"], "frozen_backbone_bn_eval": args.phase == "A",
                   "trainable_tensors": len(trainable_names), "frozen_tensors": len(all_params) - len(trainable_names)},
        "augmentation": augmentation, "workers": 0, "device": "cpu", "amp": False, "ema": False,
        "dataset": {"identity": args.data, "source": "ultralytics coco128.zip (COCO 2017 train subset)", "train": data["train"], "val": data["val"], "train_images": len(dataset),
                    "val_images": 128, "train_equals_val": data["train"] == data["val"], "labels_cache": "datasets/coco128/labels/train2017.cache", "preprocessing": "ultralytics YOLODataset train pipeline (mosaic, HSV, flip, letterbox) / val: letterbox rect"},
        "torch_version": torch.__version__, "ultralytics_version": ultralytics.__version__, "python": sys.version.split()[0],
        "source_artifact": portable(start_path.as_posix()), "source_artifact_sha256": start_sha, "source_artifact_trusted": trusted, "e5_1_candidate_sha256": CANDIDATE_SHA, "r1_input_sha256": R1_SHA,
        "iterations_per_epoch": iters_per_epoch, "total_iterations": total_iters, "val_interval_epochs": args.val_interval,
        "epoch_offset": args.epoch_offset, "global_epoch_range": [args.epoch_offset + 1, args.epoch_offset + args.epochs], "continuation_of": portable(args.prev_record.as_posix()) if args.prev_record else None,
        "previous_best_map50_95": prev_best, "early_stopping": {"plateau_min_improvement": args.plateau_delta, "plateau_points": args.plateau_points, "nan_inf": True, "architecture_change": True},
        "milestones": {"M1": 0.35, "M2": 0.40, "M3": 0.43369, "basis": "M3 = baseline 0.44369 - 1.0 pp PROJECT-DEFINED EXAMPLE threshold; reaching a milestone is not Release PASS"},
    }
    missing = validate_finetune_config(finetune_config)
    assert not missing, f"finetune config incomplete: {missing}"
    dump(report / "finetune_config.json", finetune_config)
    print(f"[config] phase {args.phase} epochs {args.epochs} batch {args.batch} lr {args.lr} | {iters_per_epoch} iters/epoch | train images {len(dataset)} (train==val: {finetune_config['dataset']['train_equals_val']})")

    # ---------------- 4. evaluation helper (E5-1 conditions) ----------------
    val_kwargs = dict(data=args.data, imgsz=args.imgsz, batch=8, device="cpu", conf=0.001, iou=0.7, plots=False, verbose=False, workers=0, rect=True, mode="val", task="detect", half=False,
                      project=(report / "ultralytics_runs").as_posix())

    def evaluate(body_, label: str, name: str) -> DetectionMetrics:
        det = detector_cls(deepcopy(body_).eval(), head_cls(meta_raw, DFL, make_anchors, dist2bbox).eval(), names, meta_raw["stride"]).eval()
        try:
            validator = DetectionValidator(args={**val_kwargs, "model": "e5_1_finetune_body_plus_head", "name": name})
            validator(model=det)
            box = validator.metrics.box
            return DetectionMetrics(status="MEASURED", map50_95=round(float(box.map), 5), map50=round(float(box.map50), 5), map75=round(float(box.map75), 5), precision=round(float(box.mp), 5),
                                    recall=round(float(box.mr), 5), dataset=args.data, dataset_version="ultralytics coco128.zip (COCO 2017 subset)", split="val(train2017 subset)", images=128,
                                    instances=929, imgsz=args.imgsz, conf=0.001, iou=0.7, evaluator=evaluator, model_label=label)
        except Exception as exc:  # noqa: BLE001
            return DetectionMetrics.not_available(f"validation failed: {type(exc).__name__}: {str(exc)[:300]}", dataset=args.data, imgsz=args.imgsz, evaluator=evaluator, model_label=label)

    # ---------------- 5. training loop (global epochs, milestones, best checkpoint, early stopping) ----------------
    baseline_map, compressed_map = e5["accuracy"]["baseline"]["metrics"]["mAP50-95"], e5["accuracy"]["candidate"]["metrics"]["mAP50-95"]
    history: list[dict] = []
    validation_points: list[dict] = []
    best = {"map50_95": None, "epoch": None, "global_epoch": None, "state": None}
    milestones_log: list[dict] = []
    stopped_reason: str | None = None
    t_train0 = time.perf_counter()
    it = 0
    for epoch in range(1, args.epochs + 1):
        gepoch = args.epoch_offset + epoch
        set_modes()
        sums = {"box_loss": 0.0, "cls_loss": 0.0, "dfl_loss": 0.0}
        t_ep = time.perf_counter()
        for batch in loader:
            for g in optimizer.param_groups:
                g["lr"] = lr_at(it)
            batch["img"] = batch["img"].float() / 255
            preds = train_model(batch["img"])
            loss, items = criterion(preds, batch)
            total = loss.sum()  # v8DetectionLoss returns (box, cls, dfl) * batch_size; the trainer sums it
            if not torch.isfinite(total):
                stopped_reason = f"non-finite loss at global epoch {gepoch} iteration {it}"
                break
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_([p for _, p in all_params if p.requires_grad], max_norm=10.0)
            optimizer.step()
            for k in sums:
                sums[k] += float(items[k])
            it += 1
        if stopped_reason:
            print(f"[early stop] {stopped_reason}")
            break
        rec = {"epoch": epoch, "global_epoch": gepoch, "lr_end": lr_at(it - 1), "seconds": round(time.perf_counter() - t_ep, 1), **{k: round(v / iters_per_epoch, 4) for k, v in sums.items()}}
        if epoch % args.val_interval == 0 or epoch == args.epochs:
            m = evaluate(body, f"phase {args.phase} global epoch {gepoch}", f"gep{gepoch:03d}")
            rec.update({"map50_95": m.map50_95, "map50": m.map50, "map75": m.map75, "precision": m.precision, "recall": m.recall})
            if count_torch_parameters(body) != params_start:
                stopped_reason = f"parameter count changed ({count_torch_parameters(body)} != {params_start})"
            cm = continuation_metrics(current=m.map50_95, previous_best=prev_best, baseline=baseline_map, compressed=compressed_map)
            improved = m.map50_95 is not None and (best["map50_95"] is None or m.map50_95 > best["map50_95"])
            if improved:
                best = {"map50_95": m.map50_95, "epoch": epoch, "global_epoch": gepoch, "state": deepcopy(body.state_dict())}
            for name, hit in milestones_reached(m.map50_95).items():
                if hit and name not in {x["milestone"] for x in milestones_log}:
                    milestones_log.append({"milestone": name, "threshold": {"M1": 0.35, "M2": 0.40, "M3": 0.43369}[name], "global_epoch": gepoch, "map50_95": m.map50_95})
                    print(f"[milestone] {name} reached at global epoch {gepoch}: mAP50-95 {m.map50_95} (not a Release verdict)")
            validation_points.append({"epoch": epoch, "global_epoch": gepoch, "map50_95": m.map50_95, "map50": m.map50, "map75": m.map75, "precision": m.precision, "recall": m.recall,
                                      "box_loss": rec["box_loss"], "cls_loss": rec["cls_loss"], "dfl_loss": rec["dfl_loss"], "continuation": cm, "is_best": improved})
            plateau = plateau_detected([v["map50_95"] for v in validation_points], min_improvement=args.plateau_delta, points=args.plateau_points)
            rec["plateau_check"] = plateau
            if plateau["plateau"] and not stopped_reason:
                stopped_reason = "plateau"
        history.append(rec)
        print(f"[epoch {epoch}/{args.epochs} | global {gepoch}] box {rec['box_loss']} cls {rec['cls_loss']} dfl {rec['dfl_loss']} | {rec['seconds']} s"
              + (f" | mAP50-95 {rec.get('map50_95')} mAP50 {rec.get('map50')} (best {best['map50_95']})" if "map50_95" in rec else ""))
        if stopped_reason:
            print(f"[early stop] {stopped_reason} at global epoch {gepoch}")
            break
    train_seconds = round(time.perf_counter() - t_train0, 1)
    epochs_run = len(history)
    body.eval()
    final_map = history[-1].get("map50_95") if history else None
    rollback = {"applied": False, "reason": None}
    if best["state"] is not None and final_map is not None and best["map50_95"] is not None and final_map < best["map50_95"]:
        body.load_state_dict(best["state"])
        rollback = {"applied": True, "reason": f"final epoch mAP50-95 {final_map} < best {best['map50_95']} (global epoch {best['global_epoch']}); best checkpoint restored (CASE E)"}
        print(f"[rollback] {rollback['reason']}")
    best.pop("state", None)
    for _, p in all_params:
        p.requires_grad_(False)

    # ---------------- 6. save the NEW local artifact (E5-1 files untouched) ----------------
    out_pt = args.out_dir / f"model_fx_phase{args.phase}.pt"
    torch.save(body, out_pt.as_posix())
    shutil.copyfile(head_meta_path, args.out_dir / "netspresso_head_meta.json")
    ft_sha = compute_sha256(out_pt)
    e5_files_sha_after = {p.name: compute_sha256(p) for p in (E5_RECORD, E5_RUN / "execution_result.json", args.candidate)}
    e5_immutable = e5_files_sha_before == e5_files_sha_after
    manifest = {"artifact": portable(out_pt.as_posix()), "sha256": ft_sha, "size_bytes": out_pt.stat().st_size, "kind": "LOCALLY FINE-TUNED derivative of the E5-1 NetsPresso candidate (not a NetsPresso artifact)",
                "source_artifact_sha256": start_sha, "e5_1_candidate_sha256": CANDIDATE_SHA, "r1_input_sha256": R1_SHA, "phase": args.phase, "config": portable((report / "finetune_config.json").as_posix()),
                "timestamp": datetime.now(UTC).replace(microsecond=0).isoformat(), "training_seconds": train_seconds}
    dump(args.out_dir / f"manifest_phase{args.phase}.json", manifest)
    print(f"[artifact] {portable(out_pt.as_posix())} sha {ft_sha[:16]} {out_pt.stat().st_size} B | training {train_seconds} s | E5-1 files immutable: {e5_immutable}")

    # ---------------- 7. output validation: load, structure, shapes, params, FLOPs ----------------
    ft_body = torch.load(out_pt.as_posix(), map_location="cpu", weights_only=False).eval()
    sv = validate_pt(out_pt, allow_full_unpickle=True, torch_module=torch).to_dict()  # generated locally in this run
    with torch.no_grad():
        ft_maps = ft_body(torch.zeros(*input_shape))
        ft_det = detector_cls(ft_body, head_cls(meta_raw, DFL, make_anchors, dist2bbox).eval(), names, meta_raw["stride"]).eval()
        y_ft, _ = ft_det(torch.zeros(*input_shape))
    ok_maps2, _ = validate_output_schema([list(m.shape) for m in ft_maps], expected_body_output_shapes(input_shape, head_meta))
    ok_dec2, _ = validate_output_schema([list(y_ft.shape)], [expected_decoded_output_shape(input_shape, head_meta)])
    exports = ROOT / "outputs" / "e5_1_finetune_exports" / report.name
    exports.mkdir(parents=True, exist_ok=True)

    def export_body(module, path: Path) -> dict:
        try:
            torch.onnx.export(module, torch.zeros(*input_shape), path.as_posix(), opset_version=13, input_names=["images"], do_constant_folding=True, dynamo=False)
            m = onnx.load(path.as_posix())
            macs, cov, uncov, err = estimate_onnx_macs(m, onnx)
            return {"status": "PASS", "path": portable(path.as_posix()), "params_onnx": count_onnx_parameters(m), "macs": macs, "flops_2x_macs": 2 * macs if macs else None, "covered_ops": cov, "uncovered_ops": uncov, "error": err}
        except Exception as exc:  # noqa: BLE001
            return {"status": "FAIL", "error": f"{type(exc).__name__}: {str(exc)[:300]}"}

    ft_onnx = export_body(ft_body, exports / "finetuned_body.onnx")
    e5_macs = e5["model_metrics"]["onnx_exports"]["candidate_body"].get("macs")
    arch = architecture_unchanged(params_before=882996, params_after=count_torch_parameters(ft_body), macs_before=e5_macs, macs_after=ft_onnx.get("macs"))
    artifact_validation = {"pt": {"path": portable(out_pt.as_posix()), "sha256": ft_sha, "size_bytes": out_pt.stat().st_size, "structural": sv, "trusted_basis": "generated locally by this run (full unpickle allowed in-process); not in configs trust list",
                                  "forward_output_shapes": [list(m.shape) for m in ft_maps], "schema_ok": ok_maps2, "decoded_output_shape": list(y_ft.shape), "decoded_ok": ok_dec2},
                           "params": {"e5_1_candidate": 882996, "fine_tuned": count_torch_parameters(ft_body), "unchanged": count_torch_parameters(ft_body) == 882996},
                           "flops": {"e5_1_candidate_sdk": e5["model_metrics"]["flops"]["candidate_sdk"], "e5_1_candidate_independent_2x_macs": e5["model_metrics"]["flops"]["candidate_independent_2x_macs"],
                                     "fine_tuned_independent_2x_macs": ft_onnx.get("flops_2x_macs"), "onnx_export": ft_onnx},
                           "architecture_unchanged": arch,
                           "status": "PASS" if (sv["status"] == "PASS" and sv["details"].get("object_kind") == "graph_module" and ok_maps2 and ok_dec2 and arch["status"] == "PASS") else "FAIL"}
    print(f"[validation] structural {sv['status']} maps ok={ok_maps2} decoded ok={ok_dec2} | params {artifact_validation['params']['fine_tuned']} | 2xMACs {ft_onnx.get('flops_2x_macs')} | architecture {arch['status']} -> {artifact_validation['status']}")

    # ---------------- 8. three-way accuracy ----------------
    baseline = metrics_from_record(e5["accuracy"]["baseline"])
    compressed = metrics_from_record(e5["accuracy"]["candidate"])
    fine_tuned = evaluate(ft_body, f"LOCALLY fine-tuned E5-1 candidate (phase {args.phase}, {args.epochs} epochs)", "final")
    three = three_way_comparison(baseline=baseline.map50_95, compressed=compressed.map50_95, fine_tuned=fine_tuned.map50_95)
    cmp_ft = compare_detection_accuracy(baseline, fine_tuned)
    gate_acc = config.quality_gate.profiles["release"]["accuracy"]
    recovery = classify_recovery(three, max_drop_pp=gate_acc.threshold or 1.0, structural_ok=artifact_validation["status"] == "PASS", evaluation_ok=fine_tuned.status == "MEASURED")
    secondary = {k: {"baseline": getattr(baseline, k), "compressed": getattr(compressed, k), "fine_tuned": getattr(fine_tuned, k)} for k in ("map50", "map75", "precision", "recall")}
    print(f"[accuracy] baseline {baseline.map50_95} | compressed {compressed.map50_95} | fine-tuned {fine_tuned.map50_95} | recovery {three.get('recovery_percent')} % | remaining drop {three.get('remaining_drop_pp')} pp -> {recovery['decision']}")

    # ---------------- 9. proxy + ORT (baseline R1 body+head vs fine-tuned; E5-1 vs fine-tuned) ----------------
    r1_body = torch.load((R1_DIR / "model_fx.pt").as_posix(), map_location="cpu", weights_only=False).eval()
    e5_body = torch.load(args.candidate.as_posix(), map_location="cpu", weights_only=False).eval()
    mk = lambda b: detector_cls(b, head_cls(meta_raw, DFL, make_anchors, dist2bbox).eval(), names, meta_raw["stride"]).eval()  # noqa: E731
    det_r1, det_e5 = mk(r1_body), mk(e5_body)
    img_dir = DATASETS / "coco128" / "images" / "train2017"
    imgs = sorted(img_dir.glob("*.jpg"))[: args.images]
    rep_img = img_dir / "000000000009.jpg"
    if rep_img.is_file() and rep_img not in imgs:
        imgs.append(rep_img)
    det_prev = mk(start_body) if args.init_body is not None else None
    per_image = []
    with torch.no_grad():
        for p in imgs:
            x, _ = preprocess_image(p, args.imgsz, LetterBox, np, torch)
            y_r1, _ = det_r1(x)
            y_e5, _ = det_e5(x)
            y_f, _ = ft_det(x)
            rec_img = {"image": p.name, "baseline_vs_fine_tuned": stats_of(y_r1, y_f), "compressed_vs_fine_tuned": stats_of(y_e5, y_f), "baseline_vs_compressed": stats_of(y_r1, y_e5)}
            if det_prev is not None:
                y_p, _ = det_prev(x)
                rec_img["previous_vs_fine_tuned"] = stats_of(y_p, y_f)
            per_image.append(rec_img)
    agg = {k: {"min_cosine": min(r[k]["cosine"] for r in per_image), "max_abs_diff": max(r[k]["max_abs_diff"] for r in per_image), "max_relative_l2_diff": max(r[k]["relative_l2_diff"] for r in per_image)}
           for k in per_image[0] if k != "image"}
    print(f"[proxy real images] baseline vs fine-tuned cosine {agg['baseline_vs_fine_tuned']['min_cosine']:.6f} | compressed vs fine-tuned {agg['compressed_vs_fine_tuned']['min_cosine']:.6f} | baseline vs compressed {agg['baseline_vs_compressed']['min_cosine']:.6f}"
          + (f" | previous vs fine-tuned {agg['previous_vs_fine_tuned']['min_cosine']:.6f}" if "previous_vs_fine_tuned" in agg else ""))

    class OnlyY(torch.nn.Module):
        def __init__(self, det):
            super().__init__()
            self.det = det

        def forward(self, x):
            return self.det(x)[0]

    perf: dict = {"baseline": None, "fine_tuned": None, "note": "local onnxruntime CPUExecutionProvider, development machine, process-isolated per model (fresh interpreter), same session for both models; not target-device performance",
                  "e5_1_reference": e5["performance"]}
    proxy_ort: dict = {"status": "NOT_APPLICABLE", "reason": "--skip-ort or export failed"}
    base_run = ft_run = None
    if not args.skip_ort:
        full = {}
        targets = [("baseline", det_r1), ("e5_1", det_e5), ("fine_tuned", ft_det)] + ([("previous", det_prev)] if det_prev is not None else [])
        for label, det in targets:
            path = exports / f"{label}_body_plus_head.onnx"
            try:
                torch.onnx.export(OnlyY(det).eval(), torch.zeros(*input_shape), path.as_posix(), opset_version=13, input_names=["images"], output_names=["output0"], do_constant_folding=True, dynamo=False)
                full[label] = {"status": "PASS", "path": portable(path.as_posix()), "sha256": compute_sha256(path), "structural": validate_onnx(path, onnx_module=onnx).to_dict()}
            except Exception as exc:  # noqa: BLE001
                full[label] = {"status": "FAIL", "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        perf["onnx_exports"] = full
        if all(v["status"] == "PASS" for v in full.values()):
            measure = dict(warmup=policy.warmup_iterations, iterations=args.iterations or policy.measurement_iterations, seed=policy.seed, threads=policy.intra_op_threads, provider=policy.execution_provider, input_shape=input_shape)
            runs = {label: measure_in_subprocess(exports / f"{label}_body_plus_head.onnx", **measure) for label, _ in targets}
            base_run, ft_run = runs["baseline"], runs["fine_tuned"]
            for label, run in runs.items():
                perf[label] = {"latency": run.latency.to_dict(), "memory": run.memory.to_dict(), "output_shapes": run.output_shapes}
                print(f"[ort {label}] median {run.latency.median_ms} ms p95 {run.latency.p95_ms} ms | RSS delta {run.memory.delta_mb} MB")
            pr = compare_outputs(base_run.outputs, ft_run.outputs, baseline_shapes=base_run.output_shapes, optimized_shapes=ft_run.output_shapes, baseline_dtypes=base_run.output_dtypes,
                                 optimized_dtypes=ft_run.output_dtypes, min_cosine=policy.proxy_min_cosine_similarity)
            proxy_ort = {**pr.to_dict(), "num_outputs": len(base_run.outputs)}
            print(f"[proxy ort] {pr.status}: min cosine {pr.min_cosine_similarity} max|d| {pr.max_abs_diff} rel {pr.relative_diff}")
    proxy = {"ort_baseline_vs_fine_tuned": proxy_ort, "real_images": {"images": len(imgs), "aggregate": agg, "per_image": per_image}, "e5_1_reference_baseline_vs_compressed": e5["output_equivalence_proxy"]["ort"],
             "disclaimer": PROXY_DISCLAIMER + " Proxy values never replace the measured mAP above."}

    # ---------------- 10. Result Model, gates, defects, recovery assessment ----------------
    env_fp = environment_fingerprint(execution_provider=policy.execution_provider, threads=policy.intra_op_threads, packages=("ultralytics", "torch", "onnx", "onnxruntime", "numpy", "psutil"))
    environment = Environment(adapter="local_e5_1_finetune", adapter_version="0.1.0", sdk_name="ultralytics+torch (local training)", sdk_version=ultralytics.__version__, python_version=env_fp["python"],
                              platform=f"{env_fp['os']} {env_fp['os_release']} {env_fp['architecture']}", extra={"credit_consuming": False, "netspresso_api_calls": 0, "evaluation_type": "local_finetune", **env_fp})
    cfg_run = Configuration.from_dict(e5["experiment"]["configuration"])

    def metrics_for(acc: DetectionMetrics, run, size_path: Path, extra: dict) -> Metrics:
        return Metrics(accuracy=acc.map50_95 if acc.status == "MEASURED" else None, latency_ms=run.latency.median_ms if run else None,
                       memory_mb=run.memory.delta_mb if run and run.memory.status == "PASS" else None, model_size_mb=round(size_path.stat().st_size / 1_000_000, 3), extra=extra)

    ft_extra = {"params_torch": float(artifact_validation["params"]["fine_tuned"]), "training_epochs": float(epochs_run), "training_seconds": float(train_seconds)}
    if ft_onnx.get("macs"):
        ft_extra["macs_estimate"] = float(ft_onnx["macs"])
    if proxy_ort.get("status") not in (None, "NOT_APPLICABLE"):
        ft_extra.update({"proxy_min_cosine_similarity": proxy_ort.get("min_cosine_similarity") or 0.0, "proxy_max_abs_diff": proxy_ort.get("max_abs_diff") or 0.0, "proxy_relative_diff": proxy_ort.get("relative_diff") or 0.0,
                         "proxy_shape_match": 1.0 if proxy_ort.get("shape_match") else 0.0, "proxy_dtype_match": 1.0 if proxy_ort.get("dtype_match") else 0.0})
    base_extra = {"params_torch": float(count_torch_parameters(r1_body))}
    artifact = artifact_from_file(out_pt, expected_sha256=None, source_model="E5-1 sdk_output.pt (NetsPresso candidate)", source_model_sha256=CANDIDATE_SHA, operation="local_finetune",
                                  structural_validation=sv, configuration_key=cfg_run.key, evaluation="local_finetune", generated_by="scripts/yolov8_e5_1_finetune.py", phase=args.phase)
    artifact.path = portable(artifact.path or "")
    execution = ExecutionResult(
        operation="local_finetune_recovery", configuration=cfg_run, execution_status=ExecutionStatus.COMPLETED, environment=environment,
        baseline_metrics=metrics_for(baseline, base_run, R1_DIR / "model_fx.pt", base_extra), metrics=metrics_for(fine_tuned, ft_run, out_pt, ft_extra), artifact=artifact,
        logs=[DISCLAIMER, f"start body sha256={start_sha}", f"fine-tuned sha256={ft_sha}", f"phase {args.phase}, {args.epochs} epochs, {train_seconds} s"],
        reproducibility=Reproducibility(level=ReproducibilityLevel.NOT_VERIFIED, runs=1, checksums=[ft_sha], notes="single local fine-tuning run (seeded, deterministic flags); a second run was not performed"),
        timestamp=utc_now_iso(), credit=CreditRecord(usage_type=CreditUsageType.NONE, estimated=0, actual=None, note="local fine-tuning: 0 credits, no API call"))
    gates = {name: QualityGate(config.quality_gate, profile=name).evaluate(execution) for name in ("compression", "local_eval", "release")}
    defects = {name: (None if (d := classify(execution, g)) is None else {"defect_id": f"E5-1-FT-{args.phase}-{name}-{d.category.value}", **d.to_dict()}) for name, g in gates.items()}
    for g in gates.values():
        print("\n" + render_text(g))
    e5_defects = {k: (v["defect_id"] if v else None) for k, v in e5["defects"]["by_profile"].items()}
    recovery_assessment = {"historical_e5_1_defects": e5_defects, "historical_records_modified": False,
                           "assessment": ("E5-1 ACCURACY_REGRESSION recovered by subsequent LOCAL fine-tuning on the smoke set (release evidence still pending)" if recovery["decision"] == "RECOVERY_SUCCESS"
                                          else "E5-1 ACCURACY_REGRESSION partially mitigated by local fine-tuning; regression persists" if recovery["decision"] == "PARTIAL_RECOVERY"
                                          else "E5-1 ACCURACY_REGRESSION NOT recovered by this local fine-tuning run"),
                           "recovery_decision": recovery, "root_cause_status": "pruning-without-fine-tuning hypothesis " + ("SUPPORTED by recovery evidence" if recovery["decision"] in ("RECOVERY_SUCCESS", "PARTIAL_RECOVERY") else "NOT supported by this run")}
    case = CaseResult(configuration=cfg_run, status=case_status_for(execution, gates["release"]), support=SupportState.SUPPORTED, selected=True, selection_reason="Phase 5-E local fine-tuning recovery",
                      execution=execution, gate=gates["release"], defect=classify(execution, gates["release"]))

    # ---------------- 11. registry, traceability, recommendation, report ----------------
    registry = BaselineRegistry(ROOT / "reports" / "baselines" / "registry.json")
    entry = registry.register_candidate(BaselineEntry(
        artifact_id=make_artifact_id("local_finetune", ft_sha), model_name=cfg_run.model, source_artifact=f"E5-1 sdk_output.pt sha256={CANDIDATE_SHA} (via {portable(start_path.as_posix())} sha256={start_sha})",
        sha256=ft_sha, file_size=out_pt.stat().st_size, format="pt", sdk_version=None, operation="local_finetune", configuration=cfg_run.to_dict(), input_shape=input_shape, compression_ratio=0.5,
        environment_fingerprint=environment.to_dict(), created_at=utc_now_iso(), status="candidate",
        provenance={"kind": "LOCAL fine-tuned derivative (not a NetsPresso artifact)", "phase": args.phase, "config": portable((report / "finetune_config.json").as_posix()), "registered_by": "scripts/yolov8_e5_1_finetune.py"},
        notes=["candidate only; single local run; not a NetsPresso output"]))
    registry.save()
    # previous phases: follow the record chain (C -> B -> A) so the multi-way table and traceability list every local step
    phase_a_ref = None
    previous_phases: list[dict] = []
    rec_path = args.prev_record
    while rec_path is not None and Path(rec_path).is_file():
        pr = json.loads(Path(rec_path).read_text(encoding="utf-8"))
        ref = {"phase": pr["finetune_config"]["phase"], "record": portable(Path(rec_path).as_posix()), "fine_tuned_sha256": pr["artifact_validation"]["pt"]["sha256"], "size_bytes": pr["artifact_validation"]["pt"]["size_bytes"],
               "map50_95": pr["accuracy"]["fine_tuned"]["metrics"]["mAP50-95"], "epochs": pr["training"].get("epochs_run", pr["training"]["epochs"]), "global_epoch_end": pr["finetune_config"].get("global_epoch_range", [None, pr["training"]["epochs"]])[1],
               "metrics": pr["accuracy"]["fine_tuned"]["metrics"], "params": pr["artifact_validation"]["params"]["fine_tuned"], "flops_2x_macs": pr["artifact_validation"]["flops"]["fine_tuned_independent_2x_macs"]}
        previous_phases.insert(0, ref)
        if phase_a_ref is None:
            phase_a_ref = ref  # immediate predecessor (kept under the historical key name for record compatibility)
        nxt = (pr["experiment"].get("prev_reference") or pr["experiment"].get("phase_a_reference") or {}).get("record")
        rec_path = (ROOT / nxt) if nxt else None
    comparison_table = [
        {"model": "baseline", "kind": "upstream YOLOv8n (untouched)", "metrics": e5["accuracy"]["baseline"]["metrics"], "params": count_torch_parameters(r1_body), "flops_2x_macs": e5["model_metrics"]["flops"]["baseline_independent_2x_macs"],
         "size_bytes": (R1_DIR / "model_fx.pt").stat().st_size, "sha256": R1_SHA},
        {"model": "E5-1 compressed", "kind": "NetsPresso-generated candidate", "metrics": e5["accuracy"]["candidate"]["metrics"], "params": 882996, "flops_2x_macs": e5["model_metrics"]["flops"]["candidate_independent_2x_macs"],
         "size_bytes": 3693813, "sha256": CANDIDATE_SHA},
        *[{"model": f"fine-tuned {p['phase']}", "kind": f"LOCAL derivative (global epochs ..{p['global_epoch_end']})", "metrics": p["metrics"], "params": p["params"], "flops_2x_macs": p["flops_2x_macs"], "size_bytes": p["size_bytes"], "sha256": p["fine_tuned_sha256"]}
          for p in previous_phases],
        {"model": f"fine-tuned {args.phase}", "kind": f"LOCAL derivative (global epochs {args.epoch_offset + 1}..{args.epoch_offset + epochs_run})", "metrics": fine_tuned.to_dict().get("metrics"), "params": artifact_validation["params"]["fine_tuned"],
         "flops_2x_macs": ft_onnx.get("flops_2x_macs"), "size_bytes": out_pt.stat().st_size, "sha256": ft_sha},
    ]
    plateau_final = plateau_detected([v["map50_95"] for v in validation_points], min_improvement=args.plateau_delta, points=args.plateau_points)
    phase_c = phase_c_decision(best_value=best["map50_95"], previous_best=prev_best if prev_best is not None else 0.0, stopped_reason=stopped_reason or "completed", plateau=plateau_final["plateau"],
                               final_dropped_below_best=rollback["applied"], structural_ok=artifact_validation["status"] == "PASS")
    improvement_from_prev = None if prev_best is None or best["map50_95"] is None else round(best["map50_95"] - prev_best, 5)
    phase_c_next = phase_c_next_step(phase_c, improvement_from_prev=improvement_from_prev)
    print(f"[continuation] best {best['map50_95']} (global epoch {best['global_epoch']}) vs previous best {prev_best} -> {phase_c['decision']} ({phase_c['reason']}) | next: {phase_c_next['label']} | NetsPresso: {phase_c_next['netspresso']}")
    chain = build_traceability_chain(
        source_model=f"yolov8n.pt sha256={e5['input_provenance']['source_model']['sha256']}", r1_fx_artifact="outputs/models/yolov8n_fx_r1/model_fx.pt (patch r1-upstream-body-wrapper-v1)", r1_sha256=R1_SHA,
        netspresso_execution=f"{E5_RUN.name}: automatic_compression ratio 0.5 (E5-1, NetsPresso-generated)", candidate_artifact=portable(args.candidate.as_posix()), candidate_sha256=CANDIDATE_SHA,
        local_validation=f"LOCAL fine-tuning phase {args.phase} ({epochs_run} epochs = global {args.epoch_offset + 1}..{args.epoch_offset + epochs_run}, seed {args.seed}) -> {portable(out_pt.as_posix())} sha256={ft_sha}"
                         + ("; via " + " <- ".join(f"phase {p['phase']} {p['fine_tuned_sha256'][:16]}" for p in reversed(previous_phases)) if previous_phases else ""),
        accuracy_result=f"COCO128 mAP50-95 baseline {three.get('baseline')} / compressed {three.get('compressed')} / fine-tuned {three.get('fine_tuned')} (recovery {three.get('recovery_percent')} %)",
        quality_gate={k: g.overall.value for k, g in gates.items()}, credit_record=f"ledger unchanged: {ledger_state()[:3]} sha {ledger_sha[:16]} (0 credits)")
    next_step = recommend_next_step(recovery["decision"], three, structural_ok=artifact_validation["status"] == "PASS")
    used2, remaining2, ops2, ledger_sha2 = ledger_state()
    result = {
        "experiment": {"phase": "5-E fine-tuning recovery", "finetune_phase": args.phase, "timestamp": datetime.now(UTC).replace(microsecond=0).isoformat(), "model": cfg_run.model, "report_dir": portable(report.as_posix()),
                       "e5_1_record": portable(E5_RECORD.as_posix()), "e5_1_run_dir": portable(E5_RUN.as_posix()), "phase_a_reference": phase_a_ref, "prev_reference": phase_a_ref, "previous_phases": previous_phases},
        "disclaimer": DISCLAIMER, "credit_consuming": False, "netspresso_api_calls": 0,
        "credit_safety": {"ledger_before": {"used": used, "remaining": remaining, "operations": ops, "sha256": ledger_sha}, "ledger_after": {"used": used2, "remaining": remaining2, "operations": ops2, "sha256": ledger_sha2},
                          "ledger_unchanged": ledger_sha == ledger_sha2, "api_calls": 0, "sdk_execution": 0},
        "e5_1_verification": {"checks": e5_checks, "candidate_identity": identity, "e5_1_files_immutable": e5_immutable, "files_sha256": e5_files_sha_after,
                              "e5_1_baseline": e5["accuracy"]["baseline"], "e5_1_candidate": e5["accuracy"]["candidate"], "e5_1_candidate_params": 882996},
        "trainability": trainability, "finetune_config": finetune_config,
        "training": {"epochs": args.epochs, "epochs_run": epochs_run, "epoch_offset": args.epoch_offset, "global_epoch_range": [args.epoch_offset + 1, args.epoch_offset + epochs_run], "history": history,
                     "validation_points": validation_points, "trend": epoch_trend(history), "seconds": train_seconds, "best": best, "rollback": rollback, "milestones_reached": milestones_log,
                     "early_stopping": {"stopped_reason": stopped_reason or "completed", "plateau_check_final": plateau_final}, "previous_best_map50_95": prev_best,
                     "final_losses": {k: history[-1][k] for k in ("box_loss", "cls_loss", "dfl_loss")} if history else None},
        "comparison_table": comparison_table, "continuation_decision": phase_c, "continuation_next_step": phase_c_next, "netspresso_next_execution": phase_c_next["netspresso"],
        "artifact_validation": artifact_validation,
        "accuracy": {"baseline": baseline.to_dict(), "compressed": compressed.to_dict(), "fine_tuned": fine_tuned.to_dict(), "three_way": three, "secondary": secondary, "fine_tuned_vs_baseline": cmp_ft.to_dict(),
                     "threshold_pp": gate_acc.threshold, "threshold_basis": "PROJECT-DEFINED EXAMPLE THRESHOLD (configs/quality_gate.yaml release.accuracy); COCO128 smoke -> diagnostic only, Release PASS is never declared from it",
                     "conditions": {"data": args.data, "imgsz": args.imgsz, "conf": 0.001, "iou": 0.7, "rect": True, "batch": 8, "evaluator": evaluator, "train_equals_val": finetune_config["dataset"]["train_equals_val"]}},
        "recovery": recovery, "recovery_assessment": recovery_assessment, "output_equivalence_proxy": proxy, "performance": perf,
        "quality_gate": {k: g.to_dict() for k, g in gates.items()}, "defects": defects,
        "reproducibility": {"training_runs": 1, "seed": args.seed, "deterministic_flags": True, "level": execution.reproducibility.level.value, "source_artifact_sha256": start_sha, "output_artifact_sha256": ft_sha,
                            "registry_entry": entry.to_dict(), "note": "one seeded local run; bitwise reproducibility of CPU training was not verified (would need a second run)"},
        "traceability": chain, "next_step": next_step, "case_result": case.to_dict(), "environment": environment.to_dict(),
    }
    result = json.loads(portable(json.dumps(result, default=str)))
    dump(report / "finetune_result.json", result)
    dump(report / "finetune_accuracy.json", {"credit_consuming": False, "disclaimer": DISCLAIMER, **result["accuracy"], "recovery": recovery})
    dump(report / "training_history.json", {"credit_consuming": False, "phase": args.phase, "epoch_offset": args.epoch_offset, "history": result["training"]["history"], "best": result["training"]["best"],
                                             "rollback": result["training"]["rollback"], "early_stopping": result["training"]["early_stopping"], "milestones_reached": result["training"]["milestones_reached"]})
    dump(report / "validation_results.json", {"credit_consuming": False, "conditions": result["accuracy"]["conditions"], "validation_points": result["training"]["validation_points"], "final": result["accuracy"]["fine_tuned"]})
    dump(report / "environment_fingerprint.json", {"credit_consuming": False, **result["environment"]})
    dump(report / "artifact_validation.json", {"credit_consuming": False, **result["artifact_validation"]})
    dump(report / "recovery_summary.json", {"credit_consuming": False, "disclaimer": DISCLAIMER, "three_way": result["accuracy"]["three_way"], "recovery": recovery, "recovery_assessment": result["recovery_assessment"],
                                             "comparison_table": result["comparison_table"], "continuation_decision": result["continuation_decision"], "continuation_next_step": result["continuation_next_step"],
                                             "netspresso_next_execution": result["netspresso_next_execution"], "credit_safety": result["credit_safety"], "traceability": result["traceability"]})
    dump(report / f"phase{args.phase}_result.json", result)
    try:
        from framework.reporter import write_finetune_html  # noqa: PLC0415

        write_finetune_html(result, report / "finetune_summary.html")
    except Exception as exc:  # noqa: BLE001
        print(f"[report] HTML skipped: {type(exc).__name__}: {exc}")
    assert not any(m == "netspresso" or m.startswith("netspresso.") for m in sys.modules), "SDK must never be imported here"
    print(f"\n[recovery] {recovery['decision']}: {recovery['reason']} | next step {next_step['code']} {next_step['label']}")
    print(f"ledger unchanged: {ledger_sha == ledger_sha2} ({used2}/{remaining2}/{ops2}) | NetsPresso API calls: 0 | record: {portable((report / 'finetune_result.json').as_posix())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

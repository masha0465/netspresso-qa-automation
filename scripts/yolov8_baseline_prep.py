#!/usr/bin/env python
"""Phase 5-E preparation: YOLOv8n BASELINE evidence with 0 NetsPresso credits.

Run from the isolated YOLO environment (ultralytics + onnxruntime; no netspresso SDK):

    .venv-yolo/Scripts/python scripts/yolov8_baseline_prep.py
    .venv-yolo/Scripts/python scripts/yolov8_baseline_prep.py --data coco128.yaml --iterations 100
    .venv-yolo/Scripts/python scripts/yolov8_baseline_prep.py --skip-val          # structure/perf only

Steps (all local): weights fingerprint -> structural validation (trusted full unpickle) ->
ONNX export (ultralytics) -> ONNX structural validation -> params / FLOPs
(implementation-reported vs independent) -> process-isolated ONNX Runtime latency + peak RSS ->
COCO-format validation (ultralytics `val`, mAP50-95 / mAP50 / precision / recall) ->
environment fingerprint -> reports/yolov8_baseline/<utc>/baseline.json (+ registry candidate).

It never imports the netspresso SDK and never reads NETSPRESSO_API_KEY. Model and dataset
binaries stay under gitignored directories (outputs/, datasets/).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from framework.baselines import BaselineEntry, BaselineRegistry, make_artifact_id  # noqa: E402
from framework.config import load_config  # noqa: E402
from framework.evaluation.artifact_structure import validate_onnx, validate_pt  # noqa: E402
from framework.evaluation.environment import environment_fingerprint  # noqa: E402
from framework.evaluation.local_ort import measure_in_subprocess  # noqa: E402
from framework.evaluation.model_stats import (  # noqa: E402
    count_onnx_parameters,
    estimate_onnx_macs,
    verify,
    verify_flops,
)
from framework.evaluation.onnx_checksum import normalized_onnx_checksum  # noqa: E402
from framework.validation.artifact import compute_sha256  # noqa: E402

DISCLAIMER = ("YOLOv8n baseline evidence measured locally. No NetsPresso API call was made. "
              "Latency/memory are development-machine numbers, not target-device or vendor benchmarks.")
LEDGER = ROOT / "reports" / "credit_usage.json"
OUTPUTS = ROOT / "outputs" / "models"
DATASETS = ROOT / "datasets"


def portable(text: str) -> str:
    root_fwd = ROOT.as_posix() + "/"
    root_back = str(ROOT) + "\\"
    return text.replace(root_back.replace("\\", "\\\\"), "").replace(root_back, "").replace(root_fwd, "")


def dump(path: Path, data: object) -> None:
    path.write_text(portable(json.dumps(data, indent=2, ensure_ascii=False, default=str)) + "\n", encoding="utf-8")


def na(reason: str) -> dict:
    return {"status": "N/A", "reason": reason}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, default=OUTPUTS / "yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--data", default="coco128.yaml", help="ultralytics dataset yaml (coco128.yaml smoke / coco.yaml release)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--skip-val", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--configs", type=Path, default=ROOT / "configs")
    args = ap.parse_args(argv)

    ledger_before = hashlib.sha256(LEDGER.read_bytes()).hexdigest()
    config = load_config(args.configs)
    policy = config.local_eval
    iterations = args.iterations or policy.measurement_iterations
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or (ROOT / "reports" / "yolov8_baseline" / stamp)
    out.mkdir(parents=True, exist_ok=True)
    print(DISCLAIMER)
    if not args.weights.is_file():
        print(f"ABORT: weights not found: {args.weights} (download the official asset into outputs/models/ first)")
        return 2

    import onnx  # noqa: PLC0415 - heavy, isolated env only
    import torch  # noqa: PLC0415
    from ultralytics import YOLO, settings  # noqa: PLC0415

    settings.update({"datasets_dir": DATASETS.as_posix(), "sync": False, "runs_dir": (out / "ultralytics_runs").as_posix()})
    DATASETS.mkdir(exist_ok=True)

    # ---------------- 1. weights fingerprint ----------------
    sha = compute_sha256(args.weights)
    prov_path = args.weights.with_suffix(".provenance.json")
    provenance = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.is_file() else {}
    fingerprint = {"path": portable(args.weights.as_posix()), "size_bytes": args.weights.stat().st_size, "sha256": sha,
                   "trusted_for_full_unpickle": policy.is_trusted(sha), "provenance": provenance}
    print(f"[weights] {args.weights.name} {fingerprint['size_bytes']} bytes sha256={sha[:16]}... trusted={fingerprint['trusted_for_full_unpickle']}")

    # ---------------- 2. structural validation (.pt) ----------------
    sv_pt = validate_pt(args.weights, allow_full_unpickle=policy.is_trusted(sha), torch_module=torch).to_dict()
    print(f"[structure] weights: {sv_pt['status']} - {sv_pt['message']}")

    # ---------------- 3. load with ultralytics, implementation-reported stats ----------------
    model = YOLO(args.weights.as_posix())
    nc = len(model.names)
    layers = params_impl = gflops_impl = None
    try:
        from ultralytics.utils.torch_utils import get_flops, get_num_params  # noqa: PLC0415

        params_impl = int(get_num_params(model.model))
        gflops_impl = float(get_flops(model.model, imgsz=args.imgsz))  # ultralytics GFLOPs convention (thop/profiler based)
        layers = len(list(model.model.modules()))
    except Exception as exc:  # noqa: BLE001 - optional helper; absence is recorded, never fabricated
        print(f"[impl] ultralytics torch_utils unavailable: {type(exc).__name__}")
    impl = {"ultralytics_version": __import__("ultralytics").__version__, "task": model.task, "num_classes": nc,
            "layers": layers, "params_reported": params_impl, "gflops_reported_at_640": gflops_impl,
            "note": "values reported by the ultralytics implementation (model.info); not NetsPresso SDK values"}
    print(f"[impl] task={model.task} nc={nc} params={params_impl} GFLOPs@640={gflops_impl} ultralytics={impl['ultralytics_version']}")

    # ---------------- 4. ONNX export + validation ----------------
    exported = Path(model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, simplify=False, dynamic=False, device="cpu"))
    onnx_path = OUTPUTS / f"yolov8n_{args.imgsz}_opset{args.opset}.onnx"
    shutil.move(exported.as_posix(), onnx_path.as_posix())
    onnx_sha = compute_sha256(onnx_path)
    onnx_checksums = normalized_onnx_checksum(onnx_path, onnx_module=onnx)  # raw + normalized (metadata-insensitive); raw is preserved
    sv_onnx = validate_onnx(onnx_path, onnx_module=onnx).to_dict()
    print(f"[structure] onnx: {sv_onnx['status']} - {sv_onnx['message']} | sha256={onnx_sha[:16]}...")

    # ---------------- 5. params / FLOPs independent ----------------
    onnx_model = onnx.load(onnx_path.as_posix())
    params_onnx = count_onnx_parameters(onnx_model)
    params_torch = int(sum(p.numel() for p in model.model.parameters()))
    macs, covered, uncovered, macs_err = estimate_onnx_macs(onnx_model, onnx)
    gflops_reported = float(gflops_impl) * 1e9 if gflops_impl else None
    verification = {
        "params_torch_vs_reported": verify("params", params_impl, params_torch, policy.params_tolerance_percent, "torch numel vs ultralytics model.info").to_dict(),
        "params_onnx_vs_torch": verify("params", params_torch, params_onnx, policy.params_tolerance_percent, "ONNX initializer elements vs torch numel (export may fold BN / add constants)").to_dict(),
        "flops_estimate_vs_reported": verify_flops(gflops_reported, macs, policy.flops_tolerance_percent, "ONNX MAC estimate (Conv/Gemm/MatMul) vs ultralytics GFLOPs").to_dict(),
        "macs_coverage": {"covered_ops": covered, "uncovered_ops": uncovered, "error": macs_err},
    }
    print(f"[verify] params torch={params_torch:,} onnx={params_onnx:,} | MACs est={macs} | flops status={verification['flops_estimate_vs_reported']['verification_status']}")

    # ---------------- 6. ORT latency / memory (isolated process) ----------------
    run = measure_in_subprocess(onnx_path, warmup=policy.warmup_iterations, iterations=iterations, seed=policy.seed,
                                threads=policy.intra_op_threads, provider=policy.execution_provider, input_shape=[1, 3, args.imgsz, args.imgsz])
    print(f"[ort] latency median {run.latency.median_ms} ms p95 {run.latency.p95_ms} ms | peak RSS delta {run.memory.delta_mb} MB | outputs {run.output_shapes} {run.output_dtypes}")

    # ---------------- 7. accuracy (ultralytics val) ----------------
    accuracy: dict
    if args.skip_val:
        accuracy = na("validation skipped (--skip-val)")
    else:
        try:
            res = model.val(data=args.data, imgsz=args.imgsz, batch=args.batch, device="cpu", plots=False, verbose=False, workers=0,
                            project=(out / "ultralytics_runs").as_posix(), name="val")
            box = res.box
            accuracy = {
                "status": "MEASURED",
                "dataset_yaml": args.data,
                "dataset_root": portable(DATASETS.as_posix()),
                "split": "val",
                "imgsz": args.imgsz,
                "metrics": {"mAP50-95": round(float(box.map), 5), "mAP50": round(float(box.map50), 5), "mAP75": round(float(box.map75), 5),
                            "precision": round(float(box.mp), 5), "recall": round(float(box.mr), 5)},
                "results_dict": {k: round(float(v), 5) for k, v in res.results_dict.items()},
                "speed_ms_per_image": {k: round(float(v), 3) for k, v in (res.speed or {}).items()},
                "conf_iou_settings": "ultralytics val defaults (conf=0.001, iou=0.7) unless overridden",
                "note": "smoke-scale dataset if coco128: not comparable to the published COCO val2017 figure" if "128" in args.data else "COCO val2017 scale",
            }
            print(f"[val] {args.data}: mAP50-95={accuracy['metrics']['mAP50-95']} mAP50={accuracy['metrics']['mAP50']} P={accuracy['metrics']['precision']} R={accuracy['metrics']['recall']}")
        except Exception as exc:  # noqa: BLE001
            accuracy = na(f"validation failed: {type(exc).__name__}: {str(exc)[:200]}")
            print(f"[val] {accuracy}")

    # ---------------- 8. environment / record ----------------
    env = environment_fingerprint(execution_provider=policy.execution_provider, threads=policy.intra_op_threads,
                                  packages=("ultralytics", "torch", "torchvision", "onnx", "onnxruntime", "onnxslim", "numpy", "opencv-python", "psutil"))
    record = {
        "credit_consuming": False,
        "netspresso_api_calls": 0,
        "disclaimer": DISCLAIMER,
        "model": {"name": "yolov8n", "task": model.task, "weights": fingerprint, "implementation": impl},
        "input_shape": [1, 3, args.imgsz, args.imgsz],
        "preprocessing": "ultralytics default letterbox to imgsz, RGB, /255 (val); ORT measurement uses uniform random float32 input (latency/memory only)",
        "artifacts": {
            "weights_pt": {"structural_validation": sv_pt},
            "onnx_export": {"path": portable(onnx_path.as_posix()), "size_bytes": onnx_path.stat().st_size, "sha256": onnx_sha, "checksums": onnx_checksums,
                            "opset": args.opset, "dynamic": False, "simplify": False, "structural_validation": sv_onnx},
        },
        "params_flops": {"params_torch": params_torch, "params_onnx_initializers": params_onnx, "macs_estimate": macs, **verification},
        "performance": {"source": "local_onnxruntime_cpu (isolated process)", "latency": run.latency.to_dict(), "memory": run.memory.to_dict(),
                        "gpu_memory": "N/A (no GPU)", "outputs": {"names": run.output_names, "shapes": run.output_shapes, "dtypes": run.output_dtypes}},
        "accuracy": accuracy,
        "seed": policy.seed,
        "environment": env,
        "timestamp": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    dump(out / "baseline.json", record)

    registry = BaselineRegistry(ROOT / "reports" / "baselines" / "registry.json")
    entry = registry.register_candidate(BaselineEntry(
        artifact_id=make_artifact_id("yolov8n_baseline_onnx_export", onnx_sha), model_name="yolov8n",
        source_artifact=f"yolov8n.pt sha256={sha}", sha256=onnx_sha, file_size=onnx_path.stat().st_size, format="onnx", sdk_version=None,
        operation="yolov8n_baseline_onnx_export",
        configuration={"model": "yolov8n", "device": "intel_xeon_w2233", "runtime": "onnxruntime", "backend": "default", "optimization": "none_baseline"},
        input_shape=[1, 3, args.imgsz, args.imgsz], compression_ratio=None, environment_fingerprint=env, created_at=record["timestamp"], status="candidate",
        provenance={"exporter": f"ultralytics {impl['ultralytics_version']}", "opset": args.opset, "weights_source": provenance.get("source"),
                    "normalized_sha256": onnx_checksums.get("normalized_sha256"), "normalization_policy": onnx_checksums.get("normalization_policy"),
                    "registered_by": "scripts/yolov8_baseline_prep.py"},
        notes=["baseline (un-optimized) reference artifact; candidate until a second identical export confirms reproducibility"],
    ))
    registry.save()

    assert not any(m == "netspresso" or m.startswith("netspresso.") for m in sys.modules), "SDK must never be imported here"
    ledger_after = hashlib.sha256(LEDGER.read_bytes()).hexdigest()
    print(f"\n[registry] {entry.artifact_id} status={entry.status}")
    print(f"ledger unchanged: {ledger_before == ledger_after} | NetsPresso API calls: 0 | credits consumed: 0")
    print(f"baseline record: {portable((out / 'baseline.json').as_posix())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Phase 5-E preparation: NetsPresso-compatibility PRECHECK for YOLOv8n (0 credits).

Runs the official (archived) Nota fork's `export_netspresso()` locally to produce the
exact artifacts the official ModelZoo-YOLOv8 workflow uploads to NetsPresso:

    model_fx.pt               torch.fx GraphModule (body traced in train mode)
    netspresso_head_meta.json nc / nl / anchors / stride / strides / inplace

and validates them with the framework's structural validator. Nothing is uploaded.

    .venv-yolo-nota/Scripts/python scripts/yolov8_fx_export_precheck.py
    .venv-yolo-nota/Scripts/python scripts/yolov8_fx_export_precheck.py --weights outputs/models/yolov8n.pt

This is TC-Y8-030 (compatibility precheck). It never imports the netspresso SDK.
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

from framework.evaluation.artifact_structure import validate_pt  # noqa: E402
from framework.evaluation.environment import environment_fingerprint  # noqa: E402
from framework.validation.artifact import compute_sha256  # noqa: E402

LEDGER = ROOT / "reports" / "credit_usage.json"
DISCLAIMER = "fx export precheck ran locally. No NetsPresso API call was made. READY means known contract checks passed, not server acceptance."
OUTPUTS = ROOT / "outputs" / "models" / "yolov8n_fx"


def portable(text: str) -> str:
    root_fwd = ROOT.as_posix() + "/"
    root_back = str(ROOT) + "\\"
    return text.replace(root_back.replace("\\", "\\\\"), "").replace(root_back, "").replace(root_fwd, "")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, default=ROOT / "outputs" / "models" / "yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if not args.weights.is_file():
        print(f"ABORT: weights not found: {args.weights}")
        return 2

    ledger_before = hashlib.sha256(LEDGER.read_bytes()).hexdigest()
    print(DISCLAIMER)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or (ROOT / "reports" / "yolov8_baseline" / f"{stamp}_fx_precheck")
    out.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    record: dict = {"credit_consuming": False, "netspresso_api_calls": 0, "disclaimer": DISCLAIMER, "test_case": "TC-Y8-030 NetsPresso compatibility precheck",
                    "weights": {"path": portable(args.weights.as_posix()), "sha256": compute_sha256(args.weights), "size_bytes": args.weights.stat().st_size}}

    import torch  # noqa: PLC0415
    import ultralytics  # noqa: PLC0415
    from ultralytics import YOLO  # noqa: PLC0415

    record["exporter"] = {"package": "ultralytics_nota (Nota-NetsPresso fork, archived 2024-02)", "version": ultralytics.__version__,
                          "torch": torch.__version__, "has_export_netspresso": hasattr(ultralytics, "export_netspresso")}
    # --- 1. load the checkpoint with the fork (cross-version unpickle is itself a compatibility question) ---
    try:
        model = YOLO(args.weights.as_posix())
        record["load"] = {"status": "PASS", "task": model.task, "num_classes": len(model.names)}
        print(f"[load] fork loaded {args.weights.name}: task={model.task} nc={len(model.names)}")
    except Exception as exc:  # noqa: BLE001
        record["load"] = {"status": "FAIL", "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        print(f"[load] FAIL {record['load']['error']}")
        _finish(out, record, ledger_before)
        return 3

    # --- 2. export_netspresso -> model_fx.pt + netspresso_head_meta.json ---
    try:
        model.export_netspresso(save_path=OUTPUTS.as_posix())
        fx_path, meta_path = OUTPUTS / "model_fx.pt", OUTPUTS / "netspresso_head_meta.json"
        record["export"] = {"status": "PASS" if fx_path.is_file() and meta_path.is_file() else "FAIL",
                            "model_fx": {"path": portable(fx_path.as_posix()), "size_bytes": fx_path.stat().st_size if fx_path.is_file() else None,
                                         "sha256": compute_sha256(fx_path) if fx_path.is_file() else None},
                            "head_meta": json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else None}
        hm = record["export"]["head_meta"] or {}
        print(f"[export] model_fx.pt {record['export']['model_fx']['size_bytes']} bytes | head_meta keys={sorted(hm)} nc={hm.get('nc')} nl={hm.get('nl')} stride={hm.get('stride')}")
    except Exception as exc:  # noqa: BLE001
        record["export"] = {"status": "FAIL", "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        print(f"[export] FAIL {record['export']['error']}")
        _finish(out, record, ledger_before)
        return 4

    # --- 3. structural validation of the fx artifact (generated locally from trusted weights -> full unpickle allowed) ---
    sv = validate_pt(fx_path, allow_full_unpickle=True, torch_module=torch).to_dict()
    record["structural_validation"] = sv
    print(f"[structure] model_fx.pt: {sv['status']} - {sv['message']} | kind={sv['details'].get('object_kind')} params={sv['details'].get('parameter_count')}")

    # --- 4. forward sanity: the traced body (train mode) returns the raw per-scale feature maps ---
    try:
        gm = torch.load(fx_path.as_posix(), map_location="cpu")
        gm.train()
        with torch.no_grad():
            outs = gm(torch.zeros(1, 3, args.imgsz, args.imgsz))
        shapes = [list(o.shape) for o in (outs if isinstance(outs, (list, tuple)) else [outs])]
        record["forward"] = {"status": "PASS", "mode": "train (as traced)", "input_shape": [1, 3, args.imgsz, args.imgsz], "output_shapes": shapes,
                             "note": "outputs are per-scale head feature maps (nc + 4*reg_max channels); decode lives in Detect_netspresso, rebuilt from head_meta"}
        print(f"[forward] OK output shapes {shapes}")
    except Exception as exc:  # noqa: BLE001
        record["forward"] = {"status": "FAIL", "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        print(f"[forward] FAIL {record['forward']['error']}")

    # --- 5. comparison with the compressor contract observed in Phase 1/5-B ---
    record["sdk_contract_check"] = {
        "input_format_expected_by_compressor_v2": "torch.fx GraphModule saved with torch.save (framework=pytorch, extension .pt)",
        "fx_artifact_is_graph_module": sv["details"].get("object_kind") == "graph_module",
        "input_shapes_for_sdk": [{"batch": 1, "channel": 3, "dimension": [args.imgsz, args.imgsz]}],
        "phase_5b_precedent": "graphmodule.pt (official sample) compressed successfully with the same contract",
        "note": "server-side acceptance is only proven by a real run; this precheck removes the 2025 failure mode (non-fx input) before spending credits",
    }
    record["environment"] = environment_fingerprint(packages=("ultralytics", "torch", "torchvision", "numpy", "onnx", "psutil"))
    verdict = "READY" if (record["export"]["status"] == "PASS" and sv["status"] == "PASS" and record.get("forward", {}).get("status") == "PASS") else "BLOCKED"
    record["verdict"] = verdict
    _finish(out, record, ledger_before)
    print(f"\n[precheck] verdict: {verdict}")
    return 0 if verdict == "READY" else 5


def _finish(out: Path, record: dict, ledger_before: str) -> None:
    assert not any(m == "netspresso" or m.startswith("netspresso.") for m in sys.modules), "SDK must never be imported here"
    record["timestamp"] = datetime.now(UTC).replace(microsecond=0).isoformat()
    record["ledger_unchanged"] = hashlib.sha256(LEDGER.read_bytes()).hexdigest() == ledger_before
    (out / "fx_export_precheck.json").write_text(portable(json.dumps(record, indent=2, ensure_ascii=False, default=str)) + "\n", encoding="utf-8")
    print(f"ledger unchanged: {record['ledger_unchanged']} | NetsPresso API calls: 0 | record: {portable((out / 'fx_export_precheck.json').as_posix())}")


if __name__ == "__main__":
    raise SystemExit(main())

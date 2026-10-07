#!/usr/bin/env python
"""Phase 5-E E5-1: ONE authorized real NetsPresso automatic_compression of the R1-validated YOLOv8n fx body.

Run from the SDK environment (Python 3.11, torch 2.0.1, netspresso 1.17.0):

    .venv-netspresso/Scripts/python scripts/run_e5_1_experiment.py --precheck-only          # 0 credits, no API call
    .venv-netspresso/Scripts/python scripts/run_e5_1_experiment.py --confirm-credit-use     # exactly ONE real operation

Order (every gate is a hard stop; nothing is retried):
    1. ledger snapshot (used / remaining / operations, SHA-256)
    2. input identity: exact R1 SHA-256 + size, old fork artifact forbidden        (framework.evaluation.input_gate)
    3. structural + forward + precheck_fx_bundle on the input                      (must be READY)
    4. authorization: identity PASS, precheck READY, ledger shows exactly 1 prior real op, --confirm-credit-use
    5. ONE real operation through QA Engine -> NetsPressoAdapter -> SDK            (scripts/run_real_netspresso.py)
    6. ledger re-read, observed account delta classified (never assumed to be 25), traceability record

Local 0-credit validation of the returned artifact is a separate script run in the upstream environment:
scripts/yolov8_e5_1_local_validation.py. API keys are never printed or persisted.
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
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from framework.config import load_config  # noqa: E402
from framework.evaluation.artifact_structure import validate_pt  # noqa: E402
from framework.evaluation.detection_head import (  # noqa: E402
    expected_body_output_shapes,
    load_head_meta,
    precheck_fx_bundle,
)
from framework.evaluation.input_gate import (  # noqa: E402
    authorize_single_real_operation,
    classify_credit_delta,
    ledger_transition_ok,
    verify_input_artifact,
)

LEDGER = ROOT / "reports" / "credit_usage.json"
R1_DIR = ROOT / "outputs" / "models" / "yolov8n_fx_r1"
R1_SHA = "d8e761dae29301ef51df9e7679c729338a522d397e271a900d453d171e1c022b"
R1_SIZE = 12846691
OLD_FORK_ARTIFACT = "yolov8n_fx/model_fx.pt"
EXPECTED_PRIOR_REAL_OPERATIONS = 1
EXPECTED_CLIENT_SIDE_CREDIT = 25
EXPERIMENT_ID = "E5-1"


def portable(text: str) -> str:
    root_fwd = ROOT.as_posix() + "/"
    root_back = str(ROOT) + "\\"
    return text.replace(root_back.replace("\\", "\\\\"), "").replace(root_back, "").replace(root_fwd, "")


def ledger_snapshot() -> dict:
    data = json.loads(LEDGER.read_text(encoding="utf-8"))
    return {"sha256": hashlib.sha256(LEDGER.read_bytes()).hexdigest(), "used_credit": data["used_credit"], "remaining_estimate": data["remaining_estimate"],
            "operations": len(data["operations"]), "raw": data}


def dump(path: Path, data: dict) -> None:
    path.write_text(portable(json.dumps(data, indent=2, ensure_ascii=False, default=str)) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirm-credit-use", action="store_true", help="explicit authorization for exactly ONE real operation")
    ap.add_argument("--precheck-only", action="store_true", help="run the input gate only; never calls NetsPresso")
    ap.add_argument("--model-path", type=Path, default=R1_DIR / "model_fx.pt")
    ap.add_argument("--head-meta", type=Path, default=R1_DIR / "netspresso_head_meta.json")
    ap.add_argument("--expected-sha256", default=R1_SHA)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--compression-ratio", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or (ROOT / "reports" / "yolov8_e5_1" / stamp)
    out.mkdir(parents=True, exist_ok=True)
    input_shape = [1, 3, args.imgsz, args.imgsz]
    before = ledger_snapshot()
    print(f"[ledger before] used {before['used_credit']} remaining {before['remaining_estimate']} operations {before['operations']} sha {before['sha256'][:16]}")

    # ---- 2. input identity (exact R1 SHA; old fork artifact forbidden) ----
    identity = verify_input_artifact(args.model_path, expected_sha256=args.expected_sha256, expected_size_bytes=R1_SIZE, forbidden_paths=(OLD_FORK_ARTIFACT,))
    identity["details"]["path"] = portable(identity["details"]["path"])
    print(f"[identity] {identity['status']}: {identity['message']} (sha {identity['details'].get('sha256', '')[:16]}, size {identity['details'].get('size_bytes')})")

    # ---- 3. structural / forward / precheck (torch 2.0.1 = target-side dependency) ----
    structural, precheck_status, forward_shapes = {"status": "NOT_RUN"}, "BLOCKED", None
    if identity["status"] == "PASS":
        import torch  # noqa: PLC0415

        config = load_config(ROOT / "configs")
        trusted = config.local_eval.is_trusted(identity["details"]["sha256"])
        structural = validate_pt(args.model_path, allow_full_unpickle=trusted, torch_module=torch).to_dict()
        structural["trusted_for_full_unpickle"] = trusted

        def body_forward(shape):
            gm = torch.load(args.model_path.as_posix(), map_location="cpu")
            gm.eval()
            with torch.no_grad():
                outs = gm(torch.zeros(*shape))
            return [list(o.shape) for o in outs]

        pre = precheck_fx_bundle(args.model_path, args.head_meta, input_shape=input_shape, expected_nc=80, allow_full_unpickle=trusted, torch_module=torch, forward=body_forward)
        precheck_status = pre.status
        forward_shapes = pre.details.get("observed_body_output_shapes")
        structural["precheck"] = pre.to_dict()
        print(f"[structure] {structural['status']} kind={structural['details'].get('object_kind')} params={structural['details'].get('parameter_count')} | torch {torch.__version__} forward {forward_shapes} | precheck {pre.status}")

    # ---- 4. authorization ----
    auth = authorize_single_real_operation(input_identity=identity, precheck_status=precheck_status, ledger_operations_before=before["operations"],
                                           expected_operations_before=EXPECTED_PRIOR_REAL_OPERATIONS, confirm_credit_use=args.confirm_credit_use and not args.precheck_only)
    record: dict = {
        "experiment_id": EXPERIMENT_ID, "phase": "5-E", "timestamp_start": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "planned_operation": {"operation": "automatic_compression", "model": "yolov8n", "input_shape": input_shape, "compression_ratio": args.compression_ratio, "framework": "pytorch",
                              "expected_client_side_credit": EXPECTED_CLIENT_SIDE_CREDIT, "estimate_basis": "CLIENT_SIDE_PRE_CHECK_CONSTANT (netspresso 1.17.0); actual deduction observed via account balance",
                              "max_real_operations": 1, "retry_policy": "none"},
        "input": {"artifact": portable(args.model_path.as_posix()), "head_meta": portable(args.head_meta.as_posix()), "identity": identity, "structural": structural,
                  "precheck_status": precheck_status, "expected_body_output_shapes": expected_body_output_shapes(input_shape, load_head_meta(args.head_meta)) if args.head_meta.is_file() else None,
                  "generation": "scripts/yolov8_r1_traceable_export.py (upstream ultralytics 8.4.173, patch r1-upstream-body-wrapper-v1)",
                  "r1_validation_reference": "reports/yolov8_baseline/20261005T034329Z_r1/r1_result.json"},
        "ledger_before": {k: v for k, v in before.items() if k != "raw"},
        "authorization": auth,
    }
    dump(out / "e5_1_precheck.json", record)
    print(f"[authorization] {'AUTHORIZED' if auth['authorized'] else 'NOT AUTHORIZED'} {auth['reasons']}")
    if args.precheck_only or not auth["authorized"]:
        print("No NetsPresso API call was made." if args.precheck_only else "STOP: gate not satisfied. No NetsPresso API call was made.")
        return 0 if args.precheck_only and precheck_status == "READY" and identity["status"] == "PASS" else 3

    # ---- 5. exactly ONE real operation via the framework path ----
    import run_real_netspresso as rr  # noqa: PLC0415 - QA Engine -> NetsPressoAdapter -> SDK

    run_dir = ROOT / "reports" / "real_runs" / f"{stamp}_automatic_compression"
    rr_args = ["--confirm-credit-use", "--operation", "automatic_compression", "--model-path", args.model_path.as_posix(), "--model-name", "yolov8n",
               "--input-shape", ",".join(str(x) for x in input_shape), "--compression-ratio", str(args.compression_ratio), "--out", run_dir.as_posix(),
               "--purpose", "Phase 5-E E5-1: single automatic_compression of the R1-validated upstream YOLOv8n fx body (ratio 0.5, 640x640)"]
    record["real_execution"] = {"script": "scripts/run_real_netspresso.py", "run_dir": portable(run_dir.as_posix()), "args": [portable(a) for a in rr_args]}
    print("\n=== E5-1: ONE REAL OPERATION (no retry, no second call) ===")
    try:
        rc = rr.main(rr_args)
    except Exception as exc:  # noqa: BLE001 - never retry; record and continue with accounting
        rc = -1
        record["real_execution"]["exception"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    record["real_execution"]["exit_code"] = rc

    # ---- 6. accounting (observed, not assumed) ----
    after = ledger_snapshot()
    record["ledger_after"] = {k: v for k, v in after.items() if k != "raw"}
    exec_path = run_dir / "execution_result.json"
    execution = json.loads(exec_path.read_text(encoding="utf-8")) if exec_path.is_file() else None
    credit = (execution or {}).get("credit") or {}
    logs = (execution or {}).get("logs") or []
    before_total = after_total = None
    for line in logs:
        if "credit total before:" in line:
            before_total = _int_tail(line)
        if "credit total after:" in line:
            after_total = _int_tail(line)
    delta = classify_credit_delta(before=before_total, after=after_total, expected=EXPECTED_CLIENT_SIDE_CREDIT)
    transition = ledger_transition_ok(before=before["raw"], after=after["raw"], observed_delta=delta["observed"])
    new_ops = after["raw"]["operations"][before["operations"]:]
    record["credit"] = {"account_total_before": before_total, "account_total_after": after_total, "delta_classification": delta, "framework_credit_record": credit,
                        "ledger_transition": transition, "ledger_entries_added": [{k: v for k, v in e.items() if k != "details"} for e in new_ops],
                        "service_operations_executed": (execution or {}).get("environment", {}).get("extra", {}).get("operations_executed"),
                        "auth_sessions": (execution or {}).get("environment", {}).get("extra", {}).get("api_calls_executed")}
    record["execution_summary"] = None if execution is None else {
        "execution_status": execution["execution_status"], "sdk_version": execution["environment"].get("sdk_version"), "timestamp": execution["timestamp"],
        "execution_time_s": execution.get("execution_time_s"), "error": execution.get("error"),
        "artifact": {k: execution["artifact"].get(k) for k in ("path", "exists", "size_bytes", "format", "checksum_sha256")} if execution.get("artifact") else None,
        "sdk_model_id": (execution.get("artifact") or {}).get("metadata", {}).get("sdk_model_id"),
        "compressed_onnx": (execution.get("artifact") or {}).get("metadata", {}).get("compressed_onnx_model_path"),
        "sdk_original": (execution.get("baseline_metrics") or {}).get("extra"), "sdk_compressed": (execution.get("metrics") or {}).get("extra"),
        "logs": logs}
    record["timestamp_end"] = datetime.now(UTC).replace(microsecond=0).isoformat()
    record["post_call_rule"] = "ZERO further NetsPresso calls in this phase; remaining validation is local (scripts/yolov8_e5_1_local_validation.py)"
    record["secrets"] = "API key read from environment/.env by the adapter only; never persisted (checked by tests)"
    dump(out / "e5_1_execution.json", record)
    print(f"\n[ledger after] used {after['used_credit']} remaining {after['remaining_estimate']} operations {after['operations']} | account {before_total} -> {after_total} | delta {delta['status']} ({delta['observed']}) | ledger transition ok={transition['ok']}")
    print(f"[record] {portable((out / 'e5_1_execution.json').as_posix())}")
    return 0 if rc == 0 else 1


def _int_tail(line: str) -> int | None:
    try:
        return int(line.rsplit(":", 1)[1].strip())
    except (ValueError, IndexError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())

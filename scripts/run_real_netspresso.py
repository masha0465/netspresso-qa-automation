#!/usr/bin/env python
"""Execute exactly ONE real NetsPresso operation through NetsPressoAdapter (Phase 5-B).

Run from the Python 3.11 SDK environment:

    .venv-netspresso/Scripts/python scripts/run_real_netspresso.py --confirm-credit-use \
        --operation automatic_compression --model-path outputs/models/graphmodule.pt

Safety:
* aborts without --confirm-credit-use (exit 2)
* aborts if NETSPRESSO_API_KEY is absent (exit 4) - the value is never printed
* aborts if the ledger budget check fails (exit 5)
* performs exactly one service operation; no retry, no loop, no matrix
* records the result in reports/credit_usage.json with estimated vs observed credit separated
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os  # noqa: E402

from framework.adapters.base import ExecutionRequest  # noqa: E402
from framework.adapters.netspresso_adapter import (  # noqa: E402
    REAL_IMPLEMENTED_OPERATIONS,
    MissingApiKeyError,
    NetsPressoAdapter,
    RealExecutionNotAuthorizedError,
    RealExecutionNotImplementedError,
    SdkUnavailableError,
)
from framework.config import CriterionConfig, ProviderConfig, QualityGateConfig, load_config  # noqa: E402
from framework.credit_ledger import CreditLedger  # noqa: E402
from framework.defects import classify  # noqa: E402
from framework.pipeline.result import CaseResult, Configuration, CreditUsageType, SupportState  # noqa: E402
from framework.pipeline.runner import case_status_for  # noqa: E402
from framework.quality_gate import QualityGate, render_text  # noqa: E402

LEDGER_PATH = ROOT / "reports" / "credit_usage.json"
ENV_FILE = ROOT / ".env"
ALLOWED_ENV_KEYS = ("NETSPRESSO_API_KEY", "GA_DISABLE_ANALYTICS")


def load_dotenv_minimal(path: Path) -> list[str]:
    """Load only the allowed keys from .env into os.environ (without overriding). Values are never printed."""
    loaded: list[str] = []
    if not path.is_file():
        return loaded
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key in ALLOWED_ENV_KEYS and value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def portable(text: str) -> str:
    """Strip the local repository root from persisted text so reports never embed user directories."""
    root_fwd = ROOT.as_posix() + "/"
    root_back = str(ROOT) + "\\"
    return text.replace(root_back.replace("\\", "\\\\"), "").replace(root_back, "").replace(root_fwd, "")


def parse_shape(text: str) -> list[dict]:
    dims = [int(x) for x in text.split(",")]
    if len(dims) != 4:
        raise argparse.ArgumentTypeError("input shape must be N,C,H,W")
    return [{"batch": dims[0], "channel": dims[1], "dimension": [dims[2], dims[3]]}]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--confirm-credit-use", action="store_true", help="explicit authorization; required")
    parser.add_argument("--operation", choices=REAL_IMPLEMENTED_OPERATIONS, default="automatic_compression")
    parser.add_argument("--model-path", type=Path, default=ROOT / "outputs" / "models" / "graphmodule.pt")
    parser.add_argument("--model-name", default="graphmodule_sample", help="label used in the Result Model")
    parser.add_argument("--input-shape", type=parse_shape, default="1,3,224,224")
    parser.add_argument("--framework", default="pytorch", help="netspresso.enums.Framework value")
    parser.add_argument("--compression-ratio", type=float, default=0.5)
    parser.add_argument("--device", default="intel_xeon_w2233", help="nominal matrix device (compression is device-agnostic)")
    parser.add_argument("--runtime", default="onnxruntime")
    parser.add_argument("--backend", default="default")
    parser.add_argument("--purpose", default="Phase 5-B: first real NetsPresso operation (E2E adapter validation)")
    parser.add_argument("--out", type=Path, default=None, help="run directory (default reports/real_runs/<utc>_<operation>)")
    parser.add_argument("--configs", type=Path, default=ROOT / "configs")
    args = parser.parse_args(argv)

    if isinstance(args.input_shape, str):
        args.input_shape = parse_shape(args.input_shape)

    if not args.confirm_credit_use:
        print("ABORT: this script performs a REAL NetsPresso operation that may consume credits.\n"
              "Re-run with --confirm-credit-use only after explicit approval. No API call was made.")
        return 2

    loaded = load_dotenv_minimal(ENV_FILE)
    if loaded:
        print(f"Loaded from .env: {', '.join(loaded)} (values not shown)")
    if not os.environ.get("NETSPRESSO_API_KEY"):
        print("ABORT: NETSPRESSO_API_KEY is not set (environment or .env). No API call was made.")
        return 4
    if not args.model_path.is_file():
        print(f"ABORT: model file not found: {args.model_path}")
        return 6

    config = load_config(args.configs)
    settings = ProviderConfig(provider="netspresso", mode="real", confirm_credit_use=True,
                              api_key_env=config.provider.api_key_env, disable_analytics=config.provider.disable_analytics,
                              dev_mode=config.provider.dev_mode, sleep_interval_s=config.provider.sleep_interval_s,
                              reserve_credit=config.provider.reserve_credit)
    adapter = NetsPressoAdapter(config, settings=settings)
    ledger = CreditLedger(LEDGER_PATH)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out or (ROOT / "reports" / "real_runs" / f"{stamp}_{args.operation}")
    run_dir.mkdir(parents=True, exist_ok=True)
    configuration = Configuration(args.model_name, args.device, args.runtime, args.backend, args.operation)

    plan = adapter.plan(args.operation, configuration)
    estimate = plan.estimated_credit or 0
    budget = ledger.budget_check([estimate], reserve=settings.reserve_credit)
    print("=== PLAN (pre-execution) ===")
    print(plan.render())
    print(f"Ledger:            remaining {ledger.remaining_estimate}, after plan {budget['remaining_after']} (reserve {budget['reserve']})")
    if not budget["affordable"]:
        print("ABORT: budget check failed. No API call was made.")
        return 5

    request = ExecutionRequest(
        operation=args.operation,
        configuration=configuration,
        parameters={
            "input_model_path": args.model_path.as_posix(),
            "output_dir": (run_dir / "sdk_output").as_posix(),
            "input_shapes": args.input_shape,
            "framework": args.framework,
            "compression_ratio": args.compression_ratio,
            "sdk_log_path": (run_dir / "sdk_log.txt").as_posix(),
        },
        repeat_runs=1,
    )

    print("\n=== EXECUTING ONE REAL OPERATION (no retry) ===")
    error_text = None
    try:
        execution = adapter.run(request)
    except (RealExecutionNotAuthorizedError, RealExecutionNotImplementedError, MissingApiKeyError, SdkUnavailableError) as exc:
        print(f"ABORT before API call: {type(exc).__name__}: {exc}")
        return 3
    except Exception as exc:  # noqa: BLE001 - must still account for a possibly-started operation
        error_text = f"{type(exc).__name__}: {exc}"
        print(f"UNEXPECTED FAILURE: {error_text}")
        execution = None

    # --- ledger entry (always, if an operation was attempted) ---
    attempted = adapter.operations_executed > 0
    if attempted:
        credit = execution.credit if execution is not None else None
        usage_type = credit.usage_type if credit and credit.usage_type in (CreditUsageType.ACTUAL, CreditUsageType.ESTIMATED) else CreditUsageType.ESTIMATED
        entry = ledger.record(
            operation=args.operation,
            model=args.model_name,
            purpose=args.purpose,
            usage_type=usage_type,
            estimated_credit=estimate,
            actual_credit=credit.actual if credit else None,
            result=execution.execution_status.value if execution else "EXCEPTION",
            error=(execution.error.message if execution and execution.error else error_text),
            confirmation=True,
            adapter=adapter.name,
            details={
                "estimate_basis": "CLIENT_SIDE_PRE_CHECK_CONSTANT (not verified server-side)",
                "credit_note": credit.note if credit else None,
                "run_dir": run_dir.relative_to(ROOT).as_posix(),
                "sdk_version": execution.environment.sdk_version if execution else None,
                "model_path": portable(args.model_path.as_posix()),
                "operations_executed": adapter.operations_executed,
                "api_calls_executed": adapter.api_calls_executed,
            },
        )
        ledger.save()
        print(f"\nLedger entry recorded: usage_type={entry.usage_type} estimated={entry.estimated_credit} actual={entry.actual_credit} "
              f"remaining_estimate={entry.remaining_estimate}")
    else:
        print("\nNo service operation was attempted; ledger unchanged.")

    if execution is None:
        return 1

    # --- persist raw result ---
    (run_dir / "execution_result.json").write_text(portable(json.dumps(execution.to_dict(), indent=2, ensure_ascii=False)), encoding="utf-8")
    for produced in (run_dir / "sdk_log.txt", run_dir / "sdk_output" / "metadata.json"):
        if produced.is_file():
            produced.write_text(portable(produced.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")

    # --- QA validation through the standard model ---
    full_gate = QualityGate(config.quality_gate).evaluate(execution)
    stage_gate = QualityGate(QualityGateConfig({
        "model_size": CriterionConfig("model_size", 0.0, True),  # compression must not grow the model
        "artifact": CriterionConfig("artifact", None, False),  # no reference checksum exists yet -> WARN, not FAIL
    })).evaluate(execution)
    defect = classify(execution, full_gate)
    case = CaseResult(configuration=configuration, status=case_status_for(execution, full_gate),
                      support=SupportState.SUPPORTED if execution.execution_status.value == "COMPLETED" else SupportState.UNKNOWN,
                      selected=True, selection_reason="Phase 5-B single real experiment", execution=execution, gate=full_gate, defect=defect)
    (run_dir / "case_result.json").write_text(portable(json.dumps(case.to_dict(), indent=2, ensure_ascii=False)), encoding="utf-8")
    (run_dir / "stage_gate.json").write_text(json.dumps(stage_gate.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== RESULT ===")
    print(f"execution_status:  {execution.execution_status.value}")
    print(f"sdk_version:       {execution.environment.sdk_version}")
    print(f"timestamp:         {execution.timestamp}")
    print(f"operations run:    {adapter.operations_executed} | auth sessions: {adapter.api_calls_executed}")
    print(f"credit:            {execution.credit.to_dict()}")
    if execution.error:
        print(f"error:             {execution.error.to_dict()}")
    if execution.artifact:
        print(f"artifact:          path={execution.artifact.path} exists={execution.artifact.exists} size={execution.artifact.size_bytes} sha256={execution.artifact.checksum_sha256}")
    print(f"baseline metrics:  {execution.baseline_metrics.to_dict() if execution.baseline_metrics else 'NOT_AVAILABLE'}")
    print(f"current metrics:   {execution.metrics.to_dict() if execution.metrics else 'NOT_AVAILABLE'}")
    print("accuracy / latency / memory: NOT_AVAILABLE (automatic_compression does not measure them)")
    print("\n--- Full release Quality Gate (default thresholds) ---")
    print(render_text(full_gate))
    print("\n--- Compression-stage gate (model_size, artifact) ---")
    print(render_text(stage_gate))
    print(f"\nCase status:       {case.status.value}")
    if defect:
        print(f"Defect:            {defect.category.value} [{defect.severity.value}] {defect.message}")
    print(f"\nArtifacts:         {run_dir.relative_to(ROOT).as_posix()}/ (execution_result.json, case_result.json, stage_gate.json, sdk_log.txt, sdk_output/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

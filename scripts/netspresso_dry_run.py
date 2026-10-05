#!/usr/bin/env python
"""Dry-run planner for the real NetsPresso adapter (Phase 5-A).

Default behaviour: DRY_RUN. No SDK import, no network, no authentication, 0 credits.

    python scripts/netspresso_dry_run.py                                  # single operation plan
    python scripts/netspresso_dry_run.py --operation profile --device jetson_orin_nano --runtime tensorrt
    python scripts/netspresso_dry_run.py --matrix pairwise                 # plan + budget for a whole selection
    python scripts/netspresso_dry_run.py --mode real                      # -> rejected (exit 2), no API call
    python scripts/netspresso_dry_run.py --mode real --confirm-credit-use # -> not implemented in 5-A (exit 3)

Exit codes: 0 dry-run plan produced | 2 real run rejected (unauthorized) | 3 real run not implemented.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from framework.adapters.factory import build_adapter  # noqa: E402
from framework.adapters.netspresso_adapter import (  # noqa: E402
    SDK_OPERATIONS,
    ExecutionMode,
    NetsPressoAdapter,
    RealExecutionNotAuthorizedError,
)
from framework.config import load_config  # noqa: E402
from framework.credit_ledger import CreditLedger  # noqa: E402
from framework.pipeline.result import Configuration  # noqa: E402
from framework.pipeline.runner import STRATEGIES, PipelineRunner  # noqa: E402

LEDGER_PATH = ROOT / "reports" / "credit_usage.json"


def _single(adapter: NetsPressoAdapter, args: argparse.Namespace, ledger: CreditLedger) -> dict:
    cfg = None
    if args.operation != "validate_connection":
        cfg = Configuration(args.model, args.device, args.runtime, args.backend, args.operation)
    plan = adapter.plan(args.operation, cfg)
    print(plan.render())
    budget = ledger.budget_check([plan.estimated_credit or 0], reserve=adapter.settings.reserve_credit)
    print(f"Ledger:            used {ledger.used_credit} / {ledger.starting_credit}, remaining {ledger.remaining_estimate}, "
          f"after plan {budget['remaining_after']} (reserve {budget['reserve']}) -> {'OK' if budget['affordable'] else 'EXCEEDS BUDGET'}")
    return {"plan": plan.to_dict(), "budget": budget}


def _matrix(adapter: NetsPressoAdapter, args: argparse.Namespace, ledger: CreditLedger, config) -> dict:
    report = PipelineRunner(config, adapter, strategy=args.matrix, repeat_runs=1).run()
    planned = [c for c in report.cases if c.execution is not None]
    estimates = [c.execution.credit.estimated or 0 for c in planned]
    budget = ledger.budget_check(estimates, reserve=adapter.settings.reserve_credit)
    print(f"Provider:          NetsPresso\nExecution:         {adapter.mode.value}\nStrategy:          {args.matrix}")
    print(f"Planned operations:{len(planned):>5}   (all NOT_EXECUTED - dry run)")
    print(f"Estimated Credit:  {sum(estimates)} total ({'; '.join(f'{k}={v}' for k, v in sorted(_count_by_op(planned).items()))})")
    print("Actual Credit:     NOT MEASURED\nAPI call:          NOT EXECUTED\nCredit consumed:   0")
    print(f"Ledger:            remaining {ledger.remaining_estimate}, after plan {budget['remaining_after']} "
          f"(reserve {budget['reserve']}) -> {'OK' if budget['affordable'] else 'EXCEEDS BUDGET: this selection is too large for a real run'}")
    statuses = report.status_counts()
    print(f"Matrix statuses:   {statuses}")
    return {"strategy": args.matrix, "planned": len(planned), "estimated_total": sum(estimates),
            "by_operation": _count_by_op(planned), "budget": budget, "statuses": statuses,
            "plans": [c.execution.logs for c in planned]}


def _count_by_op(cases) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in cases:
        out[c.execution.operation] = out.get(c.execution.operation, 0) + (c.execution.credit.estimated or 0)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--operation", choices=sorted(SDK_OPERATIONS), default="int8_quantization")
    parser.add_argument("--model", default="mobilenet_v2")
    parser.add_argument("--device", default="intel_xeon_w2233")
    parser.add_argument("--runtime", default="onnxruntime")
    parser.add_argument("--backend", default="default")
    parser.add_argument("--matrix", choices=STRATEGIES, default=None, help="plan every selected combination instead of one")
    parser.add_argument("--mode", choices=("dry_run", "real"), default=None, help="override configs/netspresso.yaml (default dry_run)")
    parser.add_argument("--confirm-credit-use", action="store_true", help="explicit authorization for a real run (Phase 5-B)")
    parser.add_argument("--json", type=Path, default=None, help="write the plan as JSON")
    parser.add_argument("--configs", type=Path, default=ROOT / "configs")
    args = parser.parse_args(argv)

    config = load_config(args.configs)
    adapter = build_adapter(config, provider="netspresso", mode=args.mode, confirm_credit_use=args.confirm_credit_use or None)
    assert isinstance(adapter, NetsPressoAdapter)
    ledger = CreditLedger(LEDGER_PATH)

    try:
        adapter._enforce_policy("dry-run planner")  # noqa: SLF001 - policy gate is the point of this script
    except RealExecutionNotAuthorizedError as exc:
        print(f"REJECTED: {exc}")
        return 2
    if adapter.mode == ExecutionMode.REAL_RUN_AUTHORIZED:
        print("REFUSED: the dry-run planner never executes real runs. Use scripts/run_real_netspresso.py "
              "--confirm-credit-use from the Python 3.11 environment for a single authorized operation.")
        return 3

    result = _matrix(adapter, args, ledger, config) if args.matrix else _single(adapter, args, ledger)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Plan JSON:         {args.json}")
    print(f"SDK calls made:    {adapter.api_calls_executed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

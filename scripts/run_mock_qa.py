#!/usr/bin/env python
"""Run the complete QA workflow with the deterministic MockAdapter.

Zero network access, zero NetsPresso credits. Safe to run anywhere (CI included).

Examples
--------
    python scripts/run_mock_qa.py                       # pairwise, reports/examples/mock_pairwise
    python scripts/run_mock_qa.py --strategy full
    python scripts/run_mock_qa.py --strategy pairwise_risk --out reports/examples/pw_risk
    python scripts/run_mock_qa.py --regression-demo     # baseline vs regressed scenario set
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles may default to a legacy code page

from framework.adapters.mock_adapter import MockAdapter  # noqa: E402
from framework.config import load_config  # noqa: E402
from framework.credit_ledger import CreditLedger  # noqa: E402
from framework.pipeline.runner import STRATEGIES, PipelineRunner, RunReport  # noqa: E402
from framework.quality_gate import render_text  # noqa: E402
from framework.regression import RegressionComparator, RegressionThresholds  # noqa: E402
from framework.reporter import (  # noqa: E402
    write_regression_html,
    write_regression_json,
    write_run_html,
    write_run_json,
)

FIXED_CLOCK = "2026-10-01T00:00:00+00:00"


def run_once(strategy: str, out: Path, *, scenarios: Path | None = None, deterministic_time: bool) -> RunReport:
    cfg = load_config(ROOT / "configs", mock_scenarios=scenarios)
    ledger = CreditLedger(out / "credit_usage_simulated.json", starting_credit=500)
    runner = PipelineRunner(
        cfg,
        MockAdapter(cfg, clock=(lambda: FIXED_CLOCK) if deterministic_time else None),
        strategy=strategy,
        repeat_runs=2,
        ledger=ledger,
        clock=(lambda: FIXED_CLOCK) if deterministic_time else None,
    )
    report = runner.run()
    write_run_json(report, out / "qa_report.json")
    write_run_html(report, out / "qa_report.html")
    return report


def print_summary(report: RunReport) -> None:
    s = report.summary()
    print(f"\n=== Run {report.run_id} | strategy={report.strategy} | adapter={report.environment.adapter} ===")
    print("Matrix   :", json.dumps(report.matrix, ensure_ascii=False))
    print("Selection:", json.dumps({k: v for k, v in report.selection.items() if k != 'infeasible_pairs'}, ensure_ascii=False))
    print("Risk     :", json.dumps(report.risk_distribution))
    print("Status   :", {k: s[k] for k in ("total_combinations", "selected_combinations", "tested_combinations",
                                            "PASS", "FAIL", "BLOCKED", "NOT_TESTED", "UNSUPPORTED")})
    print("Gate     :", s["quality_gate"])
    print("Defects  :", s["defects"])
    for case in report.failed_cases()[:3]:
        print(f"\n--- {case.case_id} {case.configuration.key} -> {case.status.value}")
        if case.gate and case.gate.criteria:
            print(render_text(case.gate))
        if case.defect:
            d = case.defect
            print(f"DEFECT {d.category.value} [{d.severity.value}] {d.message}")
            if d.suspected_cause:
                print(f"  suspected cause: {d.suspected_cause}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", choices=STRATEGIES, default="pairwise")
    parser.add_argument("--out", type=Path, default=None, help="output directory (default reports/examples/mock_<strategy>)")
    parser.add_argument("--regression-demo", action="store_true", help="also run the regressed scenario set and compare")
    parser.add_argument("--wall-clock", action="store_true", help="use real timestamps instead of the fixed demo clock")
    args = parser.parse_args(argv)

    out = args.out or (ROOT / "reports" / "examples" / f"mock_{args.strategy}")
    deterministic = not args.wall_clock
    baseline = run_once(args.strategy, out, deterministic_time=deterministic)
    print_summary(baseline)
    print(f"\nJSON report: {out / 'qa_report.json'}\nHTML report: {out / 'qa_report.html'}")

    if args.regression_demo:
        cur_out = out.parent / f"{out.name}_regressed"
        current = run_once(
            args.strategy, cur_out, scenarios=ROOT / "configs" / "scenarios" / "mock_scenarios_regressed.yaml",
            deterministic_time=deterministic,
        )
        print_summary(current)
        cfg = load_config(ROOT / "configs")
        comparator = RegressionComparator(RegressionThresholds.from_quality_gate(cfg.quality_gate))
        reg = comparator.compare(baseline, current)
        reg_dir = out.parent / f"{out.name}_regression"
        write_regression_json(reg, reg_dir / "regression_report.json")
        write_regression_html(reg, reg_dir / "regression_report.html")
        print(f"\n=== Regression {reg.baseline_run_id} -> {reg.current_run_id}: {reg.overall} ===")
        print("Summary  :", reg.summary(), "| not executed in both:", reg.not_executed_in_both)
        for c in reg.regressions():
            print(f"  REGRESSION {c.case_id} {c.configuration}: {'; '.join(c.reasons)}")
        print(f"Regression JSON: {reg_dir / 'regression_report.json'}\nRegression HTML: {reg_dir / 'regression_report.html'}")

    print("\nCredits: MockAdapter only - no NetsPresso API call, actual credit usage 0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

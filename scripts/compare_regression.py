#!/usr/bin/env python
"""Compare two saved QA run reports (JSON) and write a regression report.

    python scripts/compare_regression.py baseline/qa_report.json current/qa_report.json --out reports/regression

Exit code 1 when the overall result is REGRESSION (useful as a CI gate).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from framework.config import load_config  # noqa: E402
from framework.regression import RegressionComparator, RegressionThresholds  # noqa: E402
from framework.reporter import load_run_report, write_regression_html, write_regression_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "regression")
    parser.add_argument("--configs", type=Path, default=ROOT / "configs", help="thresholds are taken from quality_gate.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.configs)
    report = RegressionComparator(RegressionThresholds.from_quality_gate(cfg.quality_gate)).compare(
        load_run_report(args.baseline), load_run_report(args.current)
    )
    write_regression_json(report, args.out / "regression_report.json")
    write_regression_html(report, args.out / "regression_report.html")
    print(f"Regression {report.baseline_run_id} -> {report.current_run_id}: {report.overall} {report.summary()}")
    for c in report.regressions():
        print(f"  REGRESSION {c.case_id}: {'; '.join(c.reasons)}")
    return 1 if report.overall == "REGRESSION" else 0


if __name__ == "__main__":
    raise SystemExit(main())

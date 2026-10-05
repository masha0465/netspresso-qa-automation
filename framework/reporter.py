"""JSON and HTML reporting for pipeline runs and regression comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from framework.pipeline.runner import RunReport
from framework.regression import RegressionReport

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

_env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=select_autoescape(["html", "j2"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


_env.filters["fmt"] = _fmt


def write_json(data: dict[str, Any], path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return p


def load_run_report(path: Path | str) -> RunReport:
    with Path(path).open("r", encoding="utf-8") as fh:
        return RunReport.from_dict(json.load(fh))


def write_run_json(report: RunReport, path: Path | str) -> Path:
    return write_json(report.to_dict(), path)


def write_run_html(report: RunReport, path: Path | str) -> Path:
    template = _env.get_template("run_report.html.j2")
    data = report.to_dict()
    html = template.render(
        r=data,
        failed=[c for c in data["cases"] if c["status"] in ("FAIL", "BLOCKED")],
        tested=[c for c in data["cases"] if c["status"] in ("PASS", "FAIL", "BLOCKED")],
    )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return p


def write_regression_json(report: RegressionReport, path: Path | str) -> Path:
    return write_json(report.to_dict(), path)


def write_regression_html(report: RegressionReport, path: Path | str) -> Path:
    template = _env.get_template("regression_report.html.j2")
    html = template.render(r=report.to_dict())
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return p


def write_local_eval_html(result: dict[str, Any], path: Path | str) -> Path:
    """Render the 0-credit local evaluation result (see scripts/run_local_eval.py)."""
    template = _env.get_template("local_eval_report.html.j2")
    html = template.render(r=result)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return p

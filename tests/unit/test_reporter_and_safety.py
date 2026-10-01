import importlib
import pkgutil
import re
import sys
from pathlib import Path

import pytest

import framework
from framework.adapters.mock_adapter import MockAdapter
from framework.pipeline.runner import PipelineRunner
from framework.reporter import load_run_report, write_run_html, write_run_json
from tests.conftest import fixed_clock

pytestmark = pytest.mark.unit


def test_json_and_html_reports(config, tmp_path):
    report = PipelineRunner(config, MockAdapter(config, clock=fixed_clock), strategy="pairwise", clock=fixed_clock).run()
    json_path = write_run_json(report, tmp_path / "r.json")
    html_path = write_run_html(report, tmp_path / "r.html")
    reloaded = load_run_report(json_path)
    assert reloaded.summary() == report.summary()
    assert reloaded.to_dict() == report.to_dict()

    html = html_path.read_text(encoding="utf-8")
    for token in ("NOT_TESTED", "UNSUPPORTED", "Quality Gate", "Defect Distribution", "Personal portfolio project",
                  "simulated and deterministic", report.run_id, "pair_coverage_percent"):
        assert token in html, token
    assert "<script" not in html  # static, safe for a public repo


def test_framework_never_imports_netspresso():
    for mod in pkgutil.walk_packages(framework.__path__, "framework."):
        importlib.import_module(mod.name)
    assert not any(name == "netspresso" or name.startswith("netspresso.") for name in sys.modules)
    root = Path(framework.__file__).resolve().parent
    offenders = [p for p in root.rglob("*.py") if re.search(r"^\s*(import|from)\s+netspresso\b", p.read_text(encoding="utf-8"), re.M)]
    assert offenders == []


def test_repository_has_no_secrets_and_ignores_env(root):
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    for entry in (".env", "netspresso.env", ".venv-netspresso/", "__pycache__/", ".pytest_cache/", "*.py[cod]"):
        assert entry in gitignore, entry
    key_pattern = re.compile(r"NETSPRESSO_API_KEY\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{12,}")
    scanned = 0
    for path in root.rglob("*"):
        if path.is_dir() or any(part in {".git", ".venv", ".venv-netspresso", ".claude", "__pycache__"} for part in path.parts):
            continue
        if path.suffix.lower() in {".py", ".yaml", ".yml", ".md", ".json", ".toml", ".txt", ".j2", ".example"}:
            scanned += 1
            assert not key_pattern.search(path.read_text(encoding="utf-8", errors="ignore")), path
    assert scanned > 20
    assert not (root / ".env").exists() or ".env" in gitignore

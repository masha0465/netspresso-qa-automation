"""Phase 5-E preparation: zero-credit guarantees of the YOLOv8n baseline helper and trust-list wiring."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from scripts import yolov8_baseline_prep as prep

pytestmark = pytest.mark.unit

YOLOV8N_SHA = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"


def test_yolov8n_weights_are_in_the_trust_list_with_provenance(config):
    trusted = {t.sha256: t for t in config.local_eval.trusted_artifacts}
    assert YOLOV8N_SHA in trusted
    assert "ultralytics/assets" in trusted[YOLOV8N_SHA].trust_basis and trusted[YOLOV8N_SHA].name == "yolov8n.pt"
    assert config.local_eval.is_trusted(YOLOV8N_SHA)


def test_baseline_helper_never_touches_netspresso_or_secrets(root):
    src = Path(root / "scripts" / "yolov8_baseline_prep.py").read_text(encoding="utf-8")
    assert "netspresso_adapter" not in src and "NetsPresso(" not in src
    assert not re.search(r"^\s*(import|from)\s+netspresso\b", src, re.M)
    assert "os.environ" not in src and "getenv" not in src
    assert "credit_consuming" in src and "No NetsPresso API call was made" in src


def test_baseline_helper_aborts_without_weights(tmp_path, capsys):
    rc = prep.main(["--weights", str(tmp_path / "missing.pt"), "--out", str(tmp_path / "out"), "--skip-val"])
    assert rc == 2 and "ABORT" in capsys.readouterr().out


def test_helper_pure_functions():
    assert prep.na("x") == {"status": "N/A", "reason": "x"}
    assert prep.portable(prep.ROOT.as_posix() + "/outputs/models/yolov8n.pt") == "outputs/models/yolov8n.pt"


def test_model_binaries_and_datasets_are_ignored(root):
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    for entry in ("outputs/", "datasets/", ".venv-yolo/", "*.pt", "*.onnx"):
        assert entry in gitignore, entry

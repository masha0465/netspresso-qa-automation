import shutil
from pathlib import Path

import pytest

from framework.config import ConfigurationError, load_config

pytestmark = pytest.mark.unit


def test_default_config_loads_all_dimensions(config):
    assert [m.name for m in config.models] == ["yolov8n", "mobilenet_v2", "pidnet_s"]
    assert len(config.devices) == 3 and len(config.runtimes) == 4 and len(config.backends) == 3
    assert len(config.optimizations) == 3
    assert config.device("raspberry_pi_4b").sdk_device_name == "RaspberryPi4B"
    assert config.optimization("int8_quantization").lossy is True
    assert config.optimization("fp16_conversion").sdk_credit_constant == 50


def test_quality_gate_config_thresholds_and_required_flags(config):
    qg = config.quality_gate
    assert qg.get("accuracy").threshold == 1.0 and qg.get("accuracy").required
    assert qg.get("model_size").required is False
    assert qg.get("artifact").threshold is None and qg.get("artifact").required


def test_risk_config(config):
    assert config.risk.weights["previous_regression"] == 3
    assert config.risk.medium_min_score <= config.risk.high_min_score


def test_mock_scenarios_loaded_in_order(config):
    ids = [s.id for s in config.mock.scenarios]
    assert ids[0] == "SCN-001" and "SCN-009" in ids
    assert config.mock.optimization_profiles["int8_quantization"]["latency_factor"] == 0.47


def _copy_configs(tmp_path: Path, root: Path) -> Path:
    dst = tmp_path / "configs"
    shutil.copytree(root / "configs", dst)
    return dst


def test_missing_file_is_configuration_error(tmp_path, root):
    dst = _copy_configs(tmp_path, root)
    (dst / "devices.yaml").unlink()
    with pytest.raises(ConfigurationError, match="not found"):
        load_config(dst)


def test_duplicate_names_are_rejected(tmp_path, root):
    dst = _copy_configs(tmp_path, root)
    text = (dst / "runtimes.yaml").read_text(encoding="utf-8")
    (dst / "runtimes.yaml").write_text(text.replace("name: openvino", "name: tflite"), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Duplicate runtime"):
        load_config(dst)


def test_backend_referencing_unknown_runtime_is_rejected(tmp_path, root):
    dst = _copy_configs(tmp_path, root)
    text = (dst / "backends.yaml").read_text(encoding="utf-8")
    (dst / "backends.yaml").write_text(text.replace("applicable_runtimes: [tflite]", "applicable_runtimes: [coreml]"), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unknown runtimes"):
        load_config(dst)


def test_invalid_scenario_outcome_is_rejected(tmp_path, root):
    dst = _copy_configs(tmp_path, root)
    text = (dst / "mock_scenarios.yaml").read_text(encoding="utf-8")
    (dst / "mock_scenarios.yaml").write_text(text.replace("outcome: BLOCKED", "outcome: MAYBE"), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="invalid outcome"):
        load_config(dst)


def test_scenario_override_path(regressed_config):
    assert regressed_config.mock.scenarios[0].id == "SCN-R01"

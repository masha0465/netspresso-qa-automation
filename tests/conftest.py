from __future__ import annotations

from pathlib import Path

import pytest

from framework.adapters.mock_adapter import MockAdapter
from framework.config import FrameworkConfig, load_config

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"
FIXED_CLOCK = "2026-10-01T00:00:00+00:00"


def fixed_clock() -> str:
    return FIXED_CLOCK


@pytest.fixture(scope="session")
def config() -> FrameworkConfig:
    return load_config(CONFIG_DIR)


@pytest.fixture(scope="session")
def regressed_config() -> FrameworkConfig:
    return load_config(CONFIG_DIR, mock_scenarios=CONFIG_DIR / "scenarios" / "mock_scenarios_regressed.yaml")


@pytest.fixture
def adapter(config: FrameworkConfig) -> MockAdapter:
    return MockAdapter(config, clock=fixed_clock)


@pytest.fixture
def root() -> Path:
    return ROOT

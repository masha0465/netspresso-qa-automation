"""YAML configuration loading with explicit validation.

Configuration errors are raised as :class:`ConfigurationError` so that they can
be classified as ``CONFIGURATION_ERROR`` instead of surfacing as generic
Python exceptions deep inside the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from framework.pipeline.result import RiskLevel

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


class ConfigurationError(ValueError):
    """Raised when a configuration file is missing, malformed, or inconsistent."""


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigurationError(f"Top-level YAML structure must be a mapping: {path}")
    return data


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ConfigurationError(f"Missing required key '{key}' in {where}")
    return mapping[key]


# --------------------------------------------------------------------------- #
# Dimension specs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    name: str
    task: str
    framework: str
    input_shape: tuple[int, ...]
    accuracy_metric: str
    source: str | None
    risk: dict[str, bool]
    mock_baseline: dict[str, float]


@dataclass(frozen=True)
class UnsupportedRule:
    runtime: str | None
    backend: str | None
    optimization: str | None
    reason: str

    def matches(self, runtime: str, backend: str, optimization: str) -> bool:
        return all(
            expected is None or expected == actual
            for expected, actual in ((self.runtime, runtime), (self.backend, backend), (self.optimization, optimization))
        )


@dataclass(frozen=True)
class DeviceSpec:
    name: str
    display_name: str
    sdk_device_name: str | None
    family: str
    risk: dict[str, bool]
    known_unsupported: tuple[UnsupportedRule, ...]


@dataclass(frozen=True)
class RuntimeSpec:
    name: str
    display_name: str
    sdk_framework: str | None
    risk: dict[str, bool]


@dataclass(frozen=True)
class BackendSpec:
    name: str
    display_name: str
    applicable_runtimes: tuple[str, ...]
    risk: dict[str, bool]


@dataclass(frozen=True)
class OptimizationSpec:
    name: str
    display_name: str
    precision: str
    sdk_operation: str | None
    sdk_credit_constant: int | None  # client-side SDK constant; NOT verified server-side billing
    risk: dict[str, bool]

    @property
    def lossy(self) -> bool:
        return self.precision.lower() == "int8"


# --------------------------------------------------------------------------- #
# Quality gate / risk / mock
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CriterionConfig:
    name: str
    threshold: float | None
    required: bool


@dataclass(frozen=True)
class QualityGateConfig:
    criteria: dict[str, CriterionConfig]

    def get(self, name: str) -> CriterionConfig | None:
        return self.criteria.get(name)


@dataclass(frozen=True)
class RiskConfig:
    weights: dict[str, int]
    high_min_score: int
    medium_min_score: int
    include_levels: tuple[RiskLevel, ...]


@dataclass(frozen=True)
class MockScenario:
    id: str
    match: dict[str, str]
    outcome: str
    error_kind: str | None = None
    message: str | None = None
    metric_overrides: dict[str, float] = field(default_factory=dict)
    artifact_checksum_valid: bool | None = None
    reproducibility: str | None = None
    note: str | None = None

    def matches(self, cfg: dict[str, str]) -> bool:
        return all(value == "*" or cfg.get(dim) == value for dim, value in self.match.items())


@dataclass(frozen=True)
class MockConfig:
    optimization_profiles: dict[str, dict[str, float]]
    scenarios: tuple[MockScenario, ...]


# --------------------------------------------------------------------------- #
# Provider / execution policy (configs/netspresso.yaml)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProviderConfig:
    """Which adapter runs and under which safety policy. Defaults are the safe ones."""

    provider: str = "mock"  # mock | netspresso
    mode: str = "dry_run"  # dry_run | real
    confirm_credit_use: bool = False
    api_key_env: str = "NETSPRESSO_API_KEY"
    disable_analytics: bool = True
    dev_mode: bool = False
    sleep_interval_s: int = 30
    reserve_credit: int = 100


def load_provider(path: Path) -> ProviderConfig:
    """Load the provider policy. A missing file yields the safe defaults."""
    if not path.exists():
        return ProviderConfig()
    raw = _load_yaml(path)
    provider = str((raw.get("provider") or {}).get("type", "mock"))
    if provider not in ("mock", "netspresso"):
        raise ConfigurationError(f"provider.type must be 'mock' or 'netspresso', got '{provider}'")
    execution = raw.get("execution") or {}
    mode = str(execution.get("mode", "dry_run"))
    if mode not in ("dry_run", "real"):
        raise ConfigurationError(f"execution.mode must be 'dry_run' or 'real', got '{mode}'")
    np_cfg = raw.get("netspresso") or {}
    forbidden = {k for k in np_cfg if "key" in k.lower() and k.lower() != "api_key_env"}
    if forbidden:
        raise ConfigurationError(
            f"netspresso.yaml must not contain credential values ({sorted(forbidden)}); use the environment variable named by api_key_env"
        )
    return ProviderConfig(
        provider=provider,
        mode=mode,
        confirm_credit_use=bool(execution.get("confirm_credit_use", False)),
        api_key_env=str(np_cfg.get("api_key_env", "NETSPRESSO_API_KEY")),
        disable_analytics=bool(np_cfg.get("disable_analytics", True)),
        dev_mode=bool(np_cfg.get("dev_mode", False)),
        sleep_interval_s=int(np_cfg.get("sleep_interval_s", 30)),
        reserve_credit=int(np_cfg.get("reserve_credit", 100)),
    )


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FrameworkConfig:
    models: tuple[ModelSpec, ...]
    devices: tuple[DeviceSpec, ...]
    runtimes: tuple[RuntimeSpec, ...]
    backends: tuple[BackendSpec, ...]
    optimizations: tuple[OptimizationSpec, ...]
    quality_gate: QualityGateConfig
    risk: RiskConfig
    mock: MockConfig
    source_dir: Path
    provider: ProviderConfig = ProviderConfig()

    def model(self, name: str) -> ModelSpec:
        return _lookup(self.models, name, "model")

    def device(self, name: str) -> DeviceSpec:
        return _lookup(self.devices, name, "device")

    def runtime(self, name: str) -> RuntimeSpec:
        return _lookup(self.runtimes, name, "runtime")

    def backend(self, name: str) -> BackendSpec:
        return _lookup(self.backends, name, "backend")

    def optimization(self, name: str) -> OptimizationSpec:
        return _lookup(self.optimizations, name, "optimization")


def _lookup(items: tuple[Any, ...], name: str, kind: str) -> Any:
    for item in items:
        if item.name == name:
            return item
    raise ConfigurationError(f"Unknown {kind} '{name}'")


def _bool_flags(raw: Any, where: str) -> dict[str, bool]:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ConfigurationError(f"'risk' must be a mapping in {where}")
    return {str(k): bool(v) for k, v in raw.items()}


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_models(path: Path) -> tuple[ModelSpec, ...]:
    items = _require(_load_yaml(path), "models", str(path))
    specs = []
    for raw in items:
        where = f"models.yaml/{raw.get('name', '?')}"
        specs.append(
            ModelSpec(
                name=_require(raw, "name", where),
                task=_require(raw, "task", where),
                framework=_require(raw, "framework", where),
                input_shape=tuple(int(x) for x in _require(raw, "input_shape", where)),
                accuracy_metric=raw.get("accuracy_metric", "accuracy"),
                source=raw.get("source"),
                risk=_bool_flags(raw.get("risk"), where),
                mock_baseline={str(k): float(v) for k, v in (raw.get("mock_baseline") or {}).items()},
            )
        )
    return _unique(specs, "model")


def load_devices(path: Path) -> tuple[DeviceSpec, ...]:
    items = _require(_load_yaml(path), "devices", str(path))
    specs = []
    for raw in items:
        where = f"devices.yaml/{raw.get('name', '?')}"
        rules = tuple(
            UnsupportedRule(
                runtime=rule.get("runtime"),
                backend=rule.get("backend"),
                optimization=rule.get("optimization"),
                reason=_require(rule, "reason", f"{where}/known_unsupported"),
            )
            for rule in raw.get("known_unsupported") or []
        )
        specs.append(
            DeviceSpec(
                name=_require(raw, "name", where),
                display_name=raw.get("display_name", raw["name"]),
                sdk_device_name=raw.get("sdk_device_name"),
                family=raw.get("family", "unknown"),
                risk=_bool_flags(raw.get("risk"), where),
                known_unsupported=rules,
            )
        )
    return _unique(specs, "device")


def load_runtimes(path: Path) -> tuple[RuntimeSpec, ...]:
    items = _require(_load_yaml(path), "runtimes", str(path))
    specs = [
        RuntimeSpec(
            name=_require(raw, "name", "runtimes.yaml"),
            display_name=raw.get("display_name", raw["name"]),
            sdk_framework=raw.get("sdk_framework"),
            risk=_bool_flags(raw.get("risk"), f"runtimes.yaml/{raw['name']}"),
        )
        for raw in items
    ]
    return _unique(specs, "runtime")


def load_backends(path: Path) -> tuple[BackendSpec, ...]:
    items = _require(_load_yaml(path), "backends", str(path))
    specs = [
        BackendSpec(
            name=_require(raw, "name", "backends.yaml"),
            display_name=raw.get("display_name", raw["name"]),
            applicable_runtimes=tuple(_require(raw, "applicable_runtimes", f"backends.yaml/{raw['name']}")),
            risk=_bool_flags(raw.get("risk"), f"backends.yaml/{raw['name']}"),
        )
        for raw in items
    ]
    return _unique(specs, "backend")


def load_optimizations(path: Path) -> tuple[OptimizationSpec, ...]:
    items = _require(_load_yaml(path), "optimizations", str(path))
    specs = [
        OptimizationSpec(
            name=_require(raw, "name", "optimizations.yaml"),
            display_name=raw.get("display_name", raw["name"]),
            precision=str(raw.get("precision", "unknown")),
            sdk_operation=raw.get("sdk_operation"),
            sdk_credit_constant=raw.get("sdk_credit_constant"),
            risk=_bool_flags(raw.get("risk"), f"optimizations.yaml/{raw['name']}"),
        )
        for raw in items
    ]
    return _unique(specs, "optimization")


_THRESHOLD_KEYS = ("max_drop_percent", "max_increase_percent")


def load_quality_gate(path: Path) -> QualityGateConfig:
    raw = _require(_load_yaml(path), "quality_gate", str(path))
    criteria: dict[str, CriterionConfig] = {}
    for name, body in raw.items():
        if not isinstance(body, dict):
            raise ConfigurationError(f"quality_gate.{name} must be a mapping")
        threshold = next((float(body[k]) for k in _THRESHOLD_KEYS if k in body), None)
        if threshold is not None and threshold < 0:
            raise ConfigurationError(f"quality_gate.{name}: threshold must be >= 0")
        criteria[name] = CriterionConfig(name=name, threshold=threshold, required=bool(body.get("required", True)))
    return QualityGateConfig(criteria=criteria)


def load_risk(path: Path) -> RiskConfig:
    raw = _require(_load_yaml(path), "risk", str(path))
    weights = {str(k): int(v) for k, v in _require(raw, "weights", "risk.yaml").items()}
    levels = _require(raw, "levels", "risk.yaml")
    high = int(_require(levels, "high_min_score", "risk.yaml/levels"))
    medium = int(_require(levels, "medium_min_score", "risk.yaml/levels"))
    if medium > high:
        raise ConfigurationError("risk.levels: medium_min_score must be <= high_min_score")
    include = tuple(RiskLevel(x) for x in (raw.get("selection") or {}).get("include_levels", ["HIGH", "MEDIUM"]))
    return RiskConfig(weights=weights, high_min_score=high, medium_min_score=medium, include_levels=include)


def load_mock(path: Path) -> MockConfig:
    raw = _load_yaml(path)
    profiles = {
        str(name): {str(k): float(v) for k, v in body.items()}
        for name, body in (raw.get("optimization_profiles") or {}).items()
    }
    scenarios = []
    for s in raw.get("scenarios") or []:
        where = f"mock_scenarios.yaml/{s.get('id', '?')}"
        outcome = str(_require(s, "outcome", where))
        if outcome not in {"COMPLETED", "ERROR", "BLOCKED", "UNSUPPORTED"}:
            raise ConfigurationError(f"{where}: invalid outcome '{outcome}'")
        scenarios.append(
            MockScenario(
                id=_require(s, "id", where),
                match={str(k): str(v) for k, v in _require(s, "match", where).items()},
                outcome=outcome,
                error_kind=s.get("error_kind"),
                message=s.get("message"),
                metric_overrides={str(k): float(v) for k, v in (s.get("metric_overrides") or {}).items()},
                artifact_checksum_valid=s.get("artifact_checksum_valid"),
                reproducibility=s.get("reproducibility"),
                note=s.get("note"),
            )
        )
    return MockConfig(optimization_profiles=profiles, scenarios=tuple(scenarios))


def _unique(specs: list[Any], kind: str) -> tuple[Any, ...]:
    names = [s.name for s in specs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ConfigurationError(f"Duplicate {kind} names: {dupes}")
    return tuple(specs)


def _cross_validate(cfg: FrameworkConfig) -> None:
    runtime_names = {r.name for r in cfg.runtimes}
    for b in cfg.backends:
        unknown = set(b.applicable_runtimes) - runtime_names
        if unknown:
            raise ConfigurationError(f"backend '{b.name}' references unknown runtimes {sorted(unknown)}")
    for d in cfg.devices:
        for rule in d.known_unsupported:
            if rule.runtime is not None and rule.runtime not in runtime_names:
                raise ConfigurationError(f"device '{d.name}' known_unsupported references unknown runtime '{rule.runtime}'")
    for m in cfg.models:
        if not m.mock_baseline:
            raise ConfigurationError(f"model '{m.name}' has no mock_baseline (required by MockAdapter)")


def load_config(
    config_dir: Path | str = DEFAULT_CONFIG_DIR, *, mock_scenarios: Path | str | None = None
) -> FrameworkConfig:
    """Load and cross-validate every configuration file in ``config_dir``.

    ``mock_scenarios`` optionally points to an alternative scenario file (used to
    simulate a "current" run that differs from a baseline run).
    """
    base = Path(config_dir)
    scenarios_path = Path(mock_scenarios) if mock_scenarios else base / "mock_scenarios.yaml"
    cfg = FrameworkConfig(
        models=load_models(base / "models.yaml"),
        devices=load_devices(base / "devices.yaml"),
        runtimes=load_runtimes(base / "runtimes.yaml"),
        backends=load_backends(base / "backends.yaml"),
        optimizations=load_optimizations(base / "optimizations.yaml"),
        quality_gate=load_quality_gate(base / "quality_gate.yaml"),
        risk=load_risk(base / "risk.yaml"),
        mock=load_mock(scenarios_path),
        source_dir=base,
        provider=load_provider(base / "netspresso.yaml"),
    )
    _cross_validate(cfg)
    return cfg

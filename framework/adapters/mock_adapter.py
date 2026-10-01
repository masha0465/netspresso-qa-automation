"""Deterministic MockAdapter.

* Zero network, zero credits, zero randomness. Every number is derived from the
  model's ``mock_baseline``, the optimization profile and a SHA-256 hash of the
  configuration key, so the same configuration always produces the same result.
* Scenarios in ``configs/mock_scenarios.yaml`` inject representative failures
  (accuracy regression, conversion failure, blocked environment, ...).
* Credit information is always marked ``simulated``.
"""

from __future__ import annotations

import hashlib
import platform
from collections.abc import Callable

from framework.adapters.base import BaseAdapter, ExecutionRequest
from framework.config import FrameworkConfig, MockScenario
from framework.matrix.generator import determine_support
from framework.pipeline.result import (
    Artifact,
    Configuration,
    CreditRecord,
    CreditUsageType,
    Environment,
    ExecutionError,
    ExecutionResult,
    ExecutionStatus,
    Metrics,
    Reproducibility,
    ReproducibilityLevel,
    SupportState,
    utc_now_iso,
)

_ARTIFACT_EXT = {"onnxruntime": "onnx", "tflite": "tflite", "tensorrt": "engine", "openvino": "xml"}


def _unit_fraction(seed: str, salt: str) -> float:
    """Deterministic value in [0, 1) derived from ``seed`` and ``salt``."""
    digest = hashlib.sha256(f"{seed}::{salt}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _jitter(seed: str, salt: str, amplitude: float) -> float:
    """Deterministic multiplicative jitter in [1 - amplitude, 1 + amplitude]."""
    return 1.0 + (2.0 * _unit_fraction(seed, salt) - 1.0) * amplitude


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class MockAdapter(BaseAdapter):
    name = "mock"
    version = "1.0.0"
    credit_consuming = False

    def __init__(self, config: FrameworkConfig, clock: Callable[[], str] | None = None) -> None:
        self._config = config
        self._clock = clock or utc_now_iso

    # ------------------------------------------------------------------ #
    # BaseAdapter contract
    # ------------------------------------------------------------------ #
    def describe_environment(self) -> Environment:
        return Environment(
            adapter=self.name,
            adapter_version=self.version,
            sdk_name=None,
            sdk_version=None,
            python_version=platform.python_version(),
            platform=platform.platform(),
            extra={"deterministic": True, "network": False, "credits": "simulated"},
        )

    def check_support(self, configuration: Configuration) -> tuple[SupportState, str | None]:
        return determine_support(self._config, configuration)

    def run(self, request: ExecutionRequest) -> ExecutionResult:
        cfg = request.configuration
        scenario = self._find_scenario(cfg)
        env = self.describe_environment()
        optimization = self._config.optimization(cfg.optimization)
        credit = CreditRecord(
            usage_type=CreditUsageType.SIMULATED,
            estimated=optimization.sdk_credit_constant,
            actual=None,
            note="simulated by MockAdapter; no real credits were consumed",
        )
        common = {
            "operation": request.operation,
            "configuration": cfg,
            "environment": env,
            "execution_time_s": round(0.4 + 2.6 * _unit_fraction(cfg.key, "exec_time"), 3),
            "timestamp": self._clock(),
            "credit": credit,
            "scenario_id": scenario.id if scenario else None,
        }
        logs = [f"[mock] request operation={request.operation} case={cfg.case_id}"]
        if scenario:
            logs.append(f"[mock] scenario {scenario.id} matched: {scenario.note or scenario.outcome}")

        if scenario and scenario.outcome != "COMPLETED":
            status = ExecutionStatus(scenario.outcome)
            logs.append(f"[mock] execution status {status.value}")
            return ExecutionResult(
                execution_status=status,
                logs=logs,
                error=ExecutionError(
                    message=scenario.message or f"{status.value} (mock)",
                    kind=scenario.error_kind,
                    details={"scenario_id": scenario.id},
                ),
                **common,
            )

        baseline, current = self._derive_metrics(cfg, scenario)
        artifact = self._derive_artifact(cfg, current, scenario, env)
        repro = self._derive_reproducibility(cfg, request.repeat_runs, scenario, artifact)
        logs += [
            "[mock] baseline evaluated (local, deterministic)",
            f"[mock] optimization '{cfg.optimization}' applied",
            f"[mock] artifact written: {artifact.path}",
            f"[mock] reproducibility runs={repro.runs} level={repro.level.value}",
        ]
        return ExecutionResult(
            execution_status=ExecutionStatus.COMPLETED,
            baseline_metrics=baseline,
            metrics=current,
            artifact=artifact,
            logs=logs,
            reproducibility=repro,
            **common,
        )

    # ------------------------------------------------------------------ #
    # Deterministic derivations
    # ------------------------------------------------------------------ #
    def _find_scenario(self, cfg: Configuration) -> MockScenario | None:
        values = cfg.to_dict()
        return next((s for s in self._config.mock.scenarios if s.matches(values)), None)

    def _derive_metrics(self, cfg: Configuration, scenario: MockScenario | None) -> tuple[Metrics, Metrics]:
        base = self._config.model(cfg.model).mock_baseline
        profile = dict(self._config.mock.optimization_profiles.get(cfg.optimization, {}))
        if scenario:
            profile.update(scenario.metric_overrides)

        baseline = Metrics(
            accuracy=base.get("accuracy"),
            latency_ms=round(base.get("latency_ms", 0.0) * self._device_factor(cfg.device), 3),
            memory_mb=base.get("memory_mb"),
            model_size_mb=base.get("model_size_mb"),
        )
        acc_delta_pp = profile.get("accuracy_delta_pp", 0.0) + (2 * _unit_fraction(cfg.key, "acc") - 1) * 0.05
        current = Metrics(
            accuracy=round((baseline.accuracy or 0.0) + acc_delta_pp / 100.0, 4),
            latency_ms=round((baseline.latency_ms or 0.0) * profile.get("latency_factor", 1.0) * _jitter(cfg.key, "lat", 0.03), 3),
            memory_mb=round((baseline.memory_mb or 0.0) * profile.get("memory_factor", 1.0) * _jitter(cfg.key, "mem", 0.02), 3),
            model_size_mb=round((baseline.model_size_mb or 0.0) * profile.get("size_factor", 1.0), 3),
        )
        return baseline, current

    @staticmethod
    def _device_factor(device: str) -> float:
        # Purely illustrative relative speed factors for the mock; not measurements.
        return {"intel_xeon_w2233": 1.0, "raspberry_pi_4b": 6.5, "jetson_orin_nano": 0.6}.get(device, 1.0)

    def _derive_artifact(
        self, cfg: Configuration, metrics: Metrics, scenario: MockScenario | None, env: Environment
    ) -> Artifact:
        ext = _ARTIFACT_EXT.get(cfg.runtime, "bin")
        path = f"mock://artifacts/{cfg.case_id}/{cfg.model}_{cfg.optimization}.{ext}"
        checksum = _sha256_text(f"{cfg.key}::artifact")
        valid = True if scenario is None or scenario.artifact_checksum_valid is None else scenario.artifact_checksum_valid
        expected = checksum if valid else _sha256_text(f"{cfg.key}::artifact::expected")
        return Artifact(
            path=path,
            exists=True,
            size_bytes=int((metrics.model_size_mb or 0.0) * 1_000_000),
            format=ext,
            checksum_sha256=checksum,
            expected_checksum_sha256=expected,
            checksum_valid=valid,
            metadata={
                "source_model": cfg.model,
                "optimization": cfg.optimization,
                "runtime": cfg.runtime,
                "device": cfg.device,
                "backend": cfg.backend,
                "adapter": f"{env.adapter} {env.adapter_version}",
                "sdk_version": env.sdk_version,
                "configuration_key": cfg.key,
                "generated_by": "MockAdapter (simulated artifact, no file on disk)",
            },
        )

    @staticmethod
    def _derive_reproducibility(
        cfg: Configuration, repeat_runs: int, scenario: MockScenario | None, artifact: Artifact
    ) -> Reproducibility:
        if repeat_runs < 2:
            return Reproducibility(level=ReproducibilityLevel.NOT_VERIFIED, runs=repeat_runs, notes="single run")
        level = ReproducibilityLevel(scenario.reproducibility) if scenario and scenario.reproducibility else ReproducibilityLevel.BITWISE_REPRODUCIBLE
        first = artifact.checksum_sha256 or _sha256_text(cfg.key)
        if level == ReproducibilityLevel.BITWISE_REPRODUCIBLE:
            checksums = [first] * repeat_runs
            notes = "identical artifact checksums across runs (mock)"
        elif level == ReproducibilityLevel.FUNCTIONALLY_REPRODUCIBLE:
            checksums = [first] + [_sha256_text(f"{cfg.key}::run{i}") for i in range(1, repeat_runs)]
            notes = "checksums differ but outputs within tolerance (mock)"
        else:
            checksums = [_sha256_text(f"{cfg.key}::run{i}") for i in range(repeat_runs)]
            notes = "outputs diverge beyond tolerance (mock)"
        return Reproducibility(level=level, runs=repeat_runs, checksums=checksums, notes=notes)

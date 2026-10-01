"""Real NetsPresso adapter — safety boundary (Phase 5-A: dry-run only).

Design
------
* Same :class:`~framework.adapters.base.BaseAdapter` contract as ``MockAdapter``.
* The SDK is **never imported at module import time**. Even in a future real run it
  is loaded lazily via :func:`importlib.import_module`, so the Python 3.14 main
  framework can import this module while the SDK only exists in the Python 3.11
  environment (``.venv-netspresso``).
* The API key is **never stored, logged or rendered**. The adapter only knows the
  *name* of the environment variable and whether a value is present.
* Execution policy (:class:`ExecutionMode`):

  ==========================  ===========================================================
  DRY_RUN (default)           ``run()``/operation methods return a structured
                              :class:`DryRunPlan`; no SDK import, no network, 0 credits.
  REAL_RUN_UNAUTHORIZED       mode = real but ``confirm_credit_use`` is false ->
                              :class:`RealExecutionNotAuthorizedError` before anything happens.
  REAL_RUN_AUTHORIZED         mode = real and confirmed. **Phase 5-A deliberately raises
                              :class:`RealExecutionNotImplementedError`** before importing the
                              SDK. The real path is specified (see ``SDK_OPERATIONS``) but
                              not executed, because it cannot be exercised without
                              consuming credits.
  ==========================  ===========================================================

Every SDK name referenced here was verified against the installed ``netspresso==1.17.0``
in Phase 1 (``docs/research_cache/sdk_surface.json``, ``docs/netspresso_sdk_research.md``).
Credit figures are the SDK's **client-side pre-check constants**, not verified billing.
"""

from __future__ import annotations

import enum
import importlib
import os
import platform
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from framework.adapters.base import BaseAdapter, ExecutionRequest
from framework.config import FrameworkConfig, ProviderConfig
from framework.matrix.generator import determine_support
from framework.pipeline.result import (
    Configuration,
    CreditRecord,
    CreditUsageType,
    Environment,
    ExecutionResult,
    ExecutionStatus,
    SupportState,
    utc_now_iso,
)

SDK_PACKAGE = "netspresso"
SDK_VERSION_RESEARCHED = "1.17.0"
CREDIT_BASIS = (
    "CLIENT_SIDE_PRE_CHECK_CONSTANT (netspresso 1.17.0, netspresso/enums/credit.py); "
    "actual server-side deduction NOT verified"
)


class ExecutionMode(enum.StrEnum):
    DRY_RUN = "DRY_RUN"
    REAL_RUN_UNAUTHORIZED = "REAL_RUN_UNAUTHORIZED"
    REAL_RUN_AUTHORIZED = "REAL_RUN_AUTHORIZED"


class RealExecutionNotAuthorizedError(RuntimeError):
    """mode = real requested without explicit credit-use confirmation."""


class RealExecutionNotImplementedError(RuntimeError):
    """Real execution is authorized but intentionally not implemented in this phase."""


class SdkUnavailableError(RuntimeError):
    """The netspresso SDK is not importable in the current interpreter."""


# --------------------------------------------------------------------------- #
# Verified SDK operation table
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SdkOperation:
    name: str  # framework-level operation name
    service_factory: str  # NetsPresso.<factory>()  (verified facade method)
    method: str  # service method (verified signature)
    service_task: str  # netspresso.enums.ServiceTask member name
    client_credit_constant: int | None  # ServiceCredit.CREDITS value; None = not in table
    client_precheck: bool  # SDK calls check_credit_balance() before the request
    required_inputs: tuple[str, ...]
    notes: str


SDK_OPERATIONS: dict[str, SdkOperation] = {
    "validate_connection": SdkOperation(
        name="validate_connection",
        service_factory="NetsPresso",
        method="__init__(api_key=...) -> login_by_api_key + get_user()",
        service_task="-",
        client_credit_constant=None,
        client_precheck=False,
        required_inputs=("api_key (from environment)",),
        notes=(
            "Object construction authenticates immediately and queries /users/me and credit summary. "
            "Constructor also checks PyPI for a newer release and calls sys.exit(1) if stale (dev_mode=True skips). "
            "Not in the credit table; billing of auth/query calls is unverified (presumed none)."
        ),
    ),
    "automatic_compression": SdkOperation(
        name="automatic_compression",
        service_factory="compressor_v2",
        method="automatic_compression",
        service_task="AUTOMATIC_COMPRESSION",
        client_credit_constant=25,
        client_precheck=True,
        required_inputs=("input_model_path", "output_dir", "input_shapes", "framework", "compression_ratio=0.5"),
        notes="Synchronous; result metadata saved to output_dir/metadata.json.",
    ),
    "fp16_conversion": SdkOperation(
        name="fp16_conversion",
        service_factory="converter_v2",
        method="convert_model",
        service_task="MODEL_CONVERT",
        client_credit_constant=50,
        client_precheck=True,
        required_inputs=("input_model_path", "output_dir", "target_framework", "target_device_name", "target_data_type=FP16"),
        notes="Asynchronous; polled with sleep_interval (default 30 s) when wait_until_done=True.",
    ),
    "int8_quantization": SdkOperation(
        name="int8_quantization",
        service_factory="quantizer",
        method="automatic_quantization",
        service_task="MODEL_QUANTIZE",
        client_credit_constant=50,
        client_precheck=True,
        required_inputs=("input_model_path", "output_dir", "dataset_path (.npy calibration)", "weight_precision=INT8", "activation_precision=INT8"),
        notes="Asynchronous; polled with sleep_interval.",
    ),
    "profile": SdkOperation(
        name="profile",
        service_factory="profiler",
        method="profile_model",
        service_task="MODEL_PROFILE",
        client_credit_constant=25,
        client_precheck=True,
        required_inputs=("input_model_path", "target_device_name"),
        notes="Asynchronous; BenchmarkResult fields are int (unit/precision unverified).",
    ),
    "graph_optimize": SdkOperation(
        name="graph_optimize",
        service_factory="graph_optimizer",
        method="optimize_model",
        service_task="MODEL_GRAPH_OPTIMIZE",
        client_credit_constant=50,
        client_precheck=False,
        required_inputs=("input_model_path", "output_dir"),
        notes="Credit constant exists but the SDK performs NO client-side pre-check; not assumed free.",
    ),
}


# --------------------------------------------------------------------------- #
# Dry-run plan
# --------------------------------------------------------------------------- #
@dataclass
class DryRunPlan:
    execution_mode: str
    operation: str
    sdk_service: str
    sdk_method: str
    service_task: str
    api_key_env: str
    api_key_present: bool
    estimated_credit: int | None
    model: str | None = None
    configuration: dict[str, str] | None = None
    sdk_device_name: str | None = None
    sdk_framework: str | None = None
    provider: str = "NetsPresso"
    credit_basis: str = CREDIT_BASIS
    actual_credit: int | None = None  # NOT MEASURED in a dry run
    api_call_executed: bool = False
    credit_consumed: int = 0
    sdk_loaded: bool = False
    required_inputs: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "execution_mode": self.execution_mode,
            "operation": self.operation,
            "sdk_service": self.sdk_service,
            "sdk_method": self.sdk_method,
            "service_task": self.service_task,
            "model": self.model,
            "configuration": dict(self.configuration) if self.configuration else None,
            "sdk_device_name": self.sdk_device_name,
            "sdk_framework": self.sdk_framework,
            "estimated_credit": self.estimated_credit,
            "credit_basis": self.credit_basis,
            "actual_credit": self.actual_credit,
            "api_call_executed": self.api_call_executed,
            "credit_consumed": self.credit_consumed,
            "sdk_loaded": self.sdk_loaded,
            "api_key_env": self.api_key_env,
            "api_key_present": self.api_key_present,
            "required_inputs": list(self.required_inputs),
            "preconditions": list(self.preconditions),
            "notes": list(self.notes),
        }

    def render(self) -> str:
        est = "unknown (not in client-side table)" if self.estimated_credit is None else f"{self.estimated_credit} ({self.credit_basis})"
        lines = [
            f"Provider:          {self.provider}",
            f"Execution:         {self.execution_mode}",
            f"Operation:         {self.operation}",
            f"SDK call:          NetsPresso.{self.sdk_service}().{self.sdk_method}" if self.sdk_service != "NetsPresso" else f"SDK call:          NetsPresso.{self.sdk_method}",
            f"Model:             {self.model or '-'}",
            f"Configuration:     {self.configuration['device'] + ' / ' + self.configuration['runtime'] + ' / ' + self.configuration['backend'] if self.configuration else '-'}",
            f"SDK device name:   {self.sdk_device_name or '-'}",
            f"SDK framework:     {self.sdk_framework or '-'}",
            f"Estimated Credit:  {est}",
            "Actual Credit:     NOT MEASURED",
            f"API call:          {'EXECUTED' if self.api_call_executed else 'NOT EXECUTED'}",
            f"SDK loaded:        {'yes' if self.sdk_loaded else 'no'}",
            f"API key:           {self.api_key_env} {'present' if self.api_key_present else 'absent'} (value never read in dry run)",
            f"Credit consumed:   {self.credit_consumed}",
        ]
        if self.required_inputs:
            lines.append("Required inputs:   " + ", ".join(self.required_inputs))
        for p in self.preconditions:
            lines.append(f"Precondition:      {p}")
        for n in self.notes:
            lines.append(f"Note:              {n}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #
class NetsPressoAdapter(BaseAdapter):
    name = "netspresso"
    version = "0.1.0-dry-run"

    def __init__(
        self,
        config: FrameworkConfig,
        settings: ProviderConfig | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._config = config
        self._settings = settings or config.provider
        self._env: Mapping[str, str] = os.environ if env is None else env
        self._sdk: Any = None  # stays None in Phase 5-A
        self._api_calls_executed = 0

    # ------------------------------------------------------------------ #
    # policy
    # ------------------------------------------------------------------ #
    @property
    def settings(self) -> ProviderConfig:
        return self._settings

    @property
    def mode(self) -> ExecutionMode:
        if self._settings.mode == "dry_run":
            return ExecutionMode.DRY_RUN
        return ExecutionMode.REAL_RUN_AUTHORIZED if self._settings.confirm_credit_use else ExecutionMode.REAL_RUN_UNAUTHORIZED

    @property
    def credit_consuming(self) -> bool:  # type: ignore[override]
        """Only a real run can consume credits; a dry run provably never reaches the SDK."""
        return self.mode != ExecutionMode.DRY_RUN

    @property
    def api_key_present(self) -> bool:
        return bool(self._env.get(self._settings.api_key_env))

    @property
    def api_calls_executed(self) -> int:
        return self._api_calls_executed

    def __repr__(self) -> str:  # never includes the key value
        return f"NetsPressoAdapter(mode={self.mode.value}, api_key_env={self._settings.api_key_env!r}, api_key_present={self.api_key_present})"

    def _enforce_policy(self, operation: str) -> None:
        mode = self.mode
        if mode == ExecutionMode.DRY_RUN:
            return
        if mode == ExecutionMode.REAL_RUN_UNAUTHORIZED:
            raise RealExecutionNotAuthorizedError(
                f"operation '{operation}' requested in real mode without explicit credit-use confirmation "
                "(set execution.confirm_credit_use=true or pass --confirm-credit-use). No API call was made."
            )
        raise RealExecutionNotImplementedError(
            f"operation '{operation}' is authorized but the real execution path is intentionally not implemented "
            "in Phase 5-A. It requires the Python 3.11 SDK environment, explicit per-run approval and a credit "
            "ledger entry (Phase 5-B). No API call was made."
        )

    # ------------------------------------------------------------------ #
    # BaseAdapter contract
    # ------------------------------------------------------------------ #
    def describe_environment(self) -> Environment:
        return Environment(
            adapter=self.name,
            adapter_version=self.version,
            sdk_name=SDK_PACKAGE,
            sdk_version=None,  # unknown until the SDK is actually loaded in a real run
            python_version=platform.python_version(),
            platform=platform.platform(),
            extra={
                "execution_mode": self.mode.value,
                "sdk_loaded": self._sdk is not None,
                "sdk_version_researched": SDK_VERSION_RESEARCHED,
                "api_calls_executed": self._api_calls_executed,
                "api_key_env": self._settings.api_key_env,
                "api_key_present": self.api_key_present,
                "credits": "none consumed",
            },
        )

    def check_support(self, configuration: Configuration) -> tuple[SupportState, str | None]:
        # Same explicit knowledge as the mock; server-side support is only learned by a real run.
        return determine_support(self._config, configuration)

    def run(self, request: ExecutionRequest) -> ExecutionResult:
        self._enforce_policy(request.operation)
        plan = self.plan(request.operation, request.configuration, request.parameters)
        return ExecutionResult(
            operation=request.operation,
            configuration=request.configuration,
            execution_status=ExecutionStatus.NOT_EXECUTED,
            environment=self.describe_environment(),
            execution_time_s=0.0,
            logs=["[netspresso] DRY RUN - plan only, no SDK import, no network"] + plan.render().splitlines(),
            credit=CreditRecord(
                usage_type=CreditUsageType.NONE,
                estimated=plan.estimated_credit,
                actual=None,
                note=f"dry run: 0 credits consumed; estimate basis = {CREDIT_BASIS}",
            ),
            timestamp=utc_now_iso(),
        )

    # ------------------------------------------------------------------ #
    # operation model (future real API surface; dry-run plans in Phase 5-A)
    # ------------------------------------------------------------------ #
    def validate_connection(self) -> DryRunPlan:
        self._enforce_policy("validate_connection")
        return self.plan("validate_connection")

    def optimize(self, configuration: Configuration, **parameters: Any) -> DryRunPlan:
        self._enforce_policy("automatic_compression")
        return self.plan("automatic_compression", configuration, parameters)

    def quantize(self, configuration: Configuration, **parameters: Any) -> DryRunPlan:
        self._enforce_policy("int8_quantization")
        return self.plan("int8_quantization", configuration, parameters)

    def convert(self, configuration: Configuration, **parameters: Any) -> DryRunPlan:
        self._enforce_policy("fp16_conversion")
        return self.plan("fp16_conversion", configuration, parameters)

    def profile(self, configuration: Configuration, **parameters: Any) -> DryRunPlan:
        self._enforce_policy("profile")
        return self.plan("profile", configuration, parameters)

    def plan(
        self,
        operation: str,
        configuration: Configuration | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> DryRunPlan:
        """Build the structured dry-run plan. Pure function of config + policy; no side effects."""
        sdk_op = SDK_OPERATIONS.get(operation)
        if sdk_op is None:
            raise ValueError(f"unknown operation '{operation}'; known: {sorted(SDK_OPERATIONS)}")

        estimated = sdk_op.client_credit_constant
        model = device = framework = None
        preconditions = [
            "Python 3.11 environment (.venv-netspresso) with netspresso==1.17.0",
            f"{self._settings.api_key_env} set in the environment (never in YAML or source)",
            "explicit approval: execution.mode=real AND confirm_credit_use=true (or --confirm-credit-use)",
            f"credit ledger budget check: remaining - estimate >= reserve ({self._settings.reserve_credit})",
        ]
        notes = [sdk_op.notes]
        if self._settings.disable_analytics:
            notes.append("GA_DISABLE_ANALYTICS=1 will be set before SDK import (SDK sends GA4 events otherwise)")
        if not sdk_op.client_precheck and sdk_op.client_credit_constant is not None:
            notes.append("SDK does not pre-check balance for this task; treat as credit-consuming anyway")

        if configuration is not None:
            model = configuration.model
            device = self._config.device(configuration.device).sdk_device_name
            framework = self._config.runtime(configuration.runtime).sdk_framework
            opt = next((o for o in self._config.optimizations if o.name == operation), None)
            if opt is not None and opt.sdk_credit_constant is not None:
                estimated = opt.sdk_credit_constant
            support, reason = determine_support(self._config, configuration)
            if support == SupportState.UNSUPPORTED:
                preconditions.append(f"combination is UNSUPPORTED by static knowledge: {reason}")
            else:
                notes.append("server-side support for this combination is UNKNOWN until a real run (starts as NOT_TESTED)")
        if parameters:
            notes.append("parameters: " + ", ".join(f"{k}={v}" for k, v in sorted(parameters.items())))

        return DryRunPlan(
            execution_mode=self.mode.value,
            operation=operation,
            sdk_service=sdk_op.service_factory,
            sdk_method=sdk_op.method,
            service_task=sdk_op.service_task,
            api_key_env=self._settings.api_key_env,
            api_key_present=self.api_key_present,
            estimated_credit=estimated,
            model=model,
            configuration=configuration.to_dict() if configuration else None,
            sdk_device_name=device,
            sdk_framework=framework,
            sdk_loaded=self._sdk is not None,
            required_inputs=list(sdk_op.required_inputs),
            preconditions=preconditions,
            notes=notes,
        )

    # ------------------------------------------------------------------ #
    # SDK loading (unreachable in Phase 5-A; kept explicit for Phase 5-B)
    # ------------------------------------------------------------------ #
    def _load_sdk(self) -> Any:
        """Lazily import the SDK. Only legal after :meth:`_enforce_policy` allowed a real run."""
        if self.mode != ExecutionMode.REAL_RUN_AUTHORIZED:
            raise RealExecutionNotAuthorizedError("SDK loading is only permitted for an authorized real run")
        if self._settings.disable_analytics:
            os.environ.setdefault("GA_DISABLE_ANALYTICS", "1")
        try:
            self._sdk = importlib.import_module(SDK_PACKAGE)
        except ImportError as exc:  # pragma: no cover - depends on interpreter
            raise SdkUnavailableError(
                f"'{SDK_PACKAGE}' is not importable in {platform.python_version()}; use the Python 3.11 "
                "environment (.venv-netspresso). See docs/netspresso_sdk_research.md §2."
            ) from exc
        return self._sdk

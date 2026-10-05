"""Real NetsPresso adapter — safety boundary with a single implemented real operation.

Design
------
* Same :class:`~framework.adapters.base.BaseAdapter` contract as ``MockAdapter``.
* The SDK is **never imported at module import time**. It is loaded lazily via
  :func:`importlib.import_module` only for an *authorized* real run, so the Python 3.14
  main framework can import this module while the SDK exists only in the Python 3.11
  environment (``.venv-netspresso``).
* The API key is **never stored, logged or rendered**. The adapter only knows the *name*
  of the environment variable and whether a value is present; the value is read
  immediately before authentication and redacted from every message afterwards.
* Execution policy (:class:`ExecutionMode`):

  ==========================  ===========================================================
  DRY_RUN (default)           ``run()``/operation methods return a structured
                              :class:`DryRunPlan`; no SDK import, no network, 0 credits.
  REAL_RUN_UNAUTHORIZED       mode = real but ``confirm_credit_use`` is false ->
                              :class:`RealExecutionNotAuthorizedError` before anything happens.
  REAL_RUN_AUTHORIZED         mode = real and confirmed. Only operations listed in
                              ``REAL_IMPLEMENTED_OPERATIONS`` execute (Phase 5-B:
                              ``automatic_compression``). Everything else raises
                              :class:`RealExecutionNotImplementedError` before any SDK import.
  ==========================  ===========================================================

Every SDK name referenced here was verified against the installed ``netspresso==1.17.0``
(``docs/research_cache/sdk_surface.json``, ``docs/netspresso_sdk_research.md``) and
re-verified by ``inspect.signature`` immediately before Phase 5-B.
Credit figures are the SDK's **client-side pre-check constants**, not verified billing;
the real path records the account balance before/after so that the *observed* deduction
can be reported separately.
"""

from __future__ import annotations

import enum
import importlib
import os
import platform
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from framework.adapters.base import BaseAdapter, ExecutionRequest
from framework.config import FrameworkConfig, ProviderConfig
from framework.matrix.generator import determine_support
from framework.pipeline.result import (
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
from framework.validation.artifact import artifact_from_file

SDK_PACKAGE = "netspresso"
SDK_VERSION_RESEARCHED = "1.17.0"
CREDIT_BASIS = (
    "CLIENT_SIDE_PRE_CHECK_CONSTANT (netspresso 1.17.0, netspresso/enums/credit.py); "
    "actual server-side deduction NOT verified"
)
REAL_IMPLEMENTED_OPERATIONS: tuple[str, ...] = ("automatic_compression",)
REDACTED = "<redacted>"


class ExecutionMode(enum.StrEnum):
    DRY_RUN = "DRY_RUN"
    REAL_RUN_UNAUTHORIZED = "REAL_RUN_UNAUTHORIZED"
    REAL_RUN_AUTHORIZED = "REAL_RUN_AUTHORIZED"


class RealExecutionNotAuthorizedError(RuntimeError):
    """mode = real requested without explicit credit-use confirmation."""


class RealExecutionNotImplementedError(RuntimeError):
    """Real execution is authorized but this operation has no implemented real path yet."""


class MissingApiKeyError(RuntimeError):
    """The configured API key environment variable is not set (value never inspected)."""


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
        sdk_call = f"NetsPresso.{self.sdk_method}" if self.sdk_service == "NetsPresso" else f"NetsPresso.{self.sdk_service}().{self.sdk_method}"
        cfg = f"{self.configuration['device']} / {self.configuration['runtime']} / {self.configuration['backend']}" if self.configuration else "-"
        lines = [
            f"Provider:          {self.provider}",
            f"Execution:         {self.execution_mode}",
            f"Operation:         {self.operation}",
            f"SDK call:          {sdk_call}",
            f"Model:             {self.model or '-'}",
            f"Configuration:     {cfg}",
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
        lines.extend(f"Precondition:      {p}" for p in self.preconditions)
        lines.extend(f"Note:              {n}" for n in self.notes)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #
class NetsPressoAdapter(BaseAdapter):
    name = "netspresso"
    version = "0.2.0"

    def __init__(
        self,
        config: FrameworkConfig,
        settings: ProviderConfig | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._config = config
        self._settings = settings or config.provider
        self._env: Mapping[str, str] = os.environ if env is None else env
        self._sdk: Any = None
        self._sdk_version: str | None = None
        self._api_calls_executed = 0  # authenticated network sessions opened
        self._operations_executed = 0  # credit-relevant service operations attempted

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

    @property
    def operations_executed(self) -> int:
        return self._operations_executed

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
        # REAL_RUN_AUTHORIZED: allowed; the caller decides whether a real path exists.

    def _require_real_implementation(self, operation: str) -> None:
        if operation not in REAL_IMPLEMENTED_OPERATIONS:
            raise RealExecutionNotImplementedError(
                f"operation '{operation}' is authorized but has no implemented real execution path "
                f"(implemented: {list(REAL_IMPLEMENTED_OPERATIONS)}). No API call was made."
            )

    # ------------------------------------------------------------------ #
    # BaseAdapter contract
    # ------------------------------------------------------------------ #
    def describe_environment(self) -> Environment:
        return Environment(
            adapter=self.name,
            adapter_version=self.version,
            sdk_name=SDK_PACKAGE,
            sdk_version=self._sdk_version,  # known only after the SDK was actually loaded
            python_version=platform.python_version(),
            platform=platform.platform(),
            extra={
                "execution_mode": self.mode.value,
                "sdk_loaded": self._sdk is not None,
                "sdk_version_researched": SDK_VERSION_RESEARCHED,
                "api_calls_executed": self._api_calls_executed,
                "operations_executed": self._operations_executed,
                "api_key_env": self._settings.api_key_env,
                "api_key_present": self.api_key_present,
            },
        )

    def check_support(self, configuration: Configuration) -> tuple[SupportState, str | None]:
        # Same explicit knowledge as the mock; server-side support is only learned by a real run.
        return determine_support(self._config, configuration)

    def run(self, request: ExecutionRequest) -> ExecutionResult:
        self._enforce_policy(request.operation)
        if self.mode == ExecutionMode.DRY_RUN:
            return self._dry_run_result(request)
        return self._execute_real(request)

    # ------------------------------------------------------------------ #
    # operation model (dry-run plans; real path only via run())
    # ------------------------------------------------------------------ #
    def validate_connection(self) -> DryRunPlan:
        self._enforce_policy("validate_connection")
        if self.mode == ExecutionMode.REAL_RUN_AUTHORIZED:
            self._require_real_implementation("validate_connection")
        return self.plan("validate_connection")

    def optimize(self, configuration: Configuration, **parameters: Any) -> DryRunPlan:
        return self._planned("automatic_compression", configuration, parameters)

    def quantize(self, configuration: Configuration, **parameters: Any) -> DryRunPlan:
        return self._planned("int8_quantization", configuration, parameters)

    def convert(self, configuration: Configuration, **parameters: Any) -> DryRunPlan:
        return self._planned("fp16_conversion", configuration, parameters)

    def profile(self, configuration: Configuration, **parameters: Any) -> DryRunPlan:
        return self._planned("profile", configuration, parameters)

    def _planned(self, operation: str, configuration: Configuration, parameters: Mapping[str, Any]) -> DryRunPlan:
        self._enforce_policy(operation)
        if self.mode == ExecutionMode.REAL_RUN_AUTHORIZED:
            # Plans are always safe to produce; real execution goes through run() only.
            self._require_real_implementation(operation)
        return self.plan(operation, configuration, parameters)

    def plan(
        self,
        operation: str,
        configuration: Configuration | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> DryRunPlan:
        """Build the structured plan. Pure function of config + policy; no side effects."""
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
            notes.append("parameters: " + ", ".join(f"{k}={self._redact(str(v))}" for k, v in sorted(parameters.items())))

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

    def _dry_run_result(self, request: ExecutionRequest) -> ExecutionResult:
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
    # Real execution path (authorized runs only)
    # ------------------------------------------------------------------ #
    def _redact(self, text: str) -> str:
        key = self._env.get(self._settings.api_key_env)
        return text.replace(key, REDACTED) if key else text

    def _load_sdk(self) -> Any:
        """Lazily import the SDK. Only legal after the policy allowed a real run."""
        if self.mode != ExecutionMode.REAL_RUN_AUTHORIZED:
            raise RealExecutionNotAuthorizedError("SDK loading is only permitted for an authorized real run")
        if self._sdk is not None:
            return self._sdk
        if self._settings.disable_analytics:
            os.environ.setdefault("GA_DISABLE_ANALYTICS", "1")
        try:
            self._sdk = importlib.import_module(SDK_PACKAGE)
        except ImportError as exc:
            raise SdkUnavailableError(
                f"'{SDK_PACKAGE}' is not importable in Python {platform.python_version()}; use the Python 3.11 "
                "environment (.venv-netspresso). See docs/netspresso_sdk_research.md §2."
            ) from exc
        self._sdk_version = getattr(self._sdk, "__version__", None)
        return self._sdk

    def _attach_sdk_log_sink(self, path: Path) -> None:
        """Mirror the SDK's loguru output to a file, with the API key redacted."""
        try:
            loguru = importlib.import_module("loguru")
        except ImportError:  # pragma: no cover - loguru ships with the SDK
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        loguru.logger.add(str(path), level="INFO", format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
                          filter=lambda record: record.update(message=self._redact(record["message"])) or True)

    def _execute_real(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute exactly one real SDK operation and map the result into the common Result Model.

        Ordering is deliberate: policy -> implemented? -> API key present? -> SDK import ->
        authenticate (network) -> ONE service call -> balance re-query -> mapping. No retries.
        """
        operation = request.operation
        self._require_real_implementation(operation)
        api_key = self._env.get(self._settings.api_key_env)
        if not api_key:
            raise MissingApiKeyError(f"{self._settings.api_key_env} is not set; refusing to authenticate (no API call made)")

        params = dict(request.parameters)
        for required in ("input_model_path", "output_dir", "input_shapes"):
            if required not in params:
                raise ValueError(f"real {operation} requires parameter '{required}'")
        input_model_path = Path(params["input_model_path"])
        if not input_model_path.is_file():
            raise FileNotFoundError(f"input model not found: {input_model_path}")

        sdk = self._load_sdk()
        enums = importlib.import_module(f"{SDK_PACKAGE}.enums")
        if params.get("sdk_log_path"):
            self._attach_sdk_log_sink(Path(params["sdk_log_path"]))

        env = self.describe_environment()
        logs: list[str] = [
            f"[netspresso] REAL RUN authorized: operation={operation} sdk_version={self._sdk_version}",
            f"[netspresso] input model: {input_model_path.as_posix()} ({input_model_path.stat().st_size} bytes)",
        ]
        started = utc_now_iso()
        t0 = time.perf_counter()

        # --- 1. authenticate (network): login_by_api_key + /users/me + credit summary ---
        try:
            client = sdk.NetsPresso(api_key=api_key, dev_mode=self._settings.dev_mode)
        except SystemExit as exc:
            logs.append("[netspresso] SDK constructor exited (version gate or PyPI unreachable); operation NOT attempted")
            return self._failed(request, env, ExecutionStatus.BLOCKED, "SdkVersionGateExit",
                                f"NetsPresso() called sys.exit({exc.code}) before authentication completed", logs, started, t0)
        except Exception as exc:  # noqa: BLE001 - map any auth failure into the result model
            logs.append("[netspresso] authentication failed; operation NOT attempted")
            return self._failed(request, env, ExecutionStatus.ERROR, type(exc).__name__,
                                f"authentication failed: {self._redact(str(exc))}", logs, started, t0)
        self._api_calls_executed += 1
        credit_before = self._credit_total(getattr(client, "user_info", None))
        logs.append(f"[netspresso] authenticated; account credit total before: {credit_before}")

        # --- 2. exactly ONE service operation ---
        framework_value = str(params.get("framework", "pytorch"))
        framework_enum = enums.Framework(framework_value)
        ratio = float(params.get("compression_ratio", 0.5))
        output_dir = Path(params["output_dir"])
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        self._operations_executed += 1
        logs.append(f"[netspresso] calling compressor_v2().automatic_compression(ratio={ratio}, framework={framework_value}, input_shapes={params['input_shapes']})")
        try:
            metadata = client.compressor_v2().automatic_compression(
                input_model_path=input_model_path.as_posix(),
                output_dir=output_dir.as_posix(),
                input_shapes=list(params["input_shapes"]),
                framework=framework_enum,
                compression_ratio=ratio,
            )
        except Exception as exc:  # noqa: BLE001 - SDK normally returns ERROR metadata; this is a last resort
            credit_after = self._query_credit(client, logs)
            logs.append("[netspresso] service call raised an exception; no retry")
            result = self._failed(request, env, ExecutionStatus.ERROR, type(exc).__name__, self._redact(str(exc)), logs, started, t0)
            result.credit = self._credit_record(credit_before, credit_after)
            return result

        # --- 3. re-query balance (auth/query call, not a service operation) ---
        credit_after = self._query_credit(client, logs)

        # --- 4. map SDK metadata -> Result Model ---
        return self._map_compression_metadata(request, env, metadata, input_model_path, credit_before, credit_after, logs, started, t0)

    # ------------------------------------------------------------------ #
    # helpers for the real path
    # ------------------------------------------------------------------ #
    @staticmethod
    def _credit_total(user_info: Any) -> int | None:
        try:
            return int(user_info.credit_info.total)
        except (AttributeError, TypeError, ValueError):
            return None

    def _query_credit(self, client: Any, logs: list[str]) -> int | None:
        try:
            total = self._credit_total(client.get_user())
        except Exception as exc:  # noqa: BLE001
            logs.append(f"[netspresso] credit re-query failed: {type(exc).__name__}")
            return None
        logs.append(f"[netspresso] account credit total after: {total}")
        return total

    @staticmethod
    def _credit_record(before: int | None, after: int | None) -> CreditRecord:
        if before is not None and after is not None:
            return CreditRecord(
                usage_type=CreditUsageType.ACTUAL,
                estimated=25,
                actual=before - after,
                note=(f"actual = account credit total before ({before}) - after ({after}), observed via the SDK's "
                      f"user/credit query; estimate 25 = {CREDIT_BASIS}"),
            )
        return CreditRecord(
            usage_type=CreditUsageType.ESTIMATED,
            estimated=25,
            actual=None,
            note=f"actual deduction NOT observed (balance query unavailable); estimate 25 = {CREDIT_BASIS}",
        )

    def _failed(self, request: ExecutionRequest, env: Environment, status: ExecutionStatus, kind: str, message: str,
                logs: list[str], started: str, t0: float) -> ExecutionResult:
        env.extra.update({"api_calls_executed": self._api_calls_executed, "operations_executed": self._operations_executed})
        return ExecutionResult(
            operation=request.operation,
            configuration=request.configuration,
            execution_status=status,
            environment=env,
            execution_time_s=round(time.perf_counter() - t0, 3),
            logs=logs,
            error=ExecutionError(message=message, kind=kind),
            reproducibility=Reproducibility(level=ReproducibilityLevel.NOT_VERIFIED, runs=1 if status != ExecutionStatus.BLOCKED else 0),
            timestamp=started,
            credit=CreditRecord(usage_type=CreditUsageType.NONE, estimated=25, actual=None,
                                note="operation not completed; see ledger for observed balance") if status == ExecutionStatus.BLOCKED
            else CreditRecord(usage_type=CreditUsageType.ESTIMATED, estimated=25, actual=None, note=CREDIT_BASIS),
        )

    def _map_compression_metadata(self, request: ExecutionRequest, env: Environment, metadata: Any, input_model_path: Path,
                                  credit_before: int | None, credit_after: int | None, logs: list[str], started: str, t0: float) -> ExecutionResult:
        status_value = str(getattr(getattr(metadata, "status", None), "value", getattr(metadata, "status", "")))
        env.extra.update({"api_calls_executed": self._api_calls_executed, "operations_executed": self._operations_executed})
        credit = self._credit_record(credit_before, credit_after)
        logs.append(f"[netspresso] SDK metadata status: {status_value}")

        if status_value != "completed":
            detail = getattr(metadata, "error_detail", None)
            kind = getattr(detail, "name", None) or getattr(detail, "error_code", None) or f"SdkStatus:{status_value}"
            message = getattr(detail, "message", None) or f"SDK returned status '{status_value}'"
            data = getattr(detail, "data", None)
            error_log = getattr(data, "error_log", None)
            result = self._failed(request, env, ExecutionStatus.BLOCKED if status_value == "stopped" else ExecutionStatus.ERROR,
                                  str(kind), self._redact(str(message)), logs, started, t0)
            if error_log:
                result.error.details["error_log"] = self._redact(str(error_log))
            result.credit = credit
            return result

        compressed_path = Path(getattr(metadata, "compressed_model_path", "") or "")
        onnx_path = getattr(metadata, "compressed_onnx_model_path", None)
        results = getattr(metadata, "results", None)
        original = getattr(results, "original_model", None)
        compressed = getattr(results, "compressed_model", None)
        comp_info = getattr(metadata, "compression_info", None)

        baseline = Metrics(model_size_mb=round(input_model_path.stat().st_size / 1_000_000, 3),
                           extra=self._model_numbers("sdk_original", original))
        current = Metrics(model_size_mb=round(compressed_path.stat().st_size / 1_000_000, 3) if compressed_path.is_file() else None,
                          extra=self._model_numbers("sdk_compressed", compressed))
        artifact = artifact_from_file(
            compressed_path,
            expected_sha256=None,  # first real artifact: no reference checksum exists yet
            source_model=input_model_path.name,
            source_model_sha256=self._sha256(input_model_path),
            operation=request.operation,
            compression_method=getattr(comp_info, "method", None),
            compression_ratio=getattr(comp_info, "ratio", None),
            sdk_model_id=getattr(compressed, "model_id", None),
            sdk_version=self._sdk_version,
            sdk_metadata_path=(compressed_path.parent / "metadata.json").as_posix(),
            compressed_onnx_model_path=onnx_path,
            configuration_key=request.configuration.key,
            generated_by=f"NetsPressoAdapter {self.version} (real run)",
        )
        logs.append(f"[netspresso] compressed model: {compressed_path.as_posix()} exists={artifact.exists} sha256={artifact.checksum_sha256}")
        return ExecutionResult(
            operation=request.operation,
            configuration=request.configuration,
            execution_status=ExecutionStatus.COMPLETED,
            environment=env,
            execution_time_s=round(time.perf_counter() - t0, 3),
            baseline_metrics=baseline,
            metrics=current,
            artifact=artifact,
            logs=logs,
            reproducibility=Reproducibility(level=ReproducibilityLevel.NOT_VERIFIED, runs=1,
                                            checksums=[artifact.checksum_sha256] if artifact.checksum_sha256 else [],
                                            notes="single real run; reproducibility requires a second authorized run"),
            timestamp=started,
            credit=credit,
        )

    @staticmethod
    def _model_numbers(prefix: str, model: Any) -> dict[str, float]:
        out: dict[str, float] = {}
        for attr in ("size", "flops", "number_of_parameters", "trainable_parameters", "non_trainable_parameters", "number_of_layers"):
            value = getattr(model, attr, None)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[f"{prefix}_{attr}"] = float(value)
        return out

    @staticmethod
    def _sha256(path: Path) -> str:
        from framework.validation.artifact import compute_sha256

        return compute_sha256(path)

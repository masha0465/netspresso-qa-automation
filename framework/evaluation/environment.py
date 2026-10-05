"""Extended environment fingerprint for local measurements.

Never reads arbitrary environment variables: only an explicit allow-list of
thread-related settings is recorded, and :func:`redact_secrets` is applied to every
mapping before it is persisted.
"""

from __future__ import annotations

import os
import platform
import re
from collections.abc import Mapping
from importlib import metadata
from typing import Any

SECRET_KEY_PATTERN = re.compile(r"(key|token|secret|password|passwd|credential|auth)", re.IGNORECASE)
SECRET_VALUE_PATTERN = re.compile(r"^(eyJ[A-Za-z0-9_-]{10,}|sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})")
THREAD_ENV_ALLOWLIST = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "ORT_NUM_THREADS")
DEFAULT_PACKAGES = ("onnxruntime", "onnx", "numpy", "torch", "torchvision", "psutil", "netspresso")


def redact_secrets(data: Mapping[str, Any]) -> dict[str, Any]:
    """Drop keys that look like credentials and values that look like tokens (recursively)."""
    clean: dict[str, Any] = {}
    for key, value in data.items():
        if SECRET_KEY_PATTERN.search(str(key)):
            continue
        if isinstance(value, Mapping):
            clean[key] = redact_secrets(value)
        elif isinstance(value, str) and SECRET_VALUE_PATTERN.match(value):
            clean[key] = "<redacted>"
        else:
            clean[key] = value
    return clean


def package_versions(packages: tuple[str, ...] = DEFAULT_PACKAGES) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in packages:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def environment_fingerprint(
    *,
    execution_provider: str | None = None,
    threads: int | None = None,
    packages: tuple[str, ...] = DEFAULT_PACKAGES,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fp: dict[str, Any] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "cpu": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "execution_provider": execution_provider,
        "intra_op_threads": threads,
        "thread_env": {k: os.environ.get(k) for k in THREAD_ENV_ALLOWLIST if os.environ.get(k) is not None},
        "packages": package_versions(packages),
    }
    if extra:
        fp.update(extra)
    return redact_secrets(fp)

"""Adapter selection from configuration with explicit, safe overrides.

Defaults come from ``configs/netspresso.yaml`` (provider = mock, mode = dry_run,
confirm_credit_use = false). Every override must be passed explicitly; nothing is
inferred from the environment.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

from framework.adapters.base import BaseAdapter
from framework.adapters.mock_adapter import MockAdapter
from framework.adapters.netspresso_adapter import NetsPressoAdapter
from framework.config import ConfigurationError, FrameworkConfig, ProviderConfig

PROVIDERS = ("mock", "netspresso")
MODES = ("dry_run", "real")


def resolve_settings(
    config: FrameworkConfig,
    *,
    provider: str | None = None,
    mode: str | None = None,
    confirm_credit_use: bool | None = None,
) -> ProviderConfig:
    settings = config.provider
    changes: dict[str, object] = {}
    if provider is not None:
        if provider not in PROVIDERS:
            raise ConfigurationError(f"unknown provider '{provider}', expected one of {PROVIDERS}")
        changes["provider"] = provider
    if mode is not None:
        if mode not in MODES:
            raise ConfigurationError(f"unknown execution mode '{mode}', expected one of {MODES}")
        changes["mode"] = mode
    if confirm_credit_use is not None:
        changes["confirm_credit_use"] = bool(confirm_credit_use)
    return dataclasses.replace(settings, **changes) if changes else settings


def build_adapter(
    config: FrameworkConfig,
    *,
    provider: str | None = None,
    mode: str | None = None,
    confirm_credit_use: bool | None = None,
    env: Mapping[str, str] | None = None,
) -> BaseAdapter:
    settings = resolve_settings(config, provider=provider, mode=mode, confirm_credit_use=confirm_credit_use)
    if settings.provider == "mock":
        return MockAdapter(config)
    return NetsPressoAdapter(config, settings=settings, env=env)

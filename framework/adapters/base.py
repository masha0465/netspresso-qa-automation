"""Adapter contract.

The QA engine talks to *any* optimization backend through this interface. Two
implementations exist (or are planned):

* :class:`framework.adapters.mock_adapter.MockAdapter` – deterministic local
  simulation, 0 credits, used by pytest and CI.
* ``NetsPressoAdapter`` (Python 3.11 only) – wraps the real SDK and
  writes :class:`~framework.pipeline.result.ExecutionResult` JSON files that the
  main framework consumes offline.

Nothing here imports the NetsPresso SDK.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from framework.pipeline.result import Configuration, Environment, ExecutionResult, SupportState


@dataclass(frozen=True)
class ExecutionRequest:
    """What the pipeline asks an adapter to do for one matrix cell."""

    operation: str  # e.g. "int8_quantization" (optimization name) - vendor mapping is the adapter's job
    configuration: Configuration
    parameters: dict[str, Any] = field(default_factory=dict)
    repeat_runs: int = 1  # >1 enables reproducibility verification where the adapter supports it


class BaseAdapter(abc.ABC):
    """Abstract adapter. Implementations must be side-effect free unless documented."""

    name: str = "base"
    version: str = "0.0.0"

    #: True if calling :meth:`run` may cost money/credits or reach the network.
    #: The pipeline refuses to run such adapters unless explicitly confirmed.
    credit_consuming: bool = False

    @abc.abstractmethod
    def describe_environment(self) -> Environment:
        """Return environment facts recorded on every result (versions, platform)."""

    @abc.abstractmethod
    def check_support(self, configuration: Configuration) -> tuple[SupportState, str | None]:
        """Static support knowledge without executing anything.

        Return ``UNKNOWN`` when there is no evidence; never guess ``SUPPORTED``.
        """

    @abc.abstractmethod
    def run(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute one configuration and return raw facts (no QA verdict)."""

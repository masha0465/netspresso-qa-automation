"""Credit ledger (``reports/credit_usage.json``).

Rules
-----
* ``actual`` entries may only be written with a real account balance figure and
  an explicit confirmation flag. They are the only entries that change
  ``used_credit``/``remaining_estimate`` as *actual*.
* ``estimated`` entries (real operation executed, but balance query failed)
  also reduce ``remaining_estimate`` and are flagged as estimates.
* ``simulated`` entries (MockAdapter) never touch the real balance. They are
  accumulated separately in ``simulated_total`` so that ledger logic can be
  tested without pretending credits were spent.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from framework.pipeline.result import CreditUsageType, utc_now_iso


class CreditLedgerError(ValueError):
    """Raised when an entry would violate ledger integrity rules."""


@dataclass
class LedgerEntry:
    timestamp: str
    operation: str
    model: str
    purpose: str
    usage_type: str
    estimated_credit: int | None
    actual_credit: int | None
    remaining_estimate: int
    result: str
    error: str | None
    confirmation: bool
    adapter: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CreditLedger:
    def __init__(self, path: Path | str, starting_credit: int = 500) -> None:
        self.path = Path(path)
        self._data: dict[str, Any] = {
            "starting_credit": starting_credit,
            "used_credit": 0,
            "remaining_estimate": starting_credit,
            "simulated_total": 0,
            "last_updated": None,
            "operations": [],
        }
        if self.path.exists():
            self._load()

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        with self.path.open("r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        for key in ("starting_credit", "used_credit", "remaining_estimate", "operations"):
            if key not in loaded:
                raise CreditLedgerError(f"ledger {self.path} is missing '{key}'")
        self._data.update(loaded)
        self._data.setdefault("simulated_total", 0)

    def save(self) -> None:
        self._data["last_updated"] = utc_now_iso()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")

    # ------------------------------------------------------------------ #
    @property
    def starting_credit(self) -> int:
        return int(self._data["starting_credit"])

    @property
    def used_credit(self) -> int:
        return int(self._data["used_credit"])

    @property
    def remaining_estimate(self) -> int:
        return int(self._data["remaining_estimate"])

    @property
    def simulated_total(self) -> int:
        return int(self._data.get("simulated_total", 0))

    @property
    def operations(self) -> list[dict[str, Any]]:
        return list(self._data["operations"])

    def has_actual_usage(self) -> bool:
        return any(op.get("usage_type") in (CreditUsageType.ACTUAL.value, CreditUsageType.ESTIMATED.value)
                   for op in self._data["operations"])

    # ------------------------------------------------------------------ #
    def record(
        self,
        *,
        operation: str,
        model: str,
        purpose: str,
        usage_type: CreditUsageType | str,
        estimated_credit: int | None,
        actual_credit: int | None = None,
        result: str = "unknown",
        error: str | None = None,
        confirmation: bool = False,
        adapter: str = "unknown",
        timestamp: str | None = None,
    ) -> LedgerEntry:
        usage = CreditUsageType(usage_type)
        if usage == CreditUsageType.NONE:
            raise CreditLedgerError("usage_type 'none' cannot be recorded as an operation")

        if usage == CreditUsageType.SIMULATED:
            if actual_credit is not None:
                raise CreditLedgerError("simulated entries must not carry an actual_credit value")
            self._data["simulated_total"] = self.simulated_total + int(estimated_credit or 0)
        else:
            if not confirmation:
                raise CreditLedgerError("real (actual/estimated) entries require explicit confirmation=True")
            if usage == CreditUsageType.ACTUAL and actual_credit is None:
                raise CreditLedgerError("actual entries require an actual_credit value from the account API")
            charged = int(actual_credit if usage == CreditUsageType.ACTUAL else (estimated_credit or 0))
            self._data["used_credit"] = self.used_credit + charged
            self._data["remaining_estimate"] = self.starting_credit - self.used_credit

        entry = LedgerEntry(
            timestamp=timestamp or utc_now_iso(),
            operation=operation,
            model=model,
            purpose=purpose,
            usage_type=usage.value,
            estimated_credit=estimated_credit,
            actual_credit=actual_credit,
            remaining_estimate=self.remaining_estimate,
            result=result,
            error=error,
            confirmation=confirmation,
            adapter=adapter,
        )
        self._data["operations"].append(entry.to_dict())
        return entry

    def budget_check(self, estimates: list[int], reserve: int = 100) -> dict[str, Any]:
        """Read-only planning helper: can the planned operations run while keeping ``reserve``?

        Uses client-side estimates only; never records anything.
        """
        planned = int(sum(estimates))
        remaining_after = self.remaining_estimate - planned
        return {
            "planned_operations": len(estimates),
            "estimated_total": planned,
            "estimate_basis": "client-side SDK pre-check constants (not verified server-side)",
            "remaining_before": self.remaining_estimate,
            "remaining_after": remaining_after,
            "reserve": int(reserve),
            "affordable": remaining_after >= int(reserve),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "starting_credit": self.starting_credit,
            "used_credit": self.used_credit,
            "remaining_estimate": self.remaining_estimate,
            "simulated_total": self.simulated_total,
            "operations": len(self._data["operations"]),
            "has_actual_usage": self.has_actual_usage(),
        }

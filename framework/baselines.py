"""Baseline artifact registry.

Answers "which artifact is the reference for future reproducibility / regression
comparison, and why?".

Lifecycle (PROJECT-DEFINED POLICY, not a vendor rule):

    candidate  --promote()-->  baseline  --retire()-->  retired

* A first artifact is always registered as ``candidate``. Its checksum is *recorded*,
  not *verified*; no gate may treat it as an expected checksum.
* Promotion to ``baseline`` requires ALL of:
    1. reproducibility evidence of level BITWISE_REPRODUCIBLE or FUNCTIONALLY_REPRODUCIBLE
       (test_strategy.md Level 3 / Level 4) from a second identical-condition run,
    2. artifact integrity verified (structural validation PASS),
    3. configuration match between the runs (operation, configuration key, input shape,
       compression ratio, SDK version).
* Only ``baseline`` entries are returned by :meth:`BaselineRegistry.expected_sha256`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from framework.pipeline.result import ReproducibilityLevel, utc_now_iso

STATUSES = ("candidate", "baseline", "retired")
PROMOTABLE_LEVELS = {ReproducibilityLevel.BITWISE_REPRODUCIBLE.value, ReproducibilityLevel.FUNCTIONALLY_REPRODUCIBLE.value}


class PromotionPolicyError(ValueError):
    """Raised when a candidate does not satisfy the promotion policy."""


@dataclass
class BaselineEntry:
    artifact_id: str
    model_name: str
    source_artifact: str  # input model path or id the artifact was derived from
    sha256: str
    file_size: int
    format: str
    sdk_version: str | None
    operation: str
    configuration: dict[str, str]
    input_shape: list[int]
    compression_ratio: float | None
    environment_fingerprint: dict[str, Any]
    created_at: str
    status: str = "candidate"
    provenance: dict[str, Any] = field(default_factory=dict)
    promotion_evidence: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def configuration_key(self) -> str:
        c = self.configuration
        return "|".join((c.get("model", ""), c.get("device", ""), c.get("runtime", ""), c.get("backend", ""), c.get("optimization", "")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BaselineEntry:
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})  # type: ignore[arg-type]


def make_artifact_id(operation: str, sha256: str) -> str:
    return f"{operation}:{sha256[:12]}"


class BaselineRegistry:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._entries: list[BaselineEntry] = []
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._entries = [BaselineEntry.from_dict(e) for e in data.get("entries", [])]

    # ------------------------------------------------------------------ #
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "policy": "project-defined: candidate -> baseline requires Level 3/4 reproducibility evidence, integrity PASS, configuration match",
            "updated_at": utc_now_iso(),
            "entries": [e.to_dict() for e in self._entries],
        }
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @property
    def entries(self) -> list[BaselineEntry]:
        return list(self._entries)

    def get(self, artifact_id: str) -> BaselineEntry | None:
        return next((e for e in self._entries if e.artifact_id == artifact_id), None)

    def find(self, *, operation: str, configuration_key: str, sdk_version: str | None = None, status: str | None = None) -> list[BaselineEntry]:
        return [
            e for e in self._entries
            if e.operation == operation and e.configuration_key == configuration_key
            and (sdk_version is None or e.sdk_version == sdk_version)
            and (status is None or e.status == status)
        ]

    def expected_sha256(self, *, operation: str, configuration_key: str, sdk_version: str | None = None) -> str | None:
        """Only promoted baselines may serve as expected checksums."""
        promoted = self.find(operation=operation, configuration_key=configuration_key, sdk_version=sdk_version, status="baseline")
        return promoted[-1].sha256 if promoted else None

    # ------------------------------------------------------------------ #
    def register_candidate(self, entry: BaselineEntry) -> BaselineEntry:
        if entry.status != "candidate":
            raise PromotionPolicyError("new entries must be registered as 'candidate'")
        existing = next((e for e in self._entries if e.sha256 == entry.sha256 and e.operation == entry.operation), None)
        if existing is not None:
            return existing
        self._entries.append(entry)
        return entry

    def promote(
        self,
        artifact_id: str,
        *,
        reproducibility_level: str,
        integrity_verified: bool,
        configuration_match: bool,
        evidence: dict[str, Any] | None = None,
    ) -> BaselineEntry:
        entry = self.get(artifact_id)
        if entry is None:
            raise KeyError(artifact_id)
        if entry.status != "candidate":
            raise PromotionPolicyError(f"{artifact_id} is '{entry.status}', only candidates can be promoted")
        problems = []
        if reproducibility_level not in PROMOTABLE_LEVELS:
            problems.append(f"reproducibility level '{reproducibility_level}' is not Level 3/4 evidence")
        if not integrity_verified:
            problems.append("artifact integrity not verified")
        if not configuration_match:
            problems.append("configuration of the confirming run does not match")
        if problems:
            raise PromotionPolicyError("; ".join(problems))
        entry.status = "baseline"
        entry.promotion_evidence = {"reproducibility_level": reproducibility_level, "promoted_at": utc_now_iso(), **(evidence or {})}
        return entry

    def retire(self, artifact_id: str, reason: str) -> BaselineEntry:
        entry = self.get(artifact_id)
        if entry is None:
            raise KeyError(artifact_id)
        entry.status = "retired"
        entry.notes.append(f"retired: {reason}")
        return entry

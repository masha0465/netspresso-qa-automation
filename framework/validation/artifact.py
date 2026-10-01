"""Artifact integrity validation and helpers for real files.

For real runs the adapter fills :class:`Artifact` from disk using
:func:`compute_sha256` / :func:`verify_file_checksum`. For the mock the artifact
is virtual (``mock://``) and the checksum facts are provided directly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from framework.config import CriterionConfig
from framework.pipeline.result import Artifact, CriterionResult, CriterionStatus

CRITERION = "artifact"


def compute_sha256(path: Path | str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:  # binary mode: immune to CRLF conversion on Windows
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_checksum(path: Path | str, expected_sha256: str) -> bool:
    return compute_sha256(path).lower() == expected_sha256.lower()


def artifact_from_file(path: Path | str, expected_sha256: str | None = None, **metadata: object) -> Artifact:
    """Build an :class:`Artifact` from a real file (used by real adapters / loaders)."""
    p = Path(path)
    if not p.exists():
        return Artifact(path=p.as_posix(), exists=False, expected_checksum_sha256=expected_sha256, metadata=dict(metadata))
    checksum = compute_sha256(p)
    valid = None if expected_sha256 is None else checksum.lower() == expected_sha256.lower()
    return Artifact(
        path=p.as_posix(),
        exists=True,
        size_bytes=p.stat().st_size,
        format=p.suffix.lstrip(".") or None,
        checksum_sha256=checksum,
        expected_checksum_sha256=expected_sha256,
        checksum_valid=valid,
        metadata=dict(metadata),
    )


def validate_artifact(artifact: Artifact | None, criterion: CriterionConfig) -> CriterionResult:
    problems: list[str] = []
    if artifact is None:
        problems.append("no artifact recorded")
    else:
        if not artifact.exists:
            problems.append("artifact does not exist")
        if artifact.exists and (artifact.size_bytes is None or artifact.size_bytes <= 0):
            problems.append("artifact is empty")
        if artifact.checksum_valid is False:
            problems.append("checksum mismatch")
        if artifact.checksum_valid is None and artifact.exists:
            problems.append("checksum not verifiable (no expected checksum)")

    if not problems:
        return CriterionResult(
            name=CRITERION,
            status=CriterionStatus.PASS,
            observed="exists, non-empty, checksum valid",
            threshold="required" if criterion.required else "optional",
            message="artifact exists and checksum validated",
            required=criterion.required,
        )
    return CriterionResult(
        name=CRITERION,
        status=CriterionStatus.FAIL if criterion.required else CriterionStatus.WARN,
        observed="; ".join(problems),
        threshold="required" if criterion.required else "optional",
        message="artifact integrity failed: " + "; ".join(problems),
        required=criterion.required,
    )

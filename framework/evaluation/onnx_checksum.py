"""ONNX checksums: raw SHA-256 plus a *normalized* SHA-256 that ignores non-semantic metadata.

Observation (Phase 5-E preparation): two identical ultralytics exports of yolov8n differed
only in ``metadata_props['date']``; after removing ``metadata_props`` the serialized graph +
weights were byte-identical. A raw checksum therefore under-reports reproducibility.

Policy ``onnx-metadata-v1``:
* remove every ``metadata_props`` entry (keys are recorded),
* clear the top-level ``doc_string``,
* keep everything else (ir_version, opset_import, producer_name/version, graph, initializers).

The raw SHA-256 is always preserved alongside the normalized one; the normalized value
never replaces it.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from typing import Any

NORMALIZATION_POLICY = "onnx-metadata-v1: drop all metadata_props; clear model.doc_string; graph/initializers/opsets/producer untouched"
NORMALIZATION_VERSION = 1


def normalized_onnx_checksum(path: Path | str, *, onnx_module: Any = None) -> dict[str, Any]:
    p = Path(path)
    raw = hashlib.sha256(p.read_bytes()).hexdigest()
    onnx = onnx_module
    if onnx is None:
        try:
            onnx = importlib.import_module("onnx")
        except ImportError:
            return {"raw_sha256": raw, "normalized_sha256": None, "normalization_policy": NORMALIZATION_POLICY,
                    "normalization_version": NORMALIZATION_VERSION, "status": "NOT_APPLICABLE", "reason": "onnx not importable"}
    model = onnx.load(p.as_posix())
    removed = [kv.key for kv in model.metadata_props]
    cleared = []
    del model.metadata_props[:]
    if getattr(model, "doc_string", ""):
        cleared.append("doc_string")
        model.doc_string = ""
    normalized = hashlib.sha256(model.SerializeToString()).hexdigest()
    return {
        "raw_sha256": raw,
        "normalized_sha256": normalized,
        "normalization_policy": NORMALIZATION_POLICY,
        "normalization_version": NORMALIZATION_VERSION,
        "removed_metadata_keys": removed,
        "cleared_fields": cleared,
        "status": "COMPUTED",
    }


def checksums_match(a: dict[str, Any], b: dict[str, Any]) -> dict[str, bool | None]:
    """Compare two checksum records: raw equality and normalized equality are reported separately."""
    return {
        "raw_equal": a.get("raw_sha256") == b.get("raw_sha256"),
        "normalized_equal": (a.get("normalized_sha256") == b.get("normalized_sha256")) if a.get("normalized_sha256") and b.get("normalized_sha256") else None,
        "same_policy": a.get("normalization_policy") == b.get("normalization_policy") and a.get("normalization_version") == b.get("normalization_version"),
    }

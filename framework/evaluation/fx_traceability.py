"""FX traceability boundary + functional-equivalence bookkeeping for the YOLOv8 R1 export (pure Python).

R1 = "upstream YOLOv8n -> minimal traceability wrapper -> fx body -> head re-attach -> equivalence".
Everything that needs torch / ultralytics lives in ``scripts/yolov8_r1_traceable_export.py`` and
``scripts/yolov8_r1_fork_crosscheck.py``; this module holds the contracts and the arithmetic so
they can be unit-tested on the main interpreter without heavy dependencies.

Boundary (what the R1 fx body contains / what stays outside the graph) - see ``TRACEABILITY_BOUNDARY``.
The body ends exactly where the official Nota fork's ``export_netspresso()`` body ends: the per-level
``cat(cv2[i](x), cv3[i](x))`` feature maps ``[N, nc + 4*reg_max, H/s, W/s]``. DFL, anchors, box decode,
sigmoid and NMS are NOT part of the graph; they are rebuilt from ``netspresso_head_meta.json``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

PATCH_ID = "r1-upstream-body-wrapper-v1"
TORCH_NN_LEAF_PREFIX = "torch.nn.modules."

# Ops that must never appear inside the exported body graph (they belong to the re-attached head / post-processing).
OUT_OF_GRAPH_OP_MARKERS = ("sigmoid", "softmax", "dist2bbox", "make_anchors", "non_max_suppression", "nms", "dfl", "topk")

TRACEABILITY_BOUNDARY: dict[str, Any] = {
    "patch_id": PATCH_ID,
    "in_graph": [
        "backbone + neck layers model.0 .. model.21 (Conv/C2f/SPPF/Upsample/Concat, traced through to torch.nn leaves)",
        "Detect.cv2[i] (box branch, 4*reg_max ch) and Detect.cv3[i] (class branch, nc ch) per level, concatenated -> [N, nc+4*reg_max, H/s, W/s]",
    ],
    "out_of_graph": [
        "DFL (distribution -> distance expectation)",
        "anchor grid / stride tensors (make_anchors)",
        "box decode (dist2bbox, xywh) * stride",
        "class sigmoid",
        "NMS (non_max_suppression) and all predictor/validator post-processing",
        "pre-processing (letterbox, /255, BGR->RGB)",
    ],
    "patch_steps": [
        "wrapper module around the unmodified upstream DetectionModel: replays BaseModel._predict_once over layers 0..N-2 and emits the fork-contract maps",
        "C2f instances: bind upstream's own traceable alternative `forward_split` (same chained Bottleneck semantics; avoids iterating a Proxy from chunk()) - "
        "the same swap ultralytics' exporter applies for some export formats",
    ],
    "head_metadata_contract": ["nc", "nl", "anchors", "stride", "strides", "inplace"],
    "note": ("This boundary is identical to the official fork's export_netspresso() body/head split, so the artifact "
             "is consumable by the fork's DetectionModel_netspresso re-attach path. It is a QA-defined traceability "
             "boundary around unmodified upstream product code, not a modification of ultralytics."),
}


# --------------------------------------------------------------------------- #
# graph summary
# --------------------------------------------------------------------------- #
def summarize_graph(nodes: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize fx nodes given as ``{"op": ..., "target": str, "module_class": str|None}`` dicts.

    ``module_class`` is the fully-qualified class name of a ``call_module`` target (e.g.
    ``torch.nn.modules.conv.Conv2d``). The body is "leaf-clean" when every call_module target is a
    ``torch.nn`` module (no ultralytics container module survived tracing) and no decode/NMS marker is present.
    """
    ops: dict[str, int] = {}
    leaf_classes: dict[str, int] = {}
    non_torch_leaves: list[str] = []
    markers_found: list[str] = []
    outputs = 0
    for n in nodes:
        op, target = str(n.get("op")), str(n.get("target"))
        ops[op] = ops.get(op, 0) + 1
        if op == "call_module":
            cls = str(n.get("module_class") or "")
            leaf_classes[cls] = leaf_classes.get(cls, 0) + 1
            if not cls.startswith(TORCH_NN_LEAF_PREFIX):
                non_torch_leaves.append(f"{target}:{cls}")
        if op in ("call_function", "call_method"):
            low = target.lower()
            markers_found.extend(m for m in OUT_OF_GRAPH_OP_MARKERS if m in low)
        if op == "output":
            outputs += 1
    return {
        "node_count": sum(ops.values()),
        "ops": dict(sorted(ops.items())),
        "leaf_module_classes": dict(sorted(leaf_classes.items())),
        "non_torch_nn_leaves": non_torch_leaves,
        "out_of_graph_markers_found_in_graph": sorted(set(markers_found)),
        "leaf_clean": not non_torch_leaves and not markers_found and outputs == 1,
    }


# --------------------------------------------------------------------------- #
# numeric equivalence
# --------------------------------------------------------------------------- #
def equivalence_stats(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    """cosine / max-abs / mean-abs / relative-L2 difference between two flat sequences (a = reference)."""
    if len(a) != len(b):
        return {"status": "SHAPE_MISMATCH", "count_a": len(a), "count_b": len(b)}
    if not a:
        return {"status": "EMPTY", "count": 0}
    dot = na = nb = diff2 = max_abs = sum_abs = 0.0
    for x, y in zip(a, b, strict=True):
        x, y = float(x), float(y)
        d = x - y
        dot += x * y
        na += x * x
        nb += y * y
        diff2 += d * d
        ad = abs(d)
        sum_abs += ad
        if ad > max_abs:
            max_abs = ad
    denom = math.sqrt(na) * math.sqrt(nb)
    cosine = dot / denom if denom > 0 else (1.0 if na == nb else 0.0)
    rel = math.sqrt(diff2) / math.sqrt(na) if na > 0 else (0.0 if diff2 == 0 else float("inf"))
    return {"status": "COMPUTED", "count": len(a), "cosine": cosine, "max_abs_diff": max_abs, "mean_abs_diff": sum_abs / len(a),
            "relative_l2_diff": rel, "bitwise_identical": max_abs == 0.0}


def equivalence_verdict(stats: Mapping[str, Any], *, min_cosine: float, max_relative_l2_diff: float, max_abs_diff: float | None = None) -> dict[str, Any]:
    """Apply explicit, caller-supplied thresholds (PROJECT-DEFINED EXAMPLE) to ``equivalence_stats`` output.

    Thresholds are never baked in here: the caller must measure first and pass the values it decided on.
    """
    if stats.get("status") != "COMPUTED":
        return {"verdict": "NOT_COMPARABLE", "reason": f"stats status {stats.get('status')}"}
    failed = []
    if stats["cosine"] < min_cosine:
        failed.append(f"cosine {stats['cosine']:.8f} < {min_cosine}")
    if stats["relative_l2_diff"] > max_relative_l2_diff:
        failed.append(f"relative_l2_diff {stats['relative_l2_diff']:.3e} > {max_relative_l2_diff:.3e}")
    if max_abs_diff is not None and stats["max_abs_diff"] > max_abs_diff:
        failed.append(f"max_abs_diff {stats['max_abs_diff']:.3e} > {max_abs_diff:.3e}")
    return {"verdict": "EQUIVALENT" if not failed else "NOT_EQUIVALENT", "failed_checks": failed,
            "thresholds": {"min_cosine": min_cosine, "max_relative_l2_diff": max_relative_l2_diff, "max_abs_diff": max_abs_diff,
                           "basis": "PROJECT-DEFINED EXAMPLE THRESHOLD (set after measuring the fused/unfused noise floor)"}}


# --------------------------------------------------------------------------- #
# post-NMS detection comparison (one-image diagnostic)
# --------------------------------------------------------------------------- #
def _iou(b1: Sequence[float], b2: Sequence[float]) -> float:
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def compare_detections(ref: Sequence[Sequence[float]], cand: Sequence[Sequence[float]], *, match_iou: float = 0.5) -> dict[str, Any]:
    """Compare two post-NMS detection lists ``[x1, y1, x2, y2, conf, cls]`` (reference first).

    Greedy matching by same class and highest IoU >= ``match_iou``. Reports box counts, matched pairs,
    max coordinate / confidence differences over matched pairs, class-id agreement and unmatched boxes.
    """
    used: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for i, r in enumerate(ref):
        best, best_iou = None, 0.0
        for j, c in enumerate(cand):
            if j in used or int(round(c[5])) != int(round(r[5])):
                continue
            iou = _iou(r[:4], c[:4])
            if iou >= match_iou and iou > best_iou:
                best, best_iou = j, iou
        if best is not None:
            used.add(best)
            pairs.append((i, best, best_iou))
    coord = [abs(float(ref[i][k]) - float(cand[j][k])) for i, j, _ in pairs for k in range(4)]
    conf = [abs(float(ref[i][4]) - float(cand[j][4])) for i, j, _ in pairs]
    return {
        "ref_count": len(ref), "cand_count": len(cand), "count_equal": len(ref) == len(cand),
        "matched": len(pairs), "unmatched_ref": len(ref) - len(pairs), "unmatched_cand": len(cand) - len(pairs),
        "max_coordinate_abs_diff": max(coord) if coord else None, "max_confidence_abs_diff": max(conf) if conf else None,
        "min_matched_iou": min((p[2] for p in pairs), default=None),
        "class_ids_ref": sorted(int(round(r[5])) for r in ref), "class_ids_cand": sorted(int(round(c[5])) for c in cand),
        "all_matched": len(pairs) == len(ref) == len(cand),
    }


# --------------------------------------------------------------------------- #
# layer-level divergence
# --------------------------------------------------------------------------- #
DIVERGENCE_CATEGORIES = ("tracing boundary", "Detect head", "decode", "stride", "preprocessing", "postprocessing",
                         "tensor layout", "dtype", "export transformation", "version difference", "unknown")


def first_divergence(layer_stats: Sequence[Mapping[str, Any]], *, max_relative_l2_diff: float) -> dict[str, Any] | None:
    """Return the first ordered layer record whose relative L2 difference exceeds the tolerance (or None)."""
    for rec in layer_stats:
        st = rec.get("stats", rec)
        if st.get("status") != "COMPUTED":
            return {"layer": rec.get("layer"), "index": rec.get("index"), "reason": f"stats {st.get('status')}"}
        if st["relative_l2_diff"] > max_relative_l2_diff:
            return {"layer": rec.get("layer"), "index": rec.get("index"), "layer_type": rec.get("layer_type"), "relative_l2_diff": st["relative_l2_diff"],
                    "max_abs_diff": st["max_abs_diff"], "cosine": st["cosine"]}
    return None


def classify_divergence(first: Mapping[str, Any] | None, *, input_identical: bool, weights_identical: bool | None, layer_count: int) -> dict[str, Any]:
    """Map a first-divergence record to one of ``DIVERGENCE_CATEGORIES`` with an explicit confidence.

    The mapping is deliberately conservative: anything not pinned down by the evidence stays ``unknown``.
    """
    if first is None:
        return {"category": None, "confidence": "HIGH", "mechanism": "no divergence above tolerance", "verified": True}
    if not input_identical:
        return {"category": "preprocessing", "confidence": "HIGH", "mechanism": "inputs differ before the first layer", "verified": True}
    idx = first.get("index")
    if idx is not None and idx >= layer_count - 1:
        return {"category": "Detect head", "confidence": "MEDIUM", "mechanism": "backbone/neck agree; divergence appears at the detection head", "verified": False}
    if weights_identical is False:
        return {"category": "version difference", "confidence": "MEDIUM", "mechanism": f"parameters differ at/before layer {first.get('layer')}", "verified": False}
    if weights_identical is True:
        return {"category": "version difference", "confidence": "MEDIUM",
                "mechanism": f"identical input and parameters but layer {first.get('layer')} ({first.get('layer_type')}) computes differently -> module forward implementation differs between code versions; exact line NOT VERIFIED",
                "verified": False}
    return {"category": "unknown", "confidence": "LOW", "mechanism": f"divergence at layer {first.get('layer')}, parameter equality not checked", "verified": False}


# --------------------------------------------------------------------------- #
# export manifest
# --------------------------------------------------------------------------- #
MANIFEST_REQUIRED_KEYS = ("source_model_sha256", "source_model_version", "source_code_version", "patch_identifier", "input_shape",
                          "fx_version", "torch_version", "python_version", "timestamp", "artifact_sha256", "artifact_size_bytes")


def build_export_manifest(**fields: Any) -> dict[str, Any]:
    missing = [k for k in MANIFEST_REQUIRED_KEYS if k not in fields]
    if missing:
        raise ValueError(f"export manifest missing keys: {missing}")
    manifest = {k: fields[k] for k in MANIFEST_REQUIRED_KEYS}
    manifest.update({k: v for k, v in fields.items() if k not in manifest})
    manifest.setdefault("traceability_boundary", TRACEABILITY_BOUNDARY)
    manifest.setdefault("credit_consuming", False)
    manifest.setdefault("netspresso_api_calls", 0)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Return the list of missing/empty required keys (empty list = valid)."""
    return [k for k in MANIFEST_REQUIRED_KEYS if manifest.get(k) in (None, "", [])]

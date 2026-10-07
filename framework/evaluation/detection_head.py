"""YOLOv8 detection-head contract: head metadata, body/head schema, and the upload precheck.

Background (verified against the installed Nota fork ``ultralytics_nota`` 8.0.108):

* ``export_netspresso()`` traces the whole YOLOv8 model in **train mode** with ``torch.fx``
  and saves it as ``model_fx.pt``. In train mode the ``Detect`` head returns the raw per-scale
  feature maps, so the GraphModule's outputs are ``nl`` tensors of shape
  ``[N, nc + 4*reg_max, imgsz/stride_i, imgsz/stride_i]`` - the DFL / box decode is NOT part
  of the graph.
* ``netspresso_head_meta.json`` stores what is needed to rebuild the decode head:
  ``nc, nl, anchors, stride, strides, inplace``.
* ``DetectionModel_netspresso(graph_model_path, meta_head_json)`` re-attaches
  ``Detect_netspresso`` built from that metadata: ``nn.Sequential(torch.load(body), head)``.

This module is pure Python (no torch import) so the contract can be unit-tested on the
main interpreter; the heavy re-attach/validation itself runs in the fork environment
(``scripts/yolov8_offline_detection_eval.py``).

Precheck semantics: READY means **"known contract checks passed"** - it never means that
the NetsPresso server will accept the upload.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from framework.evaluation.artifact_structure import validate_pt

REG_MAX = 16  # YOLOv8 DFL channels (constant in the fork's Detect_netspresso)
REQUIRED_HEAD_META_KEYS = ("nc", "nl", "stride", "strides", "anchors", "inplace")

READY, BLOCKED, UNSUPPORTED, NOT_VERIFIED = "READY", "BLOCKED", "UNSUPPORTED", "NOT_VERIFIED"


class HeadMetaError(ValueError):
    """Head metadata is missing, malformed or internally inconsistent."""


@dataclass(frozen=True)
class HeadMeta:
    nc: int
    nl: int
    stride: tuple[float, ...]
    strides_shape: tuple[int, ...]  # shape of the exported `strides` tensor (per-anchor), informational
    anchors_shape: tuple[int, ...]
    inplace: bool
    source: str | None = None

    @property
    def outputs_per_anchor(self) -> int:
        return self.nc + 4 * REG_MAX

    def to_dict(self) -> dict[str, Any]:
        return {
            "nc": self.nc,
            "nl": self.nl,
            "stride": list(self.stride),
            "strides_shape": list(self.strides_shape),
            "anchors_shape": list(self.anchors_shape),
            "inplace": self.inplace,
            "outputs_per_anchor": self.outputs_per_anchor,
            "reg_max": REG_MAX,
            "source": self.source,
        }


def _shape_of(nested: Any) -> tuple[int, ...]:
    shape: list[int] = []
    cur = nested
    while isinstance(cur, list):
        shape.append(len(cur))
        cur = cur[0] if cur else None
    return tuple(shape)


def parse_head_meta(data: dict[str, Any], source: str | None = None) -> HeadMeta:
    missing = [k for k in REQUIRED_HEAD_META_KEYS if k not in data]
    if missing:
        raise HeadMetaError(f"head metadata missing keys: {missing}")
    try:
        nc, nl = int(data["nc"]), int(data["nl"])
        stride = tuple(float(s) for s in data["stride"])
    except (TypeError, ValueError) as exc:
        raise HeadMetaError(f"head metadata has non-numeric nc/nl/stride: {exc}") from exc
    if nc <= 0 or nl <= 0:
        raise HeadMetaError(f"nc and nl must be positive (nc={nc}, nl={nl})")
    if len(stride) != nl:
        raise HeadMetaError(f"stride has {len(stride)} entries but nl={nl}")
    if any(s <= 0 or s != int(s) for s in stride):
        raise HeadMetaError(f"strides must be positive integers: {stride}")
    return HeadMeta(nc=nc, nl=nl, stride=stride, strides_shape=_shape_of(data["strides"]), anchors_shape=_shape_of(data["anchors"]),
                    inplace=bool(data["inplace"]), source=source)


def load_head_meta(path: Path | str) -> HeadMeta:
    p = Path(path)
    if not p.is_file():
        raise HeadMetaError(f"head metadata file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HeadMetaError(f"head metadata is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise HeadMetaError("head metadata must be a JSON object")
    return parse_head_meta(data, source=p.as_posix())


# --------------------------------------------------------------------------- #
# shape contracts
# --------------------------------------------------------------------------- #
def validate_input_shape(shape: Sequence[int], head: HeadMeta, channels: int = 3) -> tuple[bool, str]:
    if len(shape) != 4:
        return False, f"input must be 4-D [N,C,H,W], got {list(shape)}"
    n, c, h, w = (int(x) for x in shape)
    if n < 1 or c != channels:
        return False, f"expected batch>=1 and {channels} channels, got N={n}, C={c}"
    max_stride = int(max(head.stride))
    if h % max_stride or w % max_stride:
        return False, f"H and W must be multiples of the max stride {max_stride}, got {h}x{w}"
    return True, f"input [N={n},C={c},H={h},W={w}] compatible with strides {list(head.stride)}"


def expected_body_output_shapes(input_shape: Sequence[int], head: HeadMeta) -> list[list[int]]:
    n, _c, h, w = (int(x) for x in input_shape)
    return [[n, head.outputs_per_anchor, h // int(s), w // int(s)] for s in head.stride]


def expected_decoded_output_shape(input_shape: Sequence[int], head: HeadMeta) -> list[int]:
    """Decoded head output: [N, 4 + nc, sum(H_i * W_i)] (ultralytics Detect eval/export layout)."""
    n, _c, h, w = (int(x) for x in input_shape)
    anchors = sum((h // int(s)) * (w // int(s)) for s in head.stride)
    return [n, 4 + head.nc, anchors]


def validate_output_schema(observed: Sequence[Sequence[int]], expected: Sequence[Sequence[int]]) -> tuple[bool, str]:
    obs = [list(map(int, s)) for s in observed]
    exp = [list(map(int, s)) for s in expected]
    if obs == exp:
        return True, f"output schema matches expected {exp}"
    return False, f"output schema mismatch: observed {obs}, expected {exp}"


# --------------------------------------------------------------------------- #
# precheck (TC-Y8-030 / HR-2)
# --------------------------------------------------------------------------- #
@dataclass
class PrecheckResult:
    status: str  # READY | BLOCKED | UNSUPPORTED | NOT_VERIFIED
    message: str
    checks: dict[str, bool | None] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    meaning: str = "READY = known contract checks passed; it does NOT assert that the NetsPresso server will accept the upload"

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "message": self.message, "checks": dict(self.checks), "details": dict(self.details), "meaning": self.meaning}


def precheck_fx_bundle(
    fx_path: Path | str,
    head_meta_path: Path | str,
    *,
    input_shape: Sequence[int] = (1, 3, 640, 640),
    expected_nc: int | None = None,
    expected_task: str = "detect",
    allow_full_unpickle: bool = False,
    torch_module: Any = None,
    forward: Callable[[Sequence[int]], Sequence[Sequence[int]]] | None = None,
) -> PrecheckResult:
    """Check the fx body + head metadata against the compressor contract observed in Phase 1/5-B.

    ``forward`` (optional) runs the body on a zero tensor of ``input_shape`` and returns the
    output shapes; it is injected so the check is testable without torch.
    """
    checks: dict[str, bool | None] = {}
    details: dict[str, Any] = {"task": expected_task, "input_shape": list(input_shape)}
    fx, hm = Path(fx_path), Path(head_meta_path)

    checks["fx_exists"] = fx.is_file()
    checks["fx_extension_pt"] = fx.suffix.lower() in (".pt", ".pth")
    checks["head_meta_exists"] = hm.is_file()
    if not (checks["fx_exists"] and checks["head_meta_exists"]):
        return PrecheckResult(BLOCKED, "fx body and/or head metadata file missing", checks, details)
    if not checks["fx_extension_pt"]:
        return PrecheckResult(UNSUPPORTED, f"compressor contract expects a torch.save'd .pt GraphModule, got '{fx.suffix}' (the 2025 failure mode)", checks, details)

    try:
        head = load_head_meta(hm)
        checks["head_meta_valid"] = True
        details["head_meta"] = head.to_dict()
    except HeadMetaError as exc:
        checks["head_meta_valid"] = False
        return PrecheckResult(BLOCKED, f"head metadata invalid: {exc}", checks, details)

    checks["nc_matches_expected"] = None if expected_nc is None else head.nc == expected_nc
    if checks["nc_matches_expected"] is False:
        return PrecheckResult(BLOCKED, f"head nc={head.nc} does not match expected {expected_nc}", checks, details)

    ok, msg = validate_input_shape(input_shape, head)
    checks["input_shape_compatible"] = ok
    details["input_shape_message"] = msg
    if not ok:
        return PrecheckResult(BLOCKED, msg, checks, details)

    sv = validate_pt(fx, allow_full_unpickle=allow_full_unpickle, torch_module=torch_module)
    details["structural_validation"] = sv.to_dict()
    if sv.status == "NOT_APPLICABLE":
        checks["fx_is_graph_module"] = None
        return PrecheckResult(NOT_VERIFIED, f"fx body not loaded ({sv.message}); GraphModule check not performed", checks, details)
    checks["fx_is_graph_module"] = sv.status == "PASS" and sv.details.get("object_kind") == "graph_module"
    if not checks["fx_is_graph_module"]:
        kind = sv.details.get("object_kind", "unknown")
        return PrecheckResult(UNSUPPORTED if sv.status == "PASS" else BLOCKED,
                              f"fx body is not a torch.fx GraphModule (kind={kind}, status={sv.status}); compressor contract not met", checks, details)

    expected = expected_body_output_shapes(input_shape, head)
    details["expected_body_output_shapes"] = expected
    if forward is None:
        checks["output_schema"] = None
        return PrecheckResult(READY, "known contract checks passed (forward/output schema not executed)", checks, details)
    try:
        observed = [list(s) for s in forward(input_shape)]
    except Exception as exc:  # noqa: BLE001 - any forward failure blocks the upload
        checks["output_schema"] = False
        details["forward_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return PrecheckResult(BLOCKED, f"body forward failed: {details['forward_error']}", checks, details)
    ok, msg = validate_output_schema(observed, expected)
    checks["output_schema"] = ok
    details["observed_body_output_shapes"] = observed
    details["output_schema_message"] = msg
    if not ok:
        return PrecheckResult(BLOCKED, msg, checks, details)
    return PrecheckResult(READY, "known contract checks passed (format, GraphModule, head metadata, input shape, output schema)", checks, details)

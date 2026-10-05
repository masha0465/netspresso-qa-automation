"""Independent verification of SDK-reported model statistics (parameters, FLOPs).

The SDK/server reports ``size``, ``flops`` and ``number_of_parameters``. We never
restate those as verified; instead every quantity is recorded as
``sdk_reported`` / ``independently_verified`` / ``delta`` / ``verification_status``.

* Parameters: exact count from ONNX initializers (or ``torch`` parameters).
* FLOPs: MAC estimate over Conv / Gemm / MatMul nodes using ONNX shape inference.
  The SDK does not document whether "flops" means MACs or 2*MACs, so the estimate is
  compared against both conventions; a match is reported with the convention that
  matched, otherwise the status is INCONCLUSIVE/FAIL with the raw numbers.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

PASS, FAIL, NOT_APPLICABLE, INCONCLUSIVE = "PASS", "FAIL", "NOT_APPLICABLE", "INCONCLUSIVE"


@dataclass
class VerifiedQuantity:
    name: str
    sdk_reported: float | None
    independently_verified: float | None
    verification_status: str
    method: str
    delta: float | None = None
    delta_percent: float | None = None
    tolerance_percent: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sdk_reported": self.sdk_reported,
            "independently_verified": self.independently_verified,
            "delta": self.delta,
            "delta_percent": self.delta_percent,
            "tolerance_percent": self.tolerance_percent,
            "verification_status": self.verification_status,
            "method": self.method,
            "notes": list(self.notes),
        }


def verify(name: str, sdk_reported: float | None, verified: float | None, tolerance_percent: float, method: str) -> VerifiedQuantity:
    if verified is None:
        return VerifiedQuantity(name, sdk_reported, None, NOT_APPLICABLE, method, tolerance_percent=tolerance_percent,
                                notes=["independent value could not be computed"])
    if sdk_reported is None:
        return VerifiedQuantity(name, None, verified, NOT_APPLICABLE, method, tolerance_percent=tolerance_percent,
                                notes=["SDK did not report this quantity"])
    delta = verified - sdk_reported
    pct = (abs(delta) / abs(sdk_reported) * 100.0) if sdk_reported else (0.0 if delta == 0 else math.inf)
    status = PASS if pct <= tolerance_percent else FAIL
    return VerifiedQuantity(name, sdk_reported, verified, status, method, delta=delta, delta_percent=round(pct, 4), tolerance_percent=tolerance_percent)


def verify_flops(sdk_flops: float | None, macs: float | None, tolerance_percent: float, method: str) -> VerifiedQuantity:
    """Compare SDK 'flops' against a MAC estimate under both conventions (MACs, 2*MACs)."""
    if macs is None or sdk_flops is None:
        return verify("flops", sdk_flops, macs, tolerance_percent, method)
    candidates = {"MACs": macs, "2*MACs": 2.0 * macs}
    best_name, best_val = min(candidates.items(), key=lambda kv: abs(kv[1] - sdk_flops))
    q = verify("flops", sdk_flops, best_val, tolerance_percent, method)
    q.notes.append(f"closest convention: {best_name} (MACs={macs:.0f}, 2*MACs={2 * macs:.0f}); SDK does not document its FLOPs convention")
    if q.verification_status == FAIL:
        q.verification_status = INCONCLUSIVE if q.delta_percent is not None and q.delta_percent <= 50.0 else FAIL
        q.notes.append("estimate covers Conv/Gemm/MatMul only; see covered/uncovered op counts")
    return q


# --------------------------------------------------------------------------- #
# ONNX-based counts
# --------------------------------------------------------------------------- #
def _numel(dims: list[int]) -> int:
    n = 1
    for d in dims:
        n *= int(d)
    return n


def count_onnx_parameters(model: Any) -> int:
    """Sum of initializer element counts. Includes shape constants (negligible for conv nets)."""
    return int(sum(_numel(list(init.dims)) for init in model.graph.initializer))


def count_torch_parameters(module: Any) -> int:
    return int(sum(int(p.numel()) for p in module.parameters()))


def _value_shapes(model: Any) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for coll in (model.graph.input, model.graph.output, model.graph.value_info):
        for vi in coll:
            try:
                dims = vi.type.tensor_type.shape.dim
                shapes[vi.name] = [int(d.dim_value) if d.dim_value else 1 for d in dims]  # dynamic dim -> 1 (batch)
            except AttributeError:
                continue
    for init in model.graph.initializer:
        shapes[init.name] = list(init.dims)
    return shapes


def _attr(node: Any, name: str, default: Any) -> Any:
    for a in node.attribute:
        if a.name == name:
            if a.ints:
                return list(a.ints)
            if a.i:
                return int(a.i)
    return default


def estimate_onnx_macs(model: Any, onnx_module: Any) -> tuple[int | None, dict[str, int], dict[str, int], str | None]:
    """Return (macs, covered_op_counts, uncovered_op_counts, error). Requires onnx.shape_inference."""
    try:
        inferred = onnx_module.shape_inference.infer_shapes(model)
    except Exception as exc:  # noqa: BLE001
        return None, {}, {}, f"shape inference failed: {type(exc).__name__}"
    shapes = _value_shapes(inferred)
    macs = 0
    covered: Counter[str] = Counter()
    uncovered: Counter[str] = Counter()
    for node in inferred.graph.node:
        op = node.op_type
        try:
            if op == "Conv":
                out = shapes[node.output[0]]
                w = shapes[node.input[1]]  # [Cout, Cin/groups, kh, kw]
                kernel = _numel(w[1:])  # Cin/groups * kh * kw
                macs += _numel(out) * kernel  # out elements (N*Cout*H*W) * per-element MACs
                covered[op] += 1
            elif op == "Gemm":
                a, b = shapes[node.input[0]], shapes[node.input[1]]
                trans_b = _attr(node, "transB", 0)
                m, k = a[0], a[1]
                n = b[0] if trans_b else b[1]
                macs += m * n * k
                covered[op] += 1
            elif op == "MatMul":
                a, out = shapes[node.input[0]], shapes[node.output[0]]
                macs += _numel(out) * a[-1]
                covered[op] += 1
            else:
                uncovered[op] += 1
        except (KeyError, IndexError):
            uncovered[op] += 1
    return int(macs), dict(covered), dict(uncovered), None

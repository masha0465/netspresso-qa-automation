"""Artifact structural validation (PyTorch ``.pt`` and ONNX ``.onnx``).

Five distinct statements are kept apart (see docs/test_strategy.md §4.2):
exists -> readable -> structurally valid -> checksum recorded -> reproducibility verified.
This module covers the first three.

Security policy
---------------
``torch.load`` executes pickled code unless ``weights_only=True``. We always try the
safe path first. If the artifact needs full unpickling (e.g. a ``torch.fx.GraphModule``),
it is only loaded when ``allow_full_unpickle=True``, which callers must grant explicitly
(the local-eval script does so only for SHA-256 values in a trust list).
Untrusted artifacts therefore yield ``NOT_APPLICABLE`` for the load checks, never a
silent full load.
"""

from __future__ import annotations

import importlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from framework.pipeline.result import DefectCategory

PASS, FAIL, NOT_APPLICABLE = "PASS", "FAIL", "NOT_APPLICABLE"
ZIP_MAGIC = b"PK\x03\x04"  # torch >= 1.6 zip-based serialization


@dataclass
class StructuralValidation:
    path: str
    format: str
    status: str  # PASS | FAIL | NOT_APPLICABLE
    message: str
    checks: dict[str, bool | None] = field(default_factory=dict)  # None = not evaluated
    details: dict[str, Any] = field(default_factory=dict)
    defect_category: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "format": self.format,
            "status": self.status,
            "message": self.message,
            "checks": dict(self.checks),
            "details": dict(self.details),
            "defect_category": self.defect_category,
        }


def _import(name: str, injected: Any) -> Any | None:
    if injected is not None:
        return injected
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _basic_file_checks(path: Path) -> tuple[dict[str, bool | None], dict[str, Any], str | None]:
    checks: dict[str, bool | None] = {"exists": path.is_file()}
    details: dict[str, Any] = {}
    if not checks["exists"]:
        return checks, details, "artifact does not exist"
    size = path.stat().st_size
    details["size_bytes"] = size
    checks["non_empty"] = size > 0
    if not checks["non_empty"]:
        return checks, details, "artifact is empty"
    try:
        with path.open("rb") as fh:
            head = fh.read(8)
        checks["readable"] = True
        details["magic"] = head[:4].hex()
    except OSError as exc:
        checks["readable"] = False
        return checks, details, f"artifact not readable: {exc}"
    return checks, details, None


# --------------------------------------------------------------------------- #
# PyTorch
# --------------------------------------------------------------------------- #
def validate_pt(path: Path | str, *, allow_full_unpickle: bool = False, torch_module: Any = None) -> StructuralValidation:
    p = Path(path)
    checks, details, problem = _basic_file_checks(p)
    if problem:
        return StructuralValidation(p.as_posix(), "pt", FAIL, problem, checks, details, DefectCategory.ARTIFACT_ERROR.value)
    checks["zip_container"] = details.get("magic") == ZIP_MAGIC.hex()

    torch = _import("torch", torch_module)
    if torch is None:
        checks["safe_load"] = None
        return StructuralValidation(p.as_posix(), "pt", NOT_APPLICABLE,
                                    "torch not available in this interpreter; load checks not evaluated", checks, details)

    # 1) safe path
    try:
        obj = torch.load(p.as_posix(), map_location="cpu", weights_only=True)
        checks["safe_load"] = True
        details["load_mode"] = "weights_only"
    except Exception as exc:  # noqa: BLE001 - UnpicklingError and torch-specific errors
        checks["safe_load"] = False
        details["safe_load_error"] = type(exc).__name__
        if not allow_full_unpickle:
            checks["full_load"] = None
            return StructuralValidation(
                p.as_posix(), "pt", NOT_APPLICABLE,
                "artifact requires full unpickling (weights_only=True failed); not loaded because the artifact is not in the trust list",
                checks, details,
            )
        try:
            obj = torch.load(p.as_posix(), map_location="cpu", weights_only=False)
            checks["full_load"] = True
            details["load_mode"] = "full_unpickle (trusted)"
        except Exception as exc2:  # noqa: BLE001
            checks["full_load"] = False
            details["full_load_error"] = f"{type(exc2).__name__}: {str(exc2)[:200]}"
            return StructuralValidation(p.as_posix(), "pt", FAIL, "artifact could not be unpickled (corrupt or incompatible)",
                                        checks, details, DefectCategory.ARTIFACT_ERROR.value)

    # 2) classify the loaded object
    kind, numbers = _describe_torch_object(obj, torch)
    details.update(numbers)
    details["object_kind"] = kind
    checks["recognised_structure"] = kind != "unknown"
    checks["has_parameters"] = numbers.get("parameter_count", 0) > 0 or numbers.get("tensor_count", 0) > 0
    if not checks["recognised_structure"] or not checks["has_parameters"]:
        return StructuralValidation(p.as_posix(), "pt", FAIL, f"loaded object is not a recognised model structure ({kind})",
                                    checks, details, DefectCategory.ARTIFACT_ERROR.value)
    return StructuralValidation(p.as_posix(), "pt", PASS, f"loaded as {kind} via {details['load_mode']}", checks, details)


def _describe_torch_object(obj: Any, torch: Any) -> tuple[str, dict[str, Any]]:
    nn_module = getattr(getattr(torch, "nn", None), "Module", None)
    tensor_t = getattr(torch, "Tensor", None)
    if nn_module is not None and isinstance(obj, nn_module):
        params = [p for p in obj.parameters()] if hasattr(obj, "parameters") else []
        kind = "graph_module" if type(obj).__name__ == "GraphModule" else "nn_module"
        return kind, {
            "class_name": type(obj).__name__,
            "parameter_count": int(sum(int(p.numel()) for p in params)),
            "parameter_tensors": len(params),
        }
    if isinstance(obj, dict):
        tensors = [v for v in obj.values() if tensor_t is not None and isinstance(v, tensor_t)]
        if tensors and len(tensors) == len(obj):
            return "state_dict", {"tensor_count": len(tensors), "parameter_count": int(sum(int(t.numel()) for t in tensors))}
        return "dict", {"keys": len(obj), "tensor_count": len(tensors)}
    if tensor_t is not None and isinstance(obj, tensor_t):
        return "tensor", {"tensor_count": 1, "parameter_count": int(obj.numel())}
    return "unknown", {"class_name": type(obj).__name__}


# --------------------------------------------------------------------------- #
# ONNX
# --------------------------------------------------------------------------- #
def validate_onnx(path: Path | str, *, onnx_module: Any = None) -> StructuralValidation:
    p = Path(path)
    checks, details, problem = _basic_file_checks(p)
    if problem:
        return StructuralValidation(p.as_posix(), "onnx", FAIL, problem, checks, details, DefectCategory.ARTIFACT_ERROR.value)

    onnx = _import("onnx", onnx_module)
    if onnx is None:
        checks["onnx_load"] = None
        return StructuralValidation(p.as_posix(), "onnx", NOT_APPLICABLE, "onnx not available in this interpreter; graph checks not evaluated", checks, details)

    try:
        model = onnx.load(p.as_posix())
        checks["onnx_load"] = True
    except Exception as exc:  # noqa: BLE001
        checks["onnx_load"] = False
        details["load_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return StructuralValidation(p.as_posix(), "onnx", FAIL, "onnx.load failed (corrupt or not an ONNX file)", checks, details, DefectCategory.ARTIFACT_ERROR.value)

    try:
        onnx.checker.check_model(model)
        checks["checker"] = True
    except Exception as exc:  # noqa: BLE001 - onnx.checker.ValidationError
        checks["checker"] = False
        details["checker_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"

    graph = getattr(model, "graph", None)
    checks["graph_present"] = graph is not None
    inputs = list(getattr(graph, "input", [])) if graph is not None else []
    outputs = list(getattr(graph, "output", [])) if graph is not None else []
    nodes = list(getattr(graph, "node", [])) if graph is not None else []
    inits = list(getattr(graph, "initializer", [])) if graph is not None else []
    checks["has_inputs"] = len(inputs) > 0
    checks["has_outputs"] = len(outputs) > 0
    checks["has_nodes"] = len(nodes) > 0
    checks["has_initializers"] = len(inits) > 0
    details.update({
        "ir_version": getattr(model, "ir_version", None),
        "opset_imports": [getattr(o, "version", None) for o in getattr(model, "opset_import", [])],
        "inputs": [{"name": i.name, "shape": _shape_of(i)} for i in inputs],
        "outputs": [{"name": o.name, "shape": _shape_of(o)} for o in outputs],
        "node_count": len(nodes),
        "initializer_count": len(inits),
        "op_type_histogram": dict(Counter(getattr(n, "op_type", "?") for n in nodes)),
    })
    failed = [k for k in ("checker", "graph_present", "has_inputs", "has_outputs", "has_nodes", "has_initializers") if checks.get(k) is False]
    if failed:
        return StructuralValidation(p.as_posix(), "onnx", FAIL, "ONNX structural checks failed: " + ", ".join(failed),
                                    checks, details, DefectCategory.ARTIFACT_ERROR.value)
    return StructuralValidation(p.as_posix(), "onnx", PASS, f"valid ONNX graph ({len(nodes)} nodes, {len(inits)} initializers)", checks, details)


def _shape_of(value_info: Any) -> list[int | str | None]:
    try:
        dims = value_info.type.tensor_type.shape.dim
        return [d.dim_value if getattr(d, "dim_value", 0) else (getattr(d, "dim_param", None) or None) for d in dims]
    except AttributeError:
        return []


def validate_artifact_structure(path: Path | str, **kwargs: Any) -> StructuralValidation:
    suffix = Path(path).suffix.lower()
    if suffix in (".pt", ".pth"):
        return validate_pt(path, allow_full_unpickle=kwargs.get("allow_full_unpickle", False), torch_module=kwargs.get("torch_module"))
    if suffix == ".onnx":
        return validate_onnx(path, onnx_module=kwargs.get("onnx_module"))
    return StructuralValidation(Path(path).as_posix(), suffix.lstrip(".") or "unknown", NOT_APPLICABLE,
                                f"no structural validator for '{suffix}' artifacts", {"exists": Path(path).is_file()})

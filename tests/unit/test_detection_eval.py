"""Phase 5-E Conditional GO resolution tests (TC-Y8-CG-001 .. 015).

Pure-Python checks of the detection-head contract, true-accuracy comparison, ONNX
checksum normalization, precheck semantics and zero-credit guards. No torch / onnx /
ultralytics / netspresso is imported; heavy paths are exercised with fakes.
"""

from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

import pytest

from framework.config import CriterionConfig
from framework.evaluation.detection_accuracy import (
    MEASURED,
    NA,
    NOT_COMPARABLE,
    PRIMARY_METRIC,
    DetectionMetrics,
    compare_detection_accuracy,
)
from framework.evaluation.detection_head import (
    BLOCKED,
    NOT_VERIFIED,
    READY,
    REG_MAX,
    UNSUPPORTED,
    HeadMeta,
    HeadMetaError,
    expected_body_output_shapes,
    expected_decoded_output_shape,
    load_head_meta,
    parse_head_meta,
    precheck_fx_bundle,
    validate_input_shape,
    validate_output_schema,
)
from framework.evaluation.local_ort import PROXY_DISCLAIMER, compare_outputs
from framework.evaluation.onnx_checksum import NORMALIZATION_POLICY, checksums_match, normalized_onnx_checksum
from framework.pipeline.result import CriterionStatus
from framework.validation.accuracy import validate_accuracy
from framework.validation.equivalence import validate_output_equivalence_proxy

pytestmark = pytest.mark.unit

HEAD_META = {"nc": 80, "nl": 3, "anchors": [[0.0] * 5, [0.0] * 5], "stride": [8.0, 16.0, 32.0], "strides": [[1.0] * 5], "inplace": True}


def _write_meta(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "netspresso_head_meta.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ---------------------------------------------------------------- TC-Y8-CG-001 head metadata load ----
def test_cg001_head_metadata_load(tmp_path):
    head = load_head_meta(_write_meta(tmp_path, HEAD_META))
    assert isinstance(head, HeadMeta) and head.nc == 80 and head.nl == 3 and head.stride == (8.0, 16.0, 32.0)
    assert head.outputs_per_anchor == 80 + 4 * REG_MAX == 144
    assert head.to_dict()["reg_max"] == 16 and head.source.endswith("netspresso_head_meta.json")
    with pytest.raises(HeadMetaError, match="missing keys"):
        parse_head_meta({"nc": 80})
    with pytest.raises(HeadMetaError, match="stride has"):
        parse_head_meta({**HEAD_META, "stride": [8.0, 16.0]})
    with pytest.raises(HeadMetaError, match="positive"):
        parse_head_meta({**HEAD_META, "nc": 0})
    with pytest.raises(HeadMetaError, match="not found"):
        load_head_meta(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(HeadMetaError, match="valid JSON"):
        load_head_meta(bad)


# ---------------------------------------------------------------- TC-Y8-CG-003 input shape ----
def test_cg003_input_shape_validation():
    head = parse_head_meta(HEAD_META)
    assert validate_input_shape([1, 3, 640, 640], head)[0]
    assert not validate_input_shape([1, 3, 630, 640], head)[0]  # not a multiple of max stride 32
    assert not validate_input_shape([1, 1, 640, 640], head)[0]
    assert not validate_input_shape([3, 640, 640], head)[0]


# ---------------------------------------------------------------- TC-Y8-CG-004 output schema ----
def test_cg004_output_schema_validation():
    head = parse_head_meta(HEAD_META)
    expected = expected_body_output_shapes([1, 3, 640, 640], head)
    assert expected == [[1, 144, 80, 80], [1, 144, 40, 40], [1, 144, 20, 20]]  # matches the real fx precheck observation
    assert expected_decoded_output_shape([1, 3, 640, 640], head) == [1, 84, 8400]  # matches ultralytics ONNX output0
    assert validate_output_schema(expected, expected)[0]
    ok, msg = validate_output_schema([[1, 144, 80, 80], [1, 144, 40, 40]], expected)
    assert not ok and "mismatch" in msg


# ---------------------------------------------------------------- TC-Y8-CG-002 / 014 re-attach contract + precheck semantics ----
class _Tensor:
    def __init__(self, n):
        self._n = n

    def numel(self):
        return self._n


class _Module:
    def __init__(self, sizes):
        self._p = [_Tensor(s) for s in sizes]

    def parameters(self):
        return list(self._p)


class GraphModule(_Module):
    pass


def _fake_torch(full_obj, safe_error=True):
    class UnpicklingError(Exception):
        pass

    def load(path, map_location=None, weights_only=True):
        if weights_only and safe_error:
            raise UnpicklingError("weights only failed")
        return full_obj

    return types.SimpleNamespace(load=load, nn=types.SimpleNamespace(Module=_Module), Tensor=_Tensor)


def _fx_file(tmp_path: Path, name="model_fx.pt") -> Path:
    f = tmp_path / name
    f.write_bytes(b"PK\x03\x04" + b"x" * 64)
    return f


def test_cg002_cg014_precheck_ready_blocked_unsupported_not_verified(tmp_path):
    meta = _write_meta(tmp_path, HEAD_META)
    fx = _fx_file(tmp_path)
    good_forward = lambda shape: expected_body_output_shapes(shape, parse_head_meta(HEAD_META))  # noqa: E731

    ready = precheck_fx_bundle(fx, meta, expected_nc=80, allow_full_unpickle=True, torch_module=_fake_torch(GraphModule([100])), forward=good_forward)
    assert ready.status == READY and ready.checks["fx_is_graph_module"] and ready.checks["output_schema"]
    assert "does NOT assert" in ready.meaning  # READY != server acceptance

    # output schema mismatch -> BLOCKED
    blocked = precheck_fx_bundle(fx, meta, allow_full_unpickle=True, torch_module=_fake_torch(GraphModule([1])), forward=lambda s: [[1, 84, 8400]])
    assert blocked.status == BLOCKED and blocked.checks["output_schema"] is False

    # ONNX handed to the compressor contract (the 2025 failure mode) -> UNSUPPORTED before any load
    onnx_like = tmp_path / "yolov8l.onnx"
    onnx_like.write_bytes(b"\x08\x07" * 10)
    unsupported = precheck_fx_bundle(onnx_like, meta, torch_module=_fake_torch(GraphModule([1])))
    assert unsupported.status == UNSUPPORTED and "2025" in unsupported.message

    # trusted-load policy respected: untrusted body is never fully unpickled -> NOT_VERIFIED
    nv = precheck_fx_bundle(fx, meta, allow_full_unpickle=False, torch_module=_fake_torch(GraphModule([1])))
    assert nv.status == NOT_VERIFIED and nv.checks["fx_is_graph_module"] is None

    # a plain nn.Module (not fx) is not what the compressor expects
    not_fx = precheck_fx_bundle(fx, meta, allow_full_unpickle=True, torch_module=_fake_torch(_Module([1])))
    assert not_fx.status == UNSUPPORTED and "GraphModule" in not_fx.message

    # bad metadata / nc mismatch / missing file
    assert precheck_fx_bundle(fx, _write_meta(tmp_path, {**HEAD_META, "nl": 2}), torch_module=_fake_torch(GraphModule([1]))).status == BLOCKED
    assert precheck_fx_bundle(fx, meta, expected_nc=20, torch_module=_fake_torch(GraphModule([1]))).status == BLOCKED
    assert precheck_fx_bundle(tmp_path / "nope.pt", meta).status == BLOCKED
    assert precheck_fx_bundle(fx, meta, input_shape=[1, 3, 630, 640], allow_full_unpickle=True, torch_module=_fake_torch(GraphModule([1]))).status == BLOCKED
    assert precheck_fx_bundle(fx, meta, allow_full_unpickle=True, torch_module=_fake_torch(GraphModule([1])), forward=lambda s: (_ for _ in ()).throw(RuntimeError("boom"))).status == BLOCKED


# ---------------------------------------------------------------- TC-Y8-CG-006/007/008 accuracy metrics, delta, N/A ----
def _measured(map_, label, **kw):
    base = dict(status=MEASURED, map50_95=map_, map50=map_ + 0.15, map75=map_ + 0.02, precision=0.6, recall=0.5, dataset="coco128.yaml", split="val",
                imgsz=640, conf=0.001, iou=0.7, evaluator="ultralytics_nota 8.0.108 DetectionValidator", model_label=label)
    base.update(kw)
    return DetectionMetrics(**base)


def test_cg006_cg007_map_structure_and_delta():
    b, c = _measured(0.44369, "baseline"), _measured(0.42000, "candidate")
    assert b.primary == 0.44369 and PRIMARY_METRIC == "mAP50-95"
    fm = b.to_framework_metrics()
    assert fm.accuracy == 0.44369 and fm.extra["detection_mAP50"] == pytest.approx(0.59369)
    cmp = compare_detection_accuracy(b, c)
    assert cmp.status == MEASURED and cmp.baseline_metric == 0.44369 and cmp.candidate_metric == 0.42
    assert cmp.absolute_delta == pytest.approx(-0.02369, abs=1e-6) and cmp.drop_pp == pytest.approx(2.369, abs=1e-3)
    assert cmp.relative_delta_percent == pytest.approx(-5.3393, abs=1e-3)
    assert cmp.secondary["mAP50"]["drop_pp"] == pytest.approx(2.369, abs=1e-3)
    assert cmp.to_dict()["is_output_equivalence_proxy"] is False
    identity = compare_detection_accuracy(b, _measured(0.44369, "identity fixture"))
    assert identity.status == MEASURED and identity.drop_pp == 0.0 and identity.absolute_delta == 0.0


def test_cg008_accuracy_na_semantics_never_become_pass_or_fail():
    b = _measured(0.44369, "baseline")
    na = DetectionMetrics.not_available("labeled evaluation dataset unavailable", dataset=None, model_label="candidate")
    cmp = compare_detection_accuracy(b, na)
    assert cmp.status == NA and "not measured" in cmp.reason and cmp.drop_pp is None
    assert compare_detection_accuracy(None, b).status == NA
    # the gate criterion stays NOT_APPLICABLE (never PASS, never a regression FAIL) when the candidate metric is missing
    crit = validate_accuracy(b.to_framework_metrics(), None, CriterionConfig("accuracy", 1.0, True))
    assert crit.status == CriterionStatus.NOT_APPLICABLE
    # different evaluation conditions -> NOT_COMPARABLE, no delta
    other = _measured(0.30, "candidate", dataset="coco.yaml")
    cmp2 = compare_detection_accuracy(b, other)
    assert cmp2.status == NOT_COMPARABLE and cmp2.drop_pp is None and "dataset" in cmp2.reason


# ---------------------------------------------------------------- TC-Y8-CG-009 identity fixture through the gate criterion ----
def test_cg009_identity_fixture_passes_accuracy_criterion_without_being_called_compressed():
    b = _measured(0.44369, "yolov8n baseline (same evaluator)")
    fixture = _measured(0.44369, "identity fixture (pre-compression baseline fx body)")
    crit = validate_accuracy(b.to_framework_metrics(), fixture.to_framework_metrics(), CriterionConfig("accuracy", 1.0, True))
    assert crit.status == CriterionStatus.PASS and crit.observed == 0.0
    assert "compressed" not in fixture.model_label.lower().replace("pre-compression", "")


# ---------------------------------------------------------------- TC-Y8-CG-010 proxy vs accuracy separation ----
def test_cg010_output_equivalence_proxy_is_not_accuracy():
    proxy = compare_outputs([[1.0, 0.0]], [[1.0, 0.0]], baseline_shapes=[[2]], optimized_shapes=[[2]], baseline_dtypes=["float32"], optimized_dtypes=["float32"], min_cosine=0.99)
    assert proxy.status == "PASS" and proxy.note == PROXY_DISCLAIMER
    # a perfect proxy does not create an accuracy value: the accuracy criterion is still NOT_APPLICABLE
    from framework.pipeline.result import Metrics

    m = Metrics(extra={"proxy_min_cosine_similarity": 1.0, "proxy_shape_match": 1.0, "proxy_dtype_match": 1.0})
    assert validate_output_equivalence_proxy(m, CriterionConfig("output_equivalence_proxy", None, True, {"min_cosine_similarity": 0.99})).status == CriterionStatus.PASS
    assert validate_accuracy(Metrics(accuracy=0.44), m, CriterionConfig("accuracy", 1.0, True)).status == CriterionStatus.NOT_APPLICABLE
    assert DetectionMetrics.not_available("x").to_dict()["is_output_equivalence_proxy"] is False


# ---------------------------------------------------------------- TC-Y8-CG-011/012 normalized + raw checksum ----
class _KV:
    def __init__(self, key, value):
        self.key, self.value = key, value


class _FakeOnnxModel:
    def __init__(self, props, doc=""):
        self.metadata_props = [_KV(k, v) for k, v in props]
        self.doc_string = doc
        self.graph_bytes = b"GRAPH+WEIGHTS"

    def SerializeToString(self):  # noqa: N802 - mirrors protobuf API
        return self.graph_bytes + b"|" + b";".join(f"{kv.key}={kv.value}".encode() for kv in self.metadata_props) + b"|" + self.doc_string.encode()


def _fake_onnx(model):
    return types.SimpleNamespace(load=lambda p: model)


def test_cg011_cg012_normalized_checksum_ignores_metadata_but_preserves_raw(tmp_path):
    f1, f2 = tmp_path / "a.onnx", tmp_path / "b.onnx"
    f1.write_bytes(b"file-one"), f2.write_bytes(b"file-two")  # raw bytes differ
    r1 = normalized_onnx_checksum(f1, onnx_module=_fake_onnx(_FakeOnnxModel([("date", "2026-10-05T11:20"), ("author", "x")], doc="d")))
    r2 = normalized_onnx_checksum(f2, onnx_module=_fake_onnx(_FakeOnnxModel([("date", "2026-10-05T11:25"), ("author", "x")], doc="")))
    assert r1["raw_sha256"] != r2["raw_sha256"]  # TC-012: raw preserved and distinct
    assert r1["normalized_sha256"] == r2["normalized_sha256"]  # TC-011: metadata-insensitive
    assert r1["removed_metadata_keys"] == ["date", "author"] and r1["cleared_fields"] == ["doc_string"] and r2["cleared_fields"] == []
    assert r1["normalization_policy"] == NORMALIZATION_POLICY and r1["normalization_version"] == 1 and r1["status"] == "COMPUTED"
    assert set(r1) >= {"raw_sha256", "normalized_sha256", "normalization_policy", "normalization_version"}
    m = checksums_match(r1, r2)
    assert m == {"raw_equal": False, "normalized_equal": True, "same_policy": True}
    # a graph change must change the normalized checksum too
    changed = _FakeOnnxModel([("date", "x")])
    changed.graph_bytes = b"DIFFERENT"
    r3 = normalized_onnx_checksum(f1, onnx_module=_fake_onnx(changed))
    assert r3["normalized_sha256"] != r1["normalized_sha256"]
    # no onnx available -> NOT_APPLICABLE but raw still recorded
    import framework.evaluation.onnx_checksum as oc

    orig = oc.importlib.import_module
    oc.importlib.import_module = lambda name: (_ for _ in ()).throw(ImportError(name))  # type: ignore[assignment]
    try:
        r4 = normalized_onnx_checksum(f1)
    finally:
        oc.importlib.import_module = orig  # type: ignore[assignment]
    assert r4["status"] == "NOT_APPLICABLE" and r4["raw_sha256"] == r1["raw_sha256"] and r4["normalized_sha256"] is None


# ---------------------------------------------------------------- TC-Y8-CG-013 reproducibility metadata ----
def test_cg013_reproducibility_metadata_recorded_in_baseline_report(root):
    runs = sorted(p for p in (root / "reports" / "yolov8_baseline").glob("2*Z") if (p / "baseline.json").is_file())
    assert runs, "baseline report expected (produced by scripts/yolov8_baseline_prep.py)"
    b = json.loads((runs[-1] / "baseline.json").read_text(encoding="utf-8"))
    assert b["credit_consuming"] is False and b["netspresso_api_calls"] == 0
    w = b["model"]["weights"]
    assert re.fullmatch(r"[0-9a-f]{64}", w["sha256"]) and w["trusted_for_full_unpickle"] is True
    onnx = b["artifacts"]["onnx_export"]
    assert re.fullmatch(r"[0-9a-f]{64}", onnx["sha256"]) and onnx["opset"] == 13
    if "checksums" in onnx:  # present once the baseline was regenerated with the normalized-checksum helper
        assert onnx["checksums"]["raw_sha256"] == onnx["sha256"] and re.fullmatch(r"[0-9a-f]{64}", onnx["checksums"]["normalized_sha256"])
    env = b["environment"]
    for key in ("os", "cpu", "python", "execution_provider", "intra_op_threads", "packages"):
        assert key in env
    assert {"torch", "onnxruntime", "onnx", "ultralytics"} <= set(env["packages"])
    assert b["input_shape"] == [1, 3, 640, 640] and b["seed"] == 0 and "preprocessing" in b
    assert b["accuracy"]["status"] in ("MEASURED", "N/A")
    assert "NETSPRESSO" not in json.dumps(b) and not re.search(r"[A-Za-z]:[\\\\/]Users|/Users/|/home/", json.dumps(b))


# ---------------------------------------------------------------- TC-Y8-CG-015 zero-credit guard ----
def test_cg015_zero_credit_execution_guard(root):
    for script in ("yolov8_baseline_prep.py", "yolov8_fx_export_precheck.py", "yolov8_offline_detection_eval.py"):
        src = (root / "scripts" / script).read_text(encoding="utf-8")
        assert not re.search(r"^\s*(import|from)\s+netspresso\b", src, re.M), script
        assert "netspresso_adapter" not in src and "NetsPresso(" not in src and "automatic_compression(" not in src, script
        assert "os.environ" not in src and "getenv" not in src, script
        assert "ledger_before" in src and "No NetsPresso API call was made" in src, script
    import framework.evaluation.detection_accuracy  # noqa: F401,E401

    assert not any(m == "netspresso" or m.startswith("netspresso.") for m in sys.modules)
    ledger = json.loads((root / "reports" / "credit_usage.json").read_text(encoding="utf-8"))
    assert (ledger["used_credit"], ledger["remaining_estimate"], len(ledger["operations"])) == (50, 450, 2)  # after E5-1 (one more real operation, 25 credits observed)

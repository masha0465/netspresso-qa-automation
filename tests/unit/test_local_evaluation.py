"""Phase 5-D: gate profiles, baseline registry, structural validation, independent stats,
local ORT evaluation arithmetic, environment fingerprint, zero-credit guarantees.

No torch / onnx / onnxruntime / numpy / psutil are required: fakes are injected.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from framework.adapters.base import ExecutionRequest
from framework.adapters.mock_adapter import MockAdapter
from framework.baselines import BaselineEntry, BaselineRegistry, PromotionPolicyError, make_artifact_id
from framework.config import ConfigurationError, CriterionConfig, load_local_eval, load_quality_gate
from framework.defects import classify
from framework.evaluation.artifact_structure import validate_artifact_structure, validate_onnx, validate_pt
from framework.evaluation.environment import environment_fingerprint, redact_secrets
from framework.evaluation.local_ort import (
    PROXY_DISCLAIMER,
    LocalOrtEvaluator,
    compare_output_pair,
    compare_outputs,
    compute_latency_stats,
    memory_stats,
    percentile,
)
from framework.evaluation.model_stats import count_onnx_parameters, estimate_onnx_macs, verify, verify_flops
from framework.pipeline.result import (
    Artifact,
    Configuration,
    CriterionStatus,
    DefectCategory,
    Metrics,
    Status,
)
from framework.quality_gate import QualityGate, render_text
from framework.validation.equivalence import validate_output_equivalence_proxy
from framework.validation.structural import validate_structural_validity

pytestmark = pytest.mark.unit

CFG = Configuration("mobilenet_v2", "intel_xeon_w2233", "onnxruntime", "default", "int8_quantization")


def _completed_execution(adapter):
    return adapter.run(ExecutionRequest(operation=CFG.optimization, configuration=CFG, repeat_runs=2))


# ------------------------------------------------------------------ gate profiles ----
def test_gate_profiles_load_and_default_is_release(config, root):
    qg = config.quality_gate
    assert set(qg.profiles) >= {"compression", "local_eval", "release"}
    assert qg.profile(None) is qg.criteria and qg.profile("release") is qg.profiles["release"]
    assert set(qg.criteria) <= set(qg.profiles["release"])  # release profile is a superset of the legacy default gate
    assert qg.profiles["compression"]["artifact"].params == {"checksum_required": False}
    assert qg.profiles["local_eval"]["output_equivalence_proxy"].params["min_cosine_similarity"] == 0.99
    assert qg.profiles["compression"]["model_size"].threshold == 0.0 and qg.profiles["compression"]["model_size"].required
    with pytest.raises(ConfigurationError, match="unknown quality gate profile"):
        qg.profile("staging")
    # a file without `profiles` still yields a release profile (backward compatible)
    minimal = root / "configs"
    assert "release" in load_quality_gate(minimal / "quality_gate.yaml").profiles


def test_profile_specific_required_metrics(config, adapter):
    ex = _completed_execution(adapter)  # mock: metrics, artifact with valid checksum, bitwise repro; no structural data
    release = QualityGate(config.quality_gate, "release").evaluate(ex)
    compression = QualityGate(config.quality_gate, "compression").evaluate(ex)
    local = QualityGate(config.quality_gate, "local_eval").evaluate(ex)
    assert release.profile == "release" and compression.profile == "compression" and local.profile == "local_eval"
    # structural validity is required in every profile and is missing for mock results -> never PASS on missing data
    assert all(not g.passed for g in (release, compression, local))
    assert any("Structural Validity" in r for r in compression.reasons)
    assert any("Output Equiv. Proxy" in r for r in local.reasons)
    assert "[compression]" in render_text(compression)
    assert QualityGate(config.quality_gate).profile_name == "release"  # default unchanged


def test_profile_passes_when_its_evidence_is_present(config, adapter):
    ex = _completed_execution(adapter)
    ex.artifact.metadata["structural_validation"] = {"status": "PASS", "message": "valid ONNX graph"}
    ex.metrics.extra.update({"proxy_min_cosine_similarity": 0.9995, "proxy_shape_match": 1.0, "proxy_dtype_match": 1.0})
    assert QualityGate(config.quality_gate, "compression").evaluate(ex).passed
    assert QualityGate(config.quality_gate, "local_eval").evaluate(ex).passed
    # release still needs accuracy -> mock has it, so release passes too for this fully-evidenced mock case
    assert QualityGate(config.quality_gate, "release").evaluate(ex).passed


def test_checksum_optional_only_where_configured():
    unverifiable = Artifact(path="a", exists=True, size_bytes=10, checksum_valid=None)
    from framework.validation.artifact import validate_artifact

    strict = validate_artifact(unverifiable, CriterionConfig("artifact", None, True))
    relaxed = validate_artifact(unverifiable, CriterionConfig("artifact", None, True, {"checksum_required": False}))
    assert strict.status == CriterionStatus.NOT_APPLICABLE and relaxed.status == CriterionStatus.PASS
    assert "no promoted baseline" in relaxed.message


def test_structural_and_proxy_criteria_semantics():
    req = CriterionConfig("structural_validity", None, True)
    assert validate_structural_validity(None, req).status == CriterionStatus.NOT_APPLICABLE
    assert validate_structural_validity(Artifact(path="a", exists=True, metadata={"structural_validation": {"status": "FAIL", "message": "corrupt"}}), req).status == CriterionStatus.FAIL
    assert validate_structural_validity(Artifact(path="a", exists=True, metadata={"structural_validation": {"status": "PASS", "message": "ok"}}), req).status == CriterionStatus.PASS

    prox = CriterionConfig("output_equivalence_proxy", None, True, {"min_cosine_similarity": 0.99})
    assert validate_output_equivalence_proxy(Metrics(), prox).status == CriterionStatus.NOT_APPLICABLE
    good = validate_output_equivalence_proxy(Metrics(extra={"proxy_min_cosine_similarity": 0.995}), prox)
    bad = validate_output_equivalence_proxy(Metrics(extra={"proxy_min_cosine_similarity": 0.5, "proxy_shape_match": 1.0}), prox)
    assert good.status == CriterionStatus.PASS and "not accuracy" in good.message
    assert bad.status == CriterionStatus.FAIL and "proxy" in bad.message
    # accuracy stays NOT_APPLICABLE even when the proxy exists: never conflated
    from framework.validation.accuracy import validate_accuracy

    assert validate_accuracy(Metrics(), Metrics(extra={"proxy_min_cosine_similarity": 0.999}), CriterionConfig("accuracy", 1.0, True)).status == CriterionStatus.NOT_APPLICABLE


def test_proxy_failure_classifies_with_proxy_wording(config, adapter):
    ex = _completed_execution(adapter)
    ex.configuration = Configuration("m", "intel_xeon_w2233", "onnxruntime", "default", "automatic_compression")
    ex.artifact.metadata["structural_validation"] = {"status": "PASS", "message": "ok"}
    ex.metrics.extra.update({"proxy_min_cosine_similarity": 0.42, "proxy_shape_match": 1.0, "proxy_dtype_match": 1.0})
    gate = QualityGate(config.quality_gate, "local_eval").evaluate(ex)
    d = classify(ex, gate)
    assert d.category == DefectCategory.ACCURACY_REGRESSION and "proxy" in d.message and "fine-tuning" in d.suspected_cause


# ------------------------------------------------------------------ baseline registry ----
def _entry(sha="a" * 64, status="candidate", op="automatic_compression"):
    return BaselineEntry(
        artifact_id=make_artifact_id(op, sha), model_name="graphmodule_sample", source_artifact="graphmodule.pt", sha256=sha,
        file_size=7456333, format="pt", sdk_version="1.17.0", operation=op,
        configuration={"model": "graphmodule_sample", "device": "intel_xeon_w2233", "runtime": "onnxruntime", "backend": "default", "optimization": op},
        input_shape=[1, 3, 224, 224], compression_ratio=0.5, environment_fingerprint={"python": "3.11"}, created_at="2026-10-05T01:04:10+00:00",
        status=status, provenance={"run_dir": "reports/real_runs/x"},
    )


def test_registry_schema_candidate_and_promotion_policy(tmp_path):
    reg = BaselineRegistry(tmp_path / "registry.json")
    e = reg.register_candidate(_entry())
    assert e.status == "candidate" and e.artifact_id == "automatic_compression:aaaaaaaaaaaa"
    assert reg.register_candidate(_entry()) is e  # dedupe by sha + operation
    assert reg.expected_sha256(operation=e.operation, configuration_key=e.configuration_key) is None  # candidates never serve as expected
    with pytest.raises(PromotionPolicyError, match="not Level 3/4"):
        reg.promote(e.artifact_id, reproducibility_level="NOT_VERIFIED", integrity_verified=True, configuration_match=True)
    with pytest.raises(PromotionPolicyError, match="integrity"):
        reg.promote(e.artifact_id, reproducibility_level="BITWISE_REPRODUCIBLE", integrity_verified=False, configuration_match=True)
    reg.promote(e.artifact_id, reproducibility_level="FUNCTIONALLY_REPRODUCIBLE", integrity_verified=True, configuration_match=True, evidence={"second_run": "r2"})
    assert e.status == "baseline" and e.promotion_evidence["second_run"] == "r2"
    assert reg.expected_sha256(operation=e.operation, configuration_key=e.configuration_key, sdk_version="1.17.0") == "a" * 64
    with pytest.raises(PromotionPolicyError):
        reg.promote(e.artifact_id, reproducibility_level="BITWISE_REPRODUCIBLE", integrity_verified=True, configuration_match=True)
    reg.save()
    reloaded = BaselineRegistry(tmp_path / "registry.json")
    data = json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))
    assert data["schema_version"] == 1 and "project-defined" in data["policy"]
    assert reloaded.get(e.artifact_id).status == "baseline"
    for key in ("artifact_id", "model_name", "source_artifact", "sha256", "file_size", "format", "sdk_version", "operation", "configuration",
                "input_shape", "compression_ratio", "environment_fingerprint", "created_at", "status", "provenance"):
        assert key in data["entries"][0]
    reg.retire(e.artifact_id, "superseded")
    assert reg.expected_sha256(operation=e.operation, configuration_key=e.configuration_key) is None
    with pytest.raises(PromotionPolicyError):
        reg.register_candidate(_entry(sha="b" * 64, status="baseline"))


# ------------------------------------------------------------------ structural validation ----
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


class GraphModule(_Module):  # name matters: the validator recognises torch.fx GraphModule by class name
    pass


def _fake_torch(safe_obj=None, full_obj=None, safe_error=True):
    class UnpicklingError(Exception):
        pass

    def load(path, map_location=None, weights_only=True):
        if weights_only:
            if safe_error:
                raise UnpicklingError("Weights only load failed")
            return safe_obj
        if isinstance(full_obj, Exception):
            raise full_obj
        return full_obj

    nn = types.SimpleNamespace(Module=_Module)
    return types.SimpleNamespace(load=load, nn=nn, Tensor=_Tensor)


def test_pt_validation_safe_path_and_trust_gate(tmp_path):
    f = tmp_path / "m.pt"
    f.write_bytes(b"PK\x03\x04" + b"x" * 100)
    # untrusted GraphModule -> NOT_APPLICABLE, never fully unpickled
    untrusted = validate_pt(f, allow_full_unpickle=False, torch_module=_fake_torch(full_obj=GraphModule([10, 20])))
    assert untrusted.status == "NOT_APPLICABLE" and untrusted.checks["safe_load"] is False and untrusted.checks["full_load"] is None
    assert "trust list" in untrusted.message
    # trusted -> full load, recognised GraphModule
    trusted = validate_pt(f, allow_full_unpickle=True, torch_module=_fake_torch(full_obj=GraphModule([10, 20])))
    assert trusted.status == "PASS" and trusted.details["object_kind"] == "graph_module" and trusted.details["parameter_count"] == 30
    assert trusted.details["load_mode"].startswith("full_unpickle")
    # safe state_dict path
    sd = validate_pt(f, torch_module=_fake_torch(safe_obj={"w": _Tensor(5), "b": _Tensor(1)}, safe_error=False))
    assert sd.status == "PASS" and sd.details["object_kind"] == "state_dict" and sd.details["parameter_count"] == 6
    # corrupt
    corrupt = validate_pt(f, allow_full_unpickle=True, torch_module=_fake_torch(full_obj=RuntimeError("bad zip")))
    assert corrupt.status == "FAIL" and corrupt.defect_category == "ARTIFACT_ERROR"
    # empty / missing / no torch
    empty = tmp_path / "e.pt"
    empty.write_bytes(b"")
    assert validate_pt(empty, torch_module=_fake_torch()).status == "FAIL"
    assert validate_pt(tmp_path / "nope.pt", torch_module=_fake_torch()).status == "FAIL"
    assert validate_pt(f, torch_module=None).status in ("NOT_APPLICABLE", "FAIL")  # no torch in 3.14 venv -> NOT_APPLICABLE


class _Dim:
    def __init__(self, v):
        self.dim_value = v
        self.dim_param = "" if v else "N"


def _vi(name, dims):
    return types.SimpleNamespace(name=name, type=types.SimpleNamespace(tensor_type=types.SimpleNamespace(shape=types.SimpleNamespace(dim=[_Dim(d) for d in dims]))))


def _init(name, dims):
    return types.SimpleNamespace(name=name, dims=list(dims))


def _node(op, inputs, outputs, **attrs):
    return types.SimpleNamespace(op_type=op, input=list(inputs), output=list(outputs),
                                 attribute=[types.SimpleNamespace(name=k, ints=v if isinstance(v, list) else [], i=v if isinstance(v, int) else 0) for k, v in attrs.items()])


def _fake_onnx_model():
    # x[1,3,8,8] -Conv(w[4,3,3,3])-> y[1,4,8,8] -Relu-> z -Gemm(W[10,256])-> out[1,10]
    graph = types.SimpleNamespace(
        input=[_vi("x", [0, 3, 8, 8])],
        output=[_vi("out", [1, 10])],
        value_info=[_vi("y", [1, 4, 8, 8]), _vi("z", [1, 256])],
        node=[_node("Conv", ["x", "w"], ["y"]), _node("Relu", ["y"], ["z0"]), _node("Reshape", ["z0"], ["z"]), _node("Gemm", ["z", "W"], ["out"], transB=1)],
        initializer=[_init("w", [4, 3, 3, 3]), _init("W", [10, 256])],
    )
    return types.SimpleNamespace(graph=graph, ir_version=7, opset_import=[types.SimpleNamespace(version=13)])


def _fake_onnx(model=None, checker_fail=False, load_fail=False):
    def load(path):
        if load_fail:
            raise ValueError("not a protobuf")
        return model or _fake_onnx_model()

    def check_model(m):
        if checker_fail:
            raise ValueError("ValidationError: broken")

    return types.SimpleNamespace(load=load, checker=types.SimpleNamespace(check_model=check_model),
                                 shape_inference=types.SimpleNamespace(infer_shapes=lambda m: m))


def test_onnx_validation(tmp_path):
    f = tmp_path / "m.onnx"
    f.write_bytes(b"\x08\x07" * 50)
    ok = validate_onnx(f, onnx_module=_fake_onnx())
    assert ok.status == "PASS" and ok.details["node_count"] == 4 and ok.details["op_type_histogram"]["Conv"] == 1
    assert ok.details["inputs"][0]["shape"] == ["N", 3, 8, 8]
    bad = validate_onnx(f, onnx_module=_fake_onnx(checker_fail=True))
    assert bad.status == "FAIL" and bad.checks["checker"] is False and bad.defect_category == "ARTIFACT_ERROR"
    assert validate_onnx(f, onnx_module=_fake_onnx(load_fail=True)).status == "FAIL"
    assert validate_onnx(f, onnx_module=None).status in ("NOT_APPLICABLE", "FAIL")
    assert validate_artifact_structure(tmp_path / "x.tflite").status == "NOT_APPLICABLE"


# ------------------------------------------------------------------ model stats ----
def test_independent_parameter_and_flops_verification():
    model = _fake_onnx_model()
    params = count_onnx_parameters(model)
    assert params == 4 * 3 * 3 * 3 + 10 * 256 == 2668
    macs, covered, uncovered, err = estimate_onnx_macs(model, _fake_onnx())
    assert err is None and covered == {"Conv": 1, "Gemm": 1} and uncovered == {"Relu": 1, "Reshape": 1}
    assert macs == 1 * 4 * 8 * 8 * (3 * 3 * 3) + 1 * 10 * 256  # 6912 + 2560

    ok = verify("params", 2668, params, 0.5, "onnx")
    assert ok.verification_status == "PASS" and ok.delta == 0
    off = verify("params", 2000, params, 0.5, "onnx")
    assert off.verification_status == "FAIL" and off.delta_percent > 30
    na = verify("params", 2668, None, 0.5, "onnx")
    assert na.verification_status == "NOT_APPLICABLE" and na.independently_verified is None
    assert verify("params", None, 5, 0.5, "onnx").verification_status == "NOT_APPLICABLE"

    f_macs = verify_flops(macs, macs, 10.0, "est")
    assert f_macs.verification_status == "PASS" and "MACs" in f_macs.notes[0]
    f_2macs = verify_flops(2 * macs, macs, 10.0, "est")
    assert f_2macs.verification_status == "PASS" and "2*MACs" in f_2macs.notes[0]
    assert verify_flops(macs * 1.3, macs, 10.0, "est").verification_status == "INCONCLUSIVE"
    assert verify_flops(macs * 5, macs, 10.0, "est").verification_status == "FAIL"
    assert verify_flops(100.0, None, 10.0, "est").verification_status == "NOT_APPLICABLE"  # unsupported -> never claimed equal
    d = verify_flops(100.0, None, 10.0, "est").to_dict()
    assert set(d) >= {"sdk_reported", "independently_verified", "delta", "verification_status"}


# ------------------------------------------------------------------ local ORT arithmetic ----
def test_latency_statistics():
    samples = [5.0, 1.0, 3.0, 2.0, 4.0]
    st = compute_latency_stats(samples, warmup=2)
    assert (st.median_ms, st.min_ms, st.max_ms, st.mean_ms, st.iterations, st.warmup) == (3.0, 1.0, 5.0, 3.0, 5, 2)
    assert st.p95_ms == 5.0 and percentile(sorted(samples), 50) == 3.0
    even = compute_latency_stats([1.0, 2.0, 3.0, 4.0], warmup=0)
    assert even.median_ms == 2.5 and even.source == "local_onnxruntime_cpu"
    hundred = compute_latency_stats([float(i) for i in range(1, 101)], warmup=10)
    assert hundred.p95_ms == 95.0
    with pytest.raises(ValueError):
        compute_latency_stats([], 0)


def test_memory_statistics():
    mb = 1024 * 1024
    m = memory_stats(100 * mb, 150 * mb, 180 * mb)
    assert m.status == "PASS" and m.delta_mb == 80.0 and m.rss_peak_mb == 180.0
    assert memory_stats(None, None, None).status == "NOT_APPLICABLE"


def test_output_equivalence_proxy_math():
    same = compare_output_pair([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert same["cosine_similarity"] == 1.0 and same["max_abs_diff"] == 0.0
    scaled = compare_output_pair([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
    assert scaled["cosine_similarity"] == 1.0 and scaled["relative_diff"] == 1.0
    orth = compare_output_pair([1.0, 0.0], [0.0, 1.0])
    assert orth["cosine_similarity"] == 0.0
    p = compare_outputs([[1.0, 2.0], [0.5]], [[1.0, 2.1], [0.5]], baseline_shapes=[[2], [1]], optimized_shapes=[[2], [1]],
                        baseline_dtypes=["float32", "float32"], optimized_dtypes=["float32", "float32"], min_cosine=0.99)
    assert p.status == "PASS" and p.shape_match and p.dtype_match and len(p.per_output) == 2 and p.note == PROXY_DISCLAIMER
    bad = compare_outputs([[1.0, 0.0]], [[0.0, 1.0]], baseline_shapes=[[2]], optimized_shapes=[[2]], baseline_dtypes=["float32"], optimized_dtypes=["float32"], min_cosine=0.99)
    assert bad.status == "FAIL" and bad.min_cosine_similarity == 0.0
    shape = compare_outputs([[1.0, 0.0]], [[1.0]], baseline_shapes=[[2]], optimized_shapes=[[1]], baseline_dtypes=["float32"], optimized_dtypes=["float32"], min_cosine=0.99)
    assert shape.status == "FAIL" and not shape.shape_match
    assert compare_outputs([], [], baseline_shapes=[], optimized_shapes=[], baseline_dtypes=[], optimized_dtypes=[], min_cosine=0.99).status == "NOT_APPLICABLE"
    assert "accuracy" not in json.dumps(p.to_dict()).lower().replace("not a labeled accuracy", "")  # the only 'accuracy' word is the disclaimer


class _FakeSession:
    def __init__(self, outputs, rss_log):
        self._outputs = outputs
        self.calls = 0
        self._rss_log = rss_log

    def get_inputs(self):
        return [types.SimpleNamespace(name="images", shape=["N", 3, 4, 4], type="tensor(float)")]

    def get_outputs(self):
        return [types.SimpleNamespace(name=f"out{i}") for i in range(len(self._outputs))]

    def run(self, _names, feed):
        assert "images" in feed
        self.calls += 1
        return [types.SimpleNamespace(ravel=lambda o=o: types.SimpleNamespace(tolist=lambda: list(o)), shape=(len(o),), dtype="float32") for o in self._outputs]


def test_local_ort_evaluator_with_fakes(tmp_path):
    sessions = []
    ticks = iter(range(0, 10_000))
    clock = lambda: next(ticks) * 0.001  # 1 ms per tick -> every run measures 1 ms  # noqa: E731
    rss = iter([100, 120, 130, 130, 130, 125])
    probe = lambda: next(rss, 125) * 1024 * 1024  # noqa: E731

    def factory(path, threads, provider):
        s = _FakeSession(outputs=[[1.0, 2.0, 3.0]], rss_log=[])
        sessions.append((path, threads, provider, s))
        return s

    ev = LocalOrtEvaluator(warmup=3, iterations=7, seed=0, threads=2, session_factory=factory,
                           input_factory=lambda shape, seed: [[0.0] * 4] * 4, memory_probe=probe, clock=clock)
    assert ev.credit_consuming is False
    run = ev.measure(tmp_path / "m.onnx")
    assert sessions[0][1:3] == (2, "CPUExecutionProvider") and sessions[0][3].calls == 10  # warmup 3 + 7 measured
    assert run.input_shape == [1, 3, 4, 4] and run.input_dtype == "float32" and run.output_names == ["out0"]
    assert run.latency.iterations == 7 and run.latency.warmup == 3 and run.latency.median_ms == pytest.approx(1.0)
    assert run.memory.status == "PASS" and run.memory.rss_before_mb == 100.0 and run.memory.rss_peak_mb == 130.0 and run.memory.delta_mb == 30.0
    assert run.output_shapes == [[3]] and run.output_dtypes == ["float32"]
    d = run.to_dict(include_outputs=True)
    assert d["outputs"] == [[1.0, 2.0, 3.0]] and d["latency"]["source"] == "local_onnxruntime_cpu"
    with pytest.raises(ValueError):
        LocalOrtEvaluator(iterations=0)


# ------------------------------------------------------------------ environment ----
def test_environment_fingerprint_filters_secrets(monkeypatch):
    monkeypatch.setenv("NETSPRESSO_API_KEY", "FAKE-KEY-SHOULD-NEVER-APPEAR")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    fp = environment_fingerprint(execution_provider="CPUExecutionProvider", threads=4, extra={"api_key": "x", "note": "ok", "nested": {"token": "t", "keep": 1}})
    text = json.dumps(fp)
    assert "FAKE-KEY-SHOULD-NEVER-APPEAR" not in text and "NETSPRESSO_API_KEY" not in text
    assert "api_key" not in fp and fp["note"] == "ok" and fp["nested"] == {"keep": 1}
    assert fp["thread_env"] == {"OMP_NUM_THREADS": "4"} and fp["execution_provider"] == "CPUExecutionProvider"
    assert set(fp) >= {"os", "architecture", "cpu_count", "python", "packages"}
    assert redact_secrets({"Authorization": "Bearer x", "value": "eyJabcdefghijklmnop.sig", "plain": 1}) == {"value": "<redacted>", "plain": 1}


# ------------------------------------------------------------------ zero credit ----
def test_local_eval_config_and_zero_credit_guarantees(config, root):
    le = config.local_eval
    assert (le.warmup_iterations, le.measurement_iterations, le.seed, le.proxy_min_cosine_similarity) == (10, 100, 0, 0.99)
    assert le.is_trusted("eca62020aeb07bdeccda2aceffe6fee3ba50b62374e3c656c81f171968db1762") and not le.is_trusted("deadbeef") and not le.is_trusted(None)
    assert load_local_eval(root / "nonexistent.yaml") == type(le)()
    # the evaluation package never imports the SDK
    import framework.evaluation.artifact_structure  # noqa: F401
    import framework.evaluation.local_ort  # noqa: F401
    import framework.evaluation.model_stats  # noqa: F401
    assert not any(m == "netspresso" or m.startswith("netspresso.") for m in sys.modules)
    src = Path(root / "scripts" / "run_local_eval.py").read_text(encoding="utf-8")
    assert "os.environ" not in src and "getenv" not in src and "netspresso_adapter" not in src and "credit_consuming" in src
    assert "No NetsPresso API call was made" in src


def test_mock_adapter_and_runner_unaffected_by_profiles(config):
    from framework.pipeline.runner import PipelineRunner

    report = PipelineRunner(config, MockAdapter(config), strategy="pairwise").run()
    assert report.status_counts()["PASS"] == 9  # same verdicts as before Phase 5-D (release profile semantics unchanged)
    assert all(c.gate.profile == "release" for c in report.cases if c.gate)
    data = json.dumps(report.to_dict())
    assert '"profile": "release"' in data and Status.PASS.value in data


def test_pt_validation_recognises_training_checkpoint_dicts(tmp_path):
    """ultralytics-style checkpoints wrap the module under 'model'; they must not be reported as unknown dicts."""
    f = tmp_path / "ckpt.pt"
    f.write_bytes(b"PK\x03\x04" + b"x" * 64)
    ckpt = {"epoch": -1, "model": _Module([100, 24]), "optimizer": None, "train_args": {"imgsz": 640}}
    sv = validate_pt(f, allow_full_unpickle=True, torch_module=_fake_torch(full_obj=ckpt))
    assert sv.status == "PASS" and sv.details["object_kind"] == "checkpoint"
    assert sv.details["checkpoint_key"] == "model" and sv.details["parameter_count"] == 124
    plain = validate_pt(f, allow_full_unpickle=True, torch_module=_fake_torch(full_obj={"a": 1, "b": "x"}))
    assert plain.status == "FAIL" and plain.details["object_kind"] == "dict"

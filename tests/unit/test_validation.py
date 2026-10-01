import pytest

from framework.config import CriterionConfig
from framework.pipeline.result import (
    Artifact,
    CriterionStatus,
    Metrics,
    Reproducibility,
    ReproducibilityLevel,
)
from framework.validation.accuracy import validate_accuracy
from framework.validation.artifact import (
    artifact_from_file,
    compute_sha256,
    validate_artifact,
    verify_file_checksum,
)
from framework.validation.performance import validate_latency, validate_memory, validate_model_size
from framework.validation.reproducibility import assess_reproducibility, validate_reproducibility

pytestmark = pytest.mark.unit

REQ = CriterionConfig("x", 1.0, True)
OPT = CriterionConfig("x", 10.0, False)


def test_accuracy_drop_semantics():
    base, cur = Metrics(accuracy=0.520), Metrics(accuracy=0.512)
    r = validate_accuracy(base, cur, CriterionConfig("accuracy", 1.0, True))
    assert r.status == CriterionStatus.PASS and r.observed == pytest.approx(0.8)
    r = validate_accuracy(base, Metrics(accuracy=0.505), CriterionConfig("accuracy", 1.0, True))
    assert r.status == CriterionStatus.FAIL and "exceeds" in r.message
    r = validate_accuracy(base, Metrics(accuracy=0.530), CriterionConfig("accuracy", 1.0, True))
    assert r.status == CriterionStatus.PASS  # improvement passes
    assert validate_accuracy(None, cur, REQ).status == CriterionStatus.NOT_APPLICABLE
    assert validate_accuracy(base, Metrics(accuracy=0.4), CriterionConfig("accuracy", 1.0, False)).status == CriterionStatus.WARN


def test_performance_increase_semantics():
    base = Metrics(latency_ms=10.0, memory_mb=100.0, model_size_mb=10.0)
    ok = Metrics(latency_ms=10.5, memory_mb=90.0, model_size_mb=10.5)
    assert validate_latency(base, ok, CriterionConfig("latency", 10.0, True)).status == CriterionStatus.PASS
    assert validate_memory(base, ok, CriterionConfig("memory", 15.0, True)).status == CriterionStatus.PASS
    bad = Metrics(latency_ms=12.0, memory_mb=120.0, model_size_mb=12.0)
    lat = validate_latency(base, bad, CriterionConfig("latency", 10.0, True))
    assert lat.status == CriterionStatus.FAIL and lat.observed == pytest.approx(20.0)
    assert validate_memory(base, bad, CriterionConfig("memory", 15.0, True)).status == CriterionStatus.FAIL
    assert validate_model_size(base, bad, CriterionConfig("model_size", 10.0, False)).status == CriterionStatus.WARN
    zero = Metrics(latency_ms=0.0)
    assert validate_latency(zero, ok, REQ).status == CriterionStatus.NOT_APPLICABLE


def test_artifact_validation_rules():
    good = Artifact(path="a", exists=True, size_bytes=10, checksum_sha256="ab", expected_checksum_sha256="ab", checksum_valid=True)
    assert validate_artifact(good, REQ).status == CriterionStatus.PASS
    assert validate_artifact(None, REQ).status == CriterionStatus.FAIL
    missing = Artifact(path="a", exists=False)
    assert "does not exist" in validate_artifact(missing, REQ).message
    mismatch = Artifact(path="a", exists=True, size_bytes=10, checksum_valid=False)
    assert "checksum mismatch" in validate_artifact(mismatch, REQ).message
    unverifiable = Artifact(path="a", exists=True, size_bytes=10, checksum_valid=None)
    assert validate_artifact(unverifiable, REQ).status == CriterionStatus.FAIL
    assert validate_artifact(unverifiable, OPT).status == CriterionStatus.WARN


def test_real_file_checksum_helpers(tmp_path):
    f = tmp_path / "model.onnx"
    f.write_bytes(b"\x00\x01binary\r\ncontent")  # CRLF bytes must survive (binary mode)
    digest = compute_sha256(f)
    assert len(digest) == 64 and verify_file_checksum(f, digest.upper())
    art = artifact_from_file(f, expected_sha256=digest, source_model="m")
    assert art.exists and art.checksum_valid is True and art.format == "onnx" and art.size_bytes == f.stat().st_size
    assert art.metadata == {"source_model": "m"}
    assert artifact_from_file(tmp_path / "nope.onnx").exists is False


def test_reproducibility_assessment_levels():
    m = Metrics(accuracy=0.5, latency_ms=10.0, memory_mb=50.0)
    assert assess_reproducibility(["a"]).level == ReproducibilityLevel.NOT_VERIFIED
    assert assess_reproducibility(["a", "a", "a"]).level == ReproducibilityLevel.BITWISE_REPRODUCIBLE
    functional = assess_reproducibility(["a", "b"], [m, Metrics(accuracy=0.5005, latency_ms=10.05, memory_mb=50.1)])
    assert functional.level == ReproducibilityLevel.FUNCTIONALLY_REPRODUCIBLE
    divergent = assess_reproducibility(["a", "b"], [m, Metrics(accuracy=0.42, latency_ms=10.0, memory_mb=50.0)])
    assert divergent.level == ReproducibilityLevel.NOT_REPRODUCIBLE
    assert assess_reproducibility(["a", "b"]).level == ReproducibilityLevel.NOT_REPRODUCIBLE


def test_reproducibility_validation():
    req = CriterionConfig("reproducibility", None, True)
    opt = CriterionConfig("reproducibility", None, False)
    assert validate_reproducibility(Reproducibility(ReproducibilityLevel.BITWISE_REPRODUCIBLE, 2), req).status == CriterionStatus.PASS
    assert validate_reproducibility(Reproducibility(ReproducibilityLevel.FUNCTIONALLY_REPRODUCIBLE, 3), req).status == CriterionStatus.PASS
    assert validate_reproducibility(Reproducibility(ReproducibilityLevel.NOT_VERIFIED, 1), req).status == CriterionStatus.FAIL
    assert validate_reproducibility(Reproducibility(ReproducibilityLevel.NOT_VERIFIED, 1), opt).status == CriterionStatus.NOT_APPLICABLE
    assert validate_reproducibility(Reproducibility(ReproducibilityLevel.NOT_REPRODUCIBLE, 2), req).status == CriterionStatus.FAIL
    assert validate_reproducibility(None, req).status == CriterionStatus.FAIL

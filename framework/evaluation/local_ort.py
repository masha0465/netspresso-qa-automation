"""Local ONNX Runtime evaluation engine (0 credits, no network).

Measures latency (median / p95 / min / max / mean), process memory (RSS before,
after load, peak, delta) and captures outputs for the **output equivalence proxy**.

The proxy compares baseline and optimized outputs on identical deterministic inputs.
It is a proxy for functional equivalence and is NEVER reported as accuracy.

All heavy dependencies are injectable so the arithmetic is unit-tested without them:
``session_factory(model_path, threads)`` -> object with ``get_inputs()``, ``get_outputs()``,
``run(None, feed)``; ``input_factory(shape, seed)`` -> array-like; ``memory_probe()`` -> RSS bytes.
"""

from __future__ import annotations

import importlib
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PASS, FAIL, NOT_APPLICABLE = "PASS", "FAIL", "NOT_APPLICABLE"
PROXY_DISCLAIMER = "output equivalence PROXY on identical deterministic inputs - NOT a labeled accuracy measurement"


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
@dataclass
class LatencyStats:
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    mean_ms: float
    iterations: int
    warmup: int
    source: str = "local_onnxruntime_cpu"

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile on an already sorted sequence (q in [0, 100])."""
    if not sorted_values:
        raise ValueError("no samples")
    rank = max(1, math.ceil(q / 100.0 * len(sorted_values)))
    return float(sorted_values[rank - 1])


def compute_latency_stats(samples_ms: Sequence[float], warmup: int) -> LatencyStats:
    if not samples_ms:
        raise ValueError("no latency samples")
    s = sorted(float(x) for x in samples_ms)
    n = len(s)
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0
    return LatencyStats(
        median_ms=round(median, 4),
        p95_ms=round(percentile(s, 95), 4),
        min_ms=round(s[0], 4),
        max_ms=round(s[-1], 4),
        mean_ms=round(sum(s) / n, 4),
        iterations=n,
        warmup=warmup,
    )


@dataclass
class MemoryStats:
    status: str  # PASS (measured) | NOT_APPLICABLE
    rss_before_mb: float | None = None
    rss_after_load_mb: float | None = None
    rss_peak_mb: float | None = None
    delta_mb: float | None = None  # peak - before (model-attributable upper bound)
    method: str = "psutil process RSS (CPU only; GPU not applicable)"

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def memory_stats(before: int | None, after_load: int | None, peak: int | None) -> MemoryStats:
    if before is None or peak is None:
        return MemoryStats(status=NOT_APPLICABLE, method="memory probe unavailable")
    mb = 1024 * 1024
    return MemoryStats(
        status=PASS,
        rss_before_mb=round(before / mb, 3),
        rss_after_load_mb=round(after_load / mb, 3) if after_load is not None else None,
        rss_peak_mb=round(peak / mb, 3),
        delta_mb=round((peak - before) / mb, 3),
    )


# --------------------------------------------------------------------------- #
# output equivalence proxy
# --------------------------------------------------------------------------- #
@dataclass
class OutputEquivalenceProxy:
    status: str
    min_cosine_similarity: float | None
    max_abs_diff: float | None
    mean_abs_diff: float | None
    relative_diff: float | None  # ||a-b|| / ||a||
    shape_match: bool
    dtype_match: bool
    per_output: list[dict[str, Any]] = field(default_factory=list)
    threshold_min_cosine: float | None = None
    note: str = PROXY_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "min_cosine_similarity": self.min_cosine_similarity,
            "max_abs_diff": self.max_abs_diff,
            "mean_abs_diff": self.mean_abs_diff,
            "relative_diff": self.relative_diff,
            "shape_match": self.shape_match,
            "dtype_match": self.dtype_match,
            "per_output": list(self.per_output),
            "threshold_min_cosine": self.threshold_min_cosine,
            "note": self.note,
        }


def _flatten(x: Any) -> list[float]:
    if hasattr(x, "ravel"):
        return [float(v) for v in x.ravel().tolist()]
    if isinstance(x, (list, tuple)):
        out: list[float] = []
        for item in x:
            out.extend(_flatten(item))
        return out
    return [float(x)]


def compare_output_pair(a: Any, b: Any) -> dict[str, float]:
    fa, fb = _flatten(a), _flatten(b)
    if len(fa) != len(fb) or not fa:
        return {"cosine_similarity": 0.0, "max_abs_diff": math.inf, "mean_abs_diff": math.inf, "relative_diff": math.inf, "elements": len(fa)}
    dot = sum(x * y for x, y in zip(fa, fb, strict=True))
    na = math.sqrt(sum(x * x for x in fa))
    nb = math.sqrt(sum(y * y for y in fb))
    diffs = [abs(x - y) for x, y in zip(fa, fb, strict=True)]
    cos = dot / (na * nb) if na > 0 and nb > 0 else (1.0 if na == nb == 0 else 0.0)
    rel = math.sqrt(sum(d * d for d in diffs)) / na if na > 0 else (0.0 if max(diffs) == 0 else math.inf)
    return {
        "cosine_similarity": round(cos, 6),
        "max_abs_diff": round(max(diffs), 6),
        "mean_abs_diff": round(sum(diffs) / len(diffs), 6),
        "relative_diff": round(rel, 6),
        "elements": len(fa),
    }


def compare_outputs(
    baseline: Sequence[Any],
    optimized: Sequence[Any],
    *,
    baseline_shapes: Sequence[Sequence[int]],
    optimized_shapes: Sequence[Sequence[int]],
    baseline_dtypes: Sequence[str],
    optimized_dtypes: Sequence[str],
    min_cosine: float,
) -> OutputEquivalenceProxy:
    if not baseline or not optimized:
        return OutputEquivalenceProxy(NOT_APPLICABLE, None, None, None, None, False, False, threshold_min_cosine=min_cosine)
    shape_match = len(baseline) == len(optimized) and all(list(a) == list(b) for a, b in zip(baseline_shapes, optimized_shapes, strict=False))
    dtype_match = len(baseline_dtypes) == len(optimized_dtypes) and all(a == b for a, b in zip(baseline_dtypes, optimized_dtypes, strict=False))
    if not shape_match:
        return OutputEquivalenceProxy(FAIL, None, None, None, None, False, dtype_match,
                                      per_output=[{"index": i, "baseline_shape": list(s)} for i, s in enumerate(baseline_shapes)],
                                      threshold_min_cosine=min_cosine)
    per = []
    for i, (a, b) in enumerate(zip(baseline, optimized, strict=True)):
        stats = compare_output_pair(a, b)
        stats["index"] = i
        stats["shape"] = list(baseline_shapes[i])
        per.append(stats)
    min_cos = min(p["cosine_similarity"] for p in per)
    status = PASS if dtype_match and min_cos >= min_cosine else FAIL
    return OutputEquivalenceProxy(
        status=status,
        min_cosine_similarity=min_cos,
        max_abs_diff=max(p["max_abs_diff"] for p in per),
        mean_abs_diff=round(sum(p["mean_abs_diff"] for p in per) / len(per), 6),
        relative_diff=max(p["relative_diff"] for p in per),
        shape_match=True,
        dtype_match=dtype_match,
        per_output=per,
        threshold_min_cosine=min_cosine,
    )


# --------------------------------------------------------------------------- #
# evaluator
# --------------------------------------------------------------------------- #
@dataclass
class RunMeasurement:
    model_path: str
    input_name: str
    input_shape: list[int]
    input_dtype: str
    output_names: list[str]
    output_shapes: list[list[int]]
    output_dtypes: list[str]
    latency: LatencyStats
    memory: MemoryStats
    outputs: list[Any]
    provider: str
    threads: int | None

    def to_dict(self, include_outputs: bool = False) -> dict[str, Any]:
        d = {
            "model_path": self.model_path,
            "input_name": self.input_name,
            "input_shape": list(self.input_shape),
            "input_dtype": self.input_dtype,
            "output_names": list(self.output_names),
            "output_shapes": [list(s) for s in self.output_shapes],
            "output_dtypes": list(self.output_dtypes),
            "latency": self.latency.to_dict(),
            "memory": self.memory.to_dict(),
            "provider": self.provider,
            "threads": self.threads,
        }
        if include_outputs:
            d["outputs"] = [_flatten(o)[:16] for o in self.outputs]  # preview only
        return d


def _default_session_factory(model_path: str, threads: int | None, provider: str) -> Any:
    ort = importlib.import_module("onnxruntime")
    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = int(threads)
    return ort.InferenceSession(model_path, sess_options=opts, providers=[provider])


def _default_input_factory(shape: Sequence[int], seed: int, dtype: str = "float32") -> Any:
    np = importlib.import_module("numpy")
    rng = np.random.default_rng(seed)
    return rng.random(tuple(int(d) if d else 1 for d in shape), dtype=getattr(np, dtype) if hasattr(np, dtype) else np.float32)


def _default_memory_probe() -> int | None:
    try:
        psutil = importlib.import_module("psutil")
    except ImportError:
        return None
    return int(psutil.Process().memory_info().rss)


def _dtype_name(ort_type: str) -> str:
    # e.g. "tensor(float)" -> float32
    inner = ort_type[len("tensor("):-1] if ort_type.startswith("tensor(") else ort_type
    return {"float": "float32", "double": "float64", "float16": "float16", "int64": "int64", "int32": "int32", "uint8": "uint8"}.get(inner, inner)


class LocalOrtEvaluator:
    credit_consuming = False

    def __init__(
        self,
        *,
        warmup: int = 10,
        iterations: int = 100,
        seed: int = 0,
        threads: int | None = None,
        provider: str = "CPUExecutionProvider",
        session_factory: Callable[[str, int | None, str], Any] | None = None,
        input_factory: Callable[[Sequence[int], int], Any] | None = None,
        memory_probe: Callable[[], int | None] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        self.warmup, self.iterations, self.seed, self.threads, self.provider = warmup, iterations, seed, threads, provider
        self._session_factory = session_factory or _default_session_factory
        self._input_factory = input_factory or _default_input_factory
        self._memory_probe = memory_probe or _default_memory_probe
        self._clock = clock

    def measure(self, model_path: Path | str, input_shape_override: Sequence[int] | None = None) -> RunMeasurement:
        path = Path(model_path).as_posix()
        rss_before = self._memory_probe()
        session = self._session_factory(path, self.threads, self.provider)
        rss_after_load = self._memory_probe()
        inp = session.get_inputs()[0]
        shape = [int(d) if isinstance(d, int) and d > 0 else 1 for d in (input_shape_override or inp.shape)]
        dtype = _dtype_name(getattr(inp, "type", "tensor(float)"))
        x = self._input_factory(shape, self.seed)
        feed = {inp.name: x}

        for _ in range(self.warmup):
            session.run(None, feed)
        peak = rss_after_load
        samples: list[float] = []
        outputs: list[Any] = []
        for i in range(self.iterations):
            t0 = self._clock()
            outputs = session.run(None, feed)
            samples.append((self._clock() - t0) * 1000.0)
            if i % 10 == 0:
                rss = self._memory_probe()
                if rss is not None and (peak is None or rss > peak):
                    peak = rss
        rss_end = self._memory_probe()
        if rss_end is not None and (peak is None or rss_end > peak):
            peak = rss_end

        return RunMeasurement(
            model_path=path,
            input_name=inp.name,
            input_shape=shape,
            input_dtype=dtype,
            output_names=[o.name for o in session.get_outputs()],
            output_shapes=[list(getattr(o, "shape", [])) for o in outputs],
            output_dtypes=[str(getattr(o, "dtype", "unknown")) for o in outputs],
            latency=compute_latency_stats(samples, self.warmup),
            memory=memory_stats(rss_before, rss_after_load, peak),
            outputs=list(outputs),
            provider=self.provider,
            threads=self.threads,
        )


# --------------------------------------------------------------------------- #
# process-isolated measurement (memory deltas are only meaningful in a fresh process)
# --------------------------------------------------------------------------- #
def measurement_from_dict(d: dict[str, Any]) -> RunMeasurement:
    lat, mem = d["latency"], d["memory"]
    return RunMeasurement(
        model_path=d["model_path"], input_name=d["input_name"], input_shape=list(d["input_shape"]), input_dtype=d["input_dtype"],
        output_names=list(d["output_names"]), output_shapes=[list(s) for s in d["output_shapes"]], output_dtypes=list(d["output_dtypes"]),
        latency=LatencyStats(**{k: lat[k] for k in ("median_ms", "p95_ms", "min_ms", "max_ms", "mean_ms", "iterations", "warmup", "source")}),
        memory=MemoryStats(**{k: mem[k] for k in ("status", "rss_before_mb", "rss_after_load_mb", "rss_peak_mb", "delta_mb", "method")}),
        outputs=list(d.get("outputs_flat", [])), provider=d["provider"], threads=d.get("threads"),
    )


def measure_in_subprocess(model_path: Path | str, *, warmup: int, iterations: int, seed: int, threads: int | None, provider: str,
                          input_shape: Sequence[int] | None = None, python: str | None = None, timeout_s: int = 1800) -> RunMeasurement:
    """Run :meth:`LocalOrtEvaluator.measure` in a fresh interpreter so RSS deltas are per model.

    The child process runs this module as a script (``python -m framework.evaluation.local_ort``);
    outputs are returned flattened for the equivalence proxy.
    """
    cmd = [python or sys.executable, "-m", "framework.evaluation.local_ort", "--model", Path(model_path).as_posix(), "--warmup", str(warmup),
           "--iterations", str(iterations), "--seed", str(seed), "--provider", provider]
    if threads:
        cmd += ["--threads", str(threads)]
    if input_shape:
        cmd += ["--input-shape", ",".join(str(int(d)) for d in input_shape)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=timeout_s, check=False)  # noqa: S603 - fixed argv, no shell
    if proc.returncode != 0:
        raise RuntimeError(f"isolated measurement failed (exit {proc.returncode}): {proc.stderr[-800:]}")
    payload = json.loads(proc.stdout.splitlines()[-1])
    return measurement_from_dict(payload)


def _cli(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Measure one ONNX model with ONNX Runtime in this process and print JSON (last line).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--provider", default="CPUExecutionProvider")
    ap.add_argument("--input-shape", default=None)
    a = ap.parse_args(argv)
    shape = [int(x) for x in a.input_shape.split(",")] if a.input_shape else None
    run = LocalOrtEvaluator(warmup=a.warmup, iterations=a.iterations, seed=a.seed, threads=a.threads, provider=a.provider).measure(a.model, shape)
    payload = run.to_dict()
    payload["outputs_flat"] = [_flatten(o) for o in run.outputs]
    sys.stdout.write(json.dumps(payload) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess in the 3.11 environment
    raise SystemExit(_cli())

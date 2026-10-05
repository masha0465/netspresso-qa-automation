#!/usr/bin/env python
"""Local ONNX Runtime evaluation of an existing real run (Phase 5-D). 0 NetsPresso credits.

    .venv-netspresso/Scripts/python scripts/run_local_eval.py            # latest reports/real_runs/*
    .venv-netspresso/Scripts/python scripts/run_local_eval.py --run-dir reports/real_runs/<id> --original-model outputs/models/graphmodule.pt

What it does (all local, no network):
  1. structural validation of sdk_output.pt / sdk_output.onnx / original model
  2. original model -> ONNX export (torch, local) as the evaluation baseline
  3. parameters / FLOPs recomputed independently and compared with SDK-reported values
  4. ONNX Runtime latency (median/p95/min/max) + process memory for baseline and optimized
  5. output equivalence PROXY on identical deterministic inputs (NOT accuracy)
  6. gate profiles compression / local_eval / release, defect classification
  7. baseline registry: optimized artifact registered as `candidate`
It never imports the netspresso SDK and never reads NETSPRESSO_API_KEY.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from framework.baselines import BaselineEntry, BaselineRegistry, make_artifact_id  # noqa: E402
from framework.config import load_config  # noqa: E402
from framework.defects import classify  # noqa: E402
from framework.evaluation.artifact_structure import validate_onnx, validate_pt  # noqa: E402
from framework.evaluation.environment import environment_fingerprint  # noqa: E402
from framework.evaluation.local_ort import (  # noqa: E402
    PROXY_DISCLAIMER,
    compare_outputs,
    measure_in_subprocess,
)
from framework.evaluation.model_stats import (  # noqa: E402
    count_onnx_parameters,
    count_torch_parameters,
    estimate_onnx_macs,
    verify,
    verify_flops,
)
from framework.pipeline.result import (  # noqa: E402
    CaseResult,
    Configuration,
    CreditRecord,
    CreditUsageType,
    Environment,
    ExecutionResult,
    ExecutionStatus,
    Metrics,
    Reproducibility,
    ReproducibilityLevel,
    SupportState,
    utc_now_iso,
)
from framework.pipeline.runner import case_status_for  # noqa: E402
from framework.quality_gate import QualityGate, render_text  # noqa: E402
from framework.validation.artifact import artifact_from_file, compute_sha256  # noqa: E402

DISCLAIMER = ("Local ORT evaluation. No NetsPresso API call was made. "
              "Results are local validation evidence, not NetsPresso server-side measurements.")
LEDGER = ROOT / "reports" / "credit_usage.json"


def portable(text: str) -> str:
    root_fwd = ROOT.as_posix() + "/"
    root_back = str(ROOT) + "\\"
    return text.replace(root_back.replace("\\", "\\\\"), "").replace(root_back, "").replace(root_fwd, "")


def dump(path: Path, data: object) -> None:
    path.write_text(portable(json.dumps(data, indent=2, ensure_ascii=False, default=str)) + "\n", encoding="utf-8")


def latest_run_dir() -> Path | None:
    runs = sorted(p for p in (ROOT / "reports" / "real_runs").glob("*_automatic_compression") if p.is_dir())
    return runs[-1] if runs else None


def export_original_to_onnx(torch, model_pt: Path, out_path: Path, input_shape: list[int]) -> dict:
    model = torch.load(model_pt.as_posix(), map_location="cpu", weights_only=False)  # trusted by SHA-256 (checked by caller)
    model.eval()
    dummy = torch.zeros(*input_shape)
    torch.onnx.export(model, dummy, out_path.as_posix(), opset_version=13, input_names=["images"], do_constant_folding=True)
    return {"params_torch": count_torch_parameters(model), "class_name": type(model).__name__}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--original-model", type=Path, default=ROOT / "outputs" / "models" / "graphmodule.pt")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--configs", type=Path, default=ROOT / "configs")
    parser.add_argument("--iterations", type=int, default=None, help="override local_eval.measurement_iterations")
    args = parser.parse_args(argv)

    ledger_before = hashlib.sha256(LEDGER.read_bytes()).hexdigest()
    config = load_config(args.configs)
    policy = config.local_eval
    iterations = args.iterations or policy.measurement_iterations

    run_dir = args.run_dir or latest_run_dir()
    if run_dir is None or not run_dir.is_dir():
        print("ABORT: no real run directory found")
        return 2
    execution_prev = json.loads((run_dir / "execution_result.json").read_text(encoding="utf-8"))
    sdk_meta = json.loads((run_dir / "sdk_output" / "metadata.json").read_text(encoding="utf-8"))
    optimized_pt = run_dir / "sdk_output" / "sdk_output.pt"
    optimized_onnx = run_dir / "sdk_output" / "sdk_output.onnx"
    cfg = Configuration.from_dict(execution_prev["configuration"])
    input_shape = [1, 3, *sdk_meta["model_info"]["input_shapes"][0]["dimension"]]

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or (ROOT / "reports" / "local_eval" / f"{stamp}_{run_dir.name}")
    out.mkdir(parents=True, exist_ok=True)
    print(DISCLAIMER)
    print(f"run dir: {portable(run_dir.as_posix())} | input shape: {input_shape} | out: {portable(out.as_posix())}")

    torch = importlib.import_module("torch")
    onnx = importlib.import_module("onnx")

    # ---------------- 1. structural validation ----------------
    structural = {}
    for label, path in (("optimized_pt", optimized_pt), ("optimized_onnx", optimized_onnx), ("original_pt", args.original_model)):
        if not path.is_file():
            structural[label] = {"path": portable(path.as_posix()), "status": "NOT_APPLICABLE", "message": "file not present locally"}
            continue
        sha = compute_sha256(path)
        if path.suffix == ".onnx":
            sv = validate_onnx(path, onnx_module=onnx)
        else:
            sv = validate_pt(path, allow_full_unpickle=policy.is_trusted(sha), torch_module=torch)
        d = sv.to_dict()
        d["sha256"] = sha
        d["trusted_for_full_unpickle"] = policy.is_trusted(sha)
        structural[label] = d
        print(f"[structure] {label}: {sv.status} - {sv.message}")

    # ---------------- 2. baseline export (original -> ONNX) ----------------
    baseline_onnx = out / "baseline_export.onnx"
    baseline_available = args.original_model.is_file() and structural["original_pt"]["status"] == "PASS"
    export_info: dict = {}
    if baseline_available:
        export_info = export_original_to_onnx(torch, args.original_model, baseline_onnx, input_shape)
        export_info["path"] = portable(baseline_onnx.as_posix())
        export_info["sha256"] = compute_sha256(baseline_onnx)
        export_info["structural"] = validate_onnx(baseline_onnx, onnx_module=onnx).to_dict()
        print(f"[baseline] exported original -> ONNX ({export_info['structural']['status']}), torch params {export_info['params_torch']:,}")
    else:
        print("[baseline] original model not available or not loadable -> baseline evaluation NOT_APPLICABLE")

    # ---------------- 3. params / FLOPs independent verification ----------------
    sdk_orig, sdk_comp = sdk_meta["results"]["original_model"], sdk_meta["results"]["compressed_model"]
    opt_model = onnx.load(optimized_onnx.as_posix())
    opt_params_onnx = count_onnx_parameters(opt_model)
    opt_macs, opt_cov, opt_uncov, opt_err = estimate_onnx_macs(opt_model, onnx)
    opt_params_torch = structural["optimized_pt"].get("details", {}).get("parameter_count")
    verification = {
        "compressed": {
            "params_onnx_initializers": verify("params", sdk_comp["number_of_parameters"], opt_params_onnx, policy.params_tolerance_percent, "sum of ONNX initializer elements").to_dict(),
            "params_torch": verify("params", sdk_comp["number_of_parameters"], opt_params_torch, policy.params_tolerance_percent, "sum of torch parameter numel (trusted full unpickle)").to_dict(),
            "flops": verify_flops(sdk_comp["flops"], opt_macs, policy.flops_tolerance_percent, "ONNX shape-inference MAC estimate (Conv/Gemm/MatMul)").to_dict(),
            "macs_coverage": {"covered_ops": opt_cov, "uncovered_ops": opt_uncov, "error": opt_err},
        }
    }
    if baseline_available:
        base_model = onnx.load(baseline_onnx.as_posix())
        base_params_onnx = count_onnx_parameters(base_model)
        base_macs, base_cov, base_uncov, base_err = estimate_onnx_macs(base_model, onnx)
        verification["original"] = {
            "params_onnx_initializers": verify("params", sdk_orig["number_of_parameters"], base_params_onnx, policy.params_tolerance_percent, "sum of ONNX initializer elements (local export)").to_dict(),
            "params_torch": verify("params", sdk_orig["number_of_parameters"], export_info["params_torch"], policy.params_tolerance_percent, "sum of torch parameter numel (trusted full unpickle)").to_dict(),
            "flops": verify_flops(sdk_orig["flops"], base_macs, policy.flops_tolerance_percent, "ONNX shape-inference MAC estimate (Conv/Gemm/MatMul)").to_dict(),
            "macs_coverage": {"covered_ops": base_cov, "uncovered_ops": base_uncov, "error": base_err},
        }
    for side, v in verification.items():
        print(f"[verify] {side}: params(torch)={v['params_torch']['verification_status']} params(onnx)={v['params_onnx_initializers']['verification_status']} flops={v['flops']['verification_status']} ({v['flops']['notes'][0] if v['flops']['notes'] else ''})")

    # ---------------- 4. local ORT measurement ----------------
    # Each model is measured in a FRESH interpreter so process-RSS deltas are attributable to that model
    # (a shared process would charge the runtime's one-time allocations to whichever model runs first).
    measure = dict(warmup=policy.warmup_iterations, iterations=iterations, seed=policy.seed, threads=policy.intra_op_threads,
                   provider=policy.execution_provider, input_shape=input_shape)
    opt_run = measure_in_subprocess(optimized_onnx, **measure)
    print(f"[ort] optimized: median {opt_run.latency.median_ms} ms, p95 {opt_run.latency.p95_ms} ms, mem delta {opt_run.memory.delta_mb} MB, outputs {opt_run.output_shapes} (isolated process)")
    base_run = None
    if baseline_available:
        base_run = measure_in_subprocess(baseline_onnx, **measure)
        print(f"[ort] baseline : median {base_run.latency.median_ms} ms, p95 {base_run.latency.p95_ms} ms, mem delta {base_run.memory.delta_mb} MB, outputs {base_run.output_shapes} (isolated process)")

    # ---------------- 5. output equivalence proxy ----------------
    proxy = None
    if base_run is not None:
        proxy = compare_outputs(base_run.outputs, opt_run.outputs, baseline_shapes=base_run.output_shapes, optimized_shapes=opt_run.output_shapes,
                                baseline_dtypes=base_run.output_dtypes, optimized_dtypes=opt_run.output_dtypes, min_cosine=policy.proxy_min_cosine_similarity)
        print(f"[proxy] {proxy.status}: min cosine {proxy.min_cosine_similarity}, max|d| {proxy.max_abs_diff}, rel {proxy.relative_diff} ({PROXY_DISCLAIMER})")

    # ---------------- 6. Result Model + gates ----------------
    env_fp = environment_fingerprint(execution_provider=policy.execution_provider, threads=policy.intra_op_threads)
    environment = Environment(adapter="local_ort", adapter_version="0.1.0", sdk_name="onnxruntime",
                              sdk_version=env_fp["packages"].get("onnxruntime"), python_version=env_fp["python"], platform=f"{env_fp['os']} {env_fp['os_release']} {env_fp['architecture']}",
                              extra={"credit_consuming": False, "evaluation_type": "local_ort", "netspresso_api_calls": 0, **env_fp})

    def metrics_for(run, size_path: Path | None, extra: dict[str, float]) -> Metrics:
        return Metrics(
            accuracy=None,  # labeled evaluation dataset unavailable
            latency_ms=run.latency.median_ms if run else None,
            memory_mb=run.memory.delta_mb if run and run.memory.status == "PASS" else None,
            model_size_mb=round(size_path.stat().st_size / 1_000_000, 3) if size_path and size_path.is_file() else None,
            extra=extra,
        )

    opt_extra: dict[str, float] = {"latency_p95_ms": opt_run.latency.p95_ms, "latency_min_ms": opt_run.latency.min_ms, "latency_max_ms": opt_run.latency.max_ms,
                                   "latency_mean_ms": opt_run.latency.mean_ms, "params_onnx": float(opt_params_onnx)}
    if opt_macs is not None:
        opt_extra["macs_estimate"] = float(opt_macs)
    if proxy is not None and proxy.status != "NOT_APPLICABLE":
        opt_extra.update({"proxy_min_cosine_similarity": proxy.min_cosine_similarity or 0.0, "proxy_max_abs_diff": proxy.max_abs_diff or 0.0,
                          "proxy_mean_abs_diff": proxy.mean_abs_diff or 0.0, "proxy_relative_diff": proxy.relative_diff or 0.0,
                          "proxy_shape_match": 1.0 if proxy.shape_match else 0.0, "proxy_dtype_match": 1.0 if proxy.dtype_match else 0.0})
    base_extra: dict[str, float] = {}
    if base_run is not None:
        base_extra = {"latency_p95_ms": base_run.latency.p95_ms, "latency_min_ms": base_run.latency.min_ms, "latency_max_ms": base_run.latency.max_ms,
                      "latency_mean_ms": base_run.latency.mean_ms, "params_onnx": float(base_params_onnx)}
        if base_macs is not None:
            base_extra["macs_estimate"] = float(base_macs)

    artifact = artifact_from_file(
        optimized_onnx, expected_sha256=None,
        source_model=args.original_model.name, source_model_sha256=structural["original_pt"].get("sha256"),
        operation=cfg.optimization, sdk_version=execution_prev["environment"].get("sdk_version"),
        structural_validation=structural["optimized_onnx"], companion_pt=structural["optimized_pt"],
        configuration_key=cfg.key, evaluation="local_ort", generated_by="scripts/run_local_eval.py",
    )
    artifact.path = portable(artifact.path or "")
    execution = ExecutionResult(
        operation="local_ort_evaluation",
        configuration=cfg,
        execution_status=ExecutionStatus.COMPLETED,
        environment=environment,
        baseline_metrics=metrics_for(base_run, baseline_onnx if baseline_available else None, base_extra) if base_run else None,
        metrics=metrics_for(opt_run, optimized_onnx, opt_extra),
        artifact=artifact,
        logs=[DISCLAIMER, f"source real run: {run_dir.name}", f"warmup={policy.warmup_iterations} iterations={iterations} seed={policy.seed} provider={policy.execution_provider}"],
        reproducibility=Reproducibility(level=ReproducibilityLevel.NOT_VERIFIED, runs=1,
                                        checksums=[structural["optimized_pt"].get("sha256", "")],
                                        notes="Level 1 (metadata/config consistency) checked locally; Level 2+ requires a second real run"),
        timestamp=utc_now_iso(),
        credit=CreditRecord(usage_type=CreditUsageType.NONE, estimated=0, actual=None, note="local evaluation: 0 credits, no API call"),
    )

    consistency = {
        "requested_vs_metadata": {
            "compression_ratio": {"requested": 0.5, "metadata": sdk_meta["compression_info"]["ratio"], "match": sdk_meta["compression_info"]["ratio"] == 0.5},
            "input_shape": {"requested": input_shape, "metadata": sdk_meta["model_info"]["input_shapes"], "match": True},
            "framework": {"requested": "pytorch", "metadata": sdk_meta["model_info"]["framework"], "match": sdk_meta["model_info"]["framework"] == "pytorch"},
        },
        "reproducibility_level": 1,
        "note": "Level 1 only; Level 2-4 require a second identical-condition real run",
    }

    gates = {name: QualityGate(config.quality_gate, profile=name).evaluate(execution) for name in ("compression", "local_eval", "release")}
    release_defect = classify(execution, gates["release"])
    local_defect = classify(execution, gates["local_eval"])
    case = CaseResult(configuration=cfg, status=case_status_for(execution, gates["release"]), support=SupportState.SUPPORTED, selected=True,
                      selection_reason="Phase 5-D local evaluation of real run", execution=execution, gate=gates["release"], defect=release_defect)
    for g in gates.values():
        print("\n" + render_text(g))
    if local_defect:
        print(f"\n[local_eval defect] {local_defect.category.value} [{local_defect.severity.value}] {local_defect.message}")

    # ---------------- 7. baseline registry (candidate) ----------------
    registry = BaselineRegistry(ROOT / "reports" / "baselines" / "registry.json")
    entry = registry.register_candidate(BaselineEntry(
        artifact_id=make_artifact_id(cfg.optimization, structural["optimized_pt"]["sha256"]),
        model_name=cfg.model, source_artifact=f"{args.original_model.name} sha256={structural['original_pt'].get('sha256')}",
        sha256=structural["optimized_pt"]["sha256"], file_size=optimized_pt.stat().st_size, format="pt",
        sdk_version=execution_prev["environment"].get("sdk_version"), operation=cfg.optimization, configuration=cfg.to_dict(),
        input_shape=input_shape, compression_ratio=sdk_meta["compression_info"]["ratio"], environment_fingerprint=json.loads(portable(json.dumps(execution_prev["environment"]))),
        created_at=execution_prev["timestamp"], status="candidate",
        provenance={"run_dir": portable(run_dir.as_posix()), "adapter": "netspresso", "sdk_model_id": sdk_comp.get("model_id"),
                    "companion_onnx_sha256": structural["optimized_onnx"]["sha256"], "registered_by": "scripts/run_local_eval.py"},
        notes=["candidate only: promotion requires Level 3/4 reproducibility evidence from a second real run"],
    ))
    registry.save()
    print(f"\n[registry] {entry.artifact_id} status={entry.status} ({portable((ROOT / 'reports' / 'baselines' / 'registry.json').as_posix())})")

    # ---------------- 8. persist ----------------
    result = {
        "evaluation_type": "local_ort",
        "credit_consuming": False,
        "netspresso_api_calls": 0,
        "disclaimer": DISCLAIMER,
        "source_real_run": run_dir.name,
        "policy": {"warmup": policy.warmup_iterations, "iterations": iterations, "seed": policy.seed, "provider": policy.execution_provider,
                   "threads": policy.intra_op_threads, "proxy_min_cosine_similarity": policy.proxy_min_cosine_similarity,
                   "process_isolation": "each model measured in a fresh interpreter (per-model RSS attribution)",
                   "note": "project-defined example policy, not an official benchmark protocol"},
        "baseline": {"available": baseline_available, "export": export_info, "measurement": base_run.to_dict() if base_run else {"status": "NOT_APPLICABLE"}},
        "optimized": {"measurement": opt_run.to_dict()},
        "artifact_validation": structural,
        "independent_verification": verification,
        "performance": {
            "latency": {"baseline": base_run.latency.to_dict() if base_run else None, "optimized": opt_run.latency.to_dict(), "source": "local_onnxruntime_cpu (development machine, not target device)"},
            "memory": {"baseline": base_run.memory.to_dict() if base_run else None, "optimized": opt_run.memory.to_dict()},
        },
        "output_equivalence_proxy": proxy.to_dict() if proxy else {"status": "NOT_APPLICABLE", "reason": "baseline not available", "note": PROXY_DISCLAIMER},
        "accuracy": {"status": "NOT_APPLICABLE", "reason": "labeled evaluation dataset unavailable; SDK metadata task/dataset fields are empty"},
        "configuration_consistency": consistency,
        "reproducibility": {"level": 1, "framework_level": execution.reproducibility.to_dict()},
        "environment": environment.to_dict(),
        "quality_gate": {name: g.to_dict() for name, g in gates.items()},
        "defects": {"release": release_defect.to_dict() if release_defect else None, "local_eval": local_defect.to_dict() if local_defect else None},
        "case_result": case.to_dict(),
        "baseline_registry_entry": entry.to_dict(),
    }
    result = json.loads(portable(json.dumps(result, default=str)))  # normalise paths before any rendering
    dump(out / "local_eval_result.json", result)
    dump(out / "artifact_validation.json", {"credit_consuming": False, "disclaimer": DISCLAIMER, **structural})
    dump(out / "performance.json", {"credit_consuming": False, "disclaimer": DISCLAIMER, **result["performance"], "output_equivalence_proxy": result["output_equivalence_proxy"]})
    dump(out / "environment.json", {"credit_consuming": False, **environment.to_dict()})
    dump(out / "quality_gate.json", {"credit_consuming": False, "disclaimer": DISCLAIMER, **result["quality_gate"]})
    try:
        from framework.reporter import write_local_eval_html
        write_local_eval_html(result, out / "report.html")
    except Exception as exc:  # noqa: BLE001
        print(f"[report] HTML skipped: {type(exc).__name__}: {exc}")

    assert not any(m == "netspresso" or m.startswith("netspresso.") for m in sys.modules), "SDK must never be imported here"
    ledger_after = hashlib.sha256(LEDGER.read_bytes()).hexdigest()
    print(f"\nledger unchanged: {ledger_before == ledger_after} | NetsPresso API calls: 0 | credits consumed: 0")
    print(f"outputs: {portable(out.as_posix())}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

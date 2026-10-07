#!/usr/bin/env python
"""Phase 5-E R1 cross-check in the target-side torch 2.0.1 environments (0 NetsPresso credits).

Two modes, both consume the dumps written by scripts/yolov8_r1_traceable_export.py
(outputs/models/yolov8n_fx_r1/diag/upstream_reference.npz):

  --torch-only   (.venv-netspresso: torch 2.0.1, NO ultralytics)                       TC-Y8-R1-011
      load the R1 GraphModule with torch 2.0.1, run it on the dumped input, compare the body outputs
      with the upstream (torch 2.14) outputs. "Loadable in a torch 2.0.1 environment" only - it does
      NOT claim that the NetsPresso server will accept the artifact.

  default        (.venv-yolo-nota: ultralytics_nota 8.0.108 + torch 2.0.1)              R2 evidence
      1. the same torch 2.0.1 load check
      2. official fork re-attach (YOLO_netspresso / DetectionModel_netspresso) of the R1 body -> decoded output
         vs upstream decoded output (pre-NMS)
      3. layer-by-layer diff: fork-loaded yolov8n.pt vs upstream per-layer dumps -> first divergence + category
      4. the OLD fork-exported body (outputs/models/yolov8n_fx/model_fx.pt) vs upstream body outputs

    env -u NETSPRESSO_API_KEY .venv-netspresso/Scripts/python scripts/yolov8_r1_fork_crosscheck.py --torch-only
    env -u NETSPRESSO_API_KEY .venv-yolo-nota/Scripts/python scripts/yolov8_r1_fork_crosscheck.py

No netspresso SDK import, no credit use. Unverified mechanisms stay NOT VERIFIED.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from framework.config import load_config  # noqa: E402
from framework.evaluation.environment import environment_fingerprint  # noqa: E402
from framework.evaluation.fx_traceability import (  # noqa: E402
    classify_divergence,
    equivalence_stats,
    first_divergence,
)
from framework.validation.artifact import compute_sha256  # noqa: E402

LEDGER = ROOT / "reports" / "credit_usage.json"
DATASETS = ROOT / "datasets"
DISCLAIMER = ("R1 cross-check ran locally. No NetsPresso API call was made. A successful torch 2.0.1 load means the artifact is loadable "
              "in a target-side torch 2.0.1 environment; it does not prove NetsPresso server acceptance.")


def portable(text: str) -> str:
    root_fwd = ROOT.as_posix() + "/"
    root_back = str(ROOT) + "\\"
    return text.replace(root_back.replace("\\", "\\\\"), "").replace(root_back, "").replace(root_fwd, "")


def stats_np(a, b) -> dict:
    return equivalence_stats(a.reshape(-1).tolist(), b.reshape(-1).tolist())


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--r1-dir", type=Path, default=ROOT / "outputs" / "models" / "yolov8n_fx_r1")
    ap.add_argument("--old-fork-body", type=Path, default=ROOT / "outputs" / "models" / "yolov8n_fx" / "model_fx.pt")
    ap.add_argument("--weights", type=Path, default=ROOT / "outputs" / "models" / "yolov8n.pt")
    ap.add_argument("--torch-only", action="store_true", help="torch-only load/forward check (no ultralytics import)")
    ap.add_argument("--allow-untrusted-body", action="store_true", help="full unpickle of a body whose SHA is not in configs/local_eval.yaml (locally generated artifacts only)")
    ap.add_argument("--layer-tol", type=float, default=1e-4, help="relative L2 tolerance for the layer-by-layer divergence search (PROJECT-DEFINED EXAMPLE)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    fx_path, meta_path, diag = args.r1_dir / "model_fx.pt", args.r1_dir / "netspresso_head_meta.json", args.r1_dir / "diag"
    ref_npz, ref_json = diag / "upstream_reference.npz", diag / "upstream_reference.json"
    if not (fx_path.is_file() and meta_path.is_file() and ref_npz.is_file() and ref_json.is_file()):
        print(f"ABORT: R1 artifacts/dumps missing under {args.r1_dir}; run scripts/yolov8_r1_traceable_export.py first")
        return 2
    ledger_before = hashlib.sha256(LEDGER.read_bytes()).hexdigest()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = "r1_torch201_load" if args.torch_only else "r1_fork_crosscheck"
    out = args.out or (ROOT / "reports" / "yolov8_baseline" / f"{stamp}_{suffix}")
    out.mkdir(parents=True, exist_ok=True)
    print(DISCLAIMER)

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    policy = load_config(ROOT / "configs").local_eval
    ref = np.load(ref_npz)
    ref_meta = json.loads(ref_json.read_text(encoding="utf-8"))
    fx_sha = compute_sha256(fx_path)
    trusted = policy.is_trusted(fx_sha) or args.allow_untrusted_body
    record: dict = {"credit_consuming": False, "netspresso_api_calls": 0, "disclaimer": DISCLAIMER, "mode": "torch-only" if args.torch_only else "fork",
                    "torch_version": torch.__version__, "python": sys.version.split()[0],
                    "r1_body": {"path": portable(fx_path.as_posix()), "sha256": fx_sha, "size_bytes": fx_path.stat().st_size, "matches_dump": fx_sha == ref_meta.get("fx_sha256"),
                                "full_unpickle_allowed": trusted, "trust_basis": "trust list" if policy.is_trusted(fx_sha) else ("--allow-untrusted-body" if trusted else "none")},
                    "reference": {"image": ref_meta["image"], "input_shape": ref_meta["input_shape"], "weights_sha256": ref_meta["weights_sha256"]}}
    if not trusted:
        record["torch_load"] = {"status": "NOT_VERIFIED", "reason": "body SHA not in trust list and --allow-untrusted-body not given; full unpickle refused"}
        print("[load] NOT_VERIFIED (untrusted body)")
        _finish(out, record, ledger_before, suffix)
        return 3

    # ---- 1. torch 2.0.1 load + forward (TC-Y8-R1-011) ----
    x = torch.from_numpy(ref["input"])
    try:
        gm = torch.load(fx_path.as_posix(), map_location="cpu")
        gm.eval()
        is_gm = isinstance(gm, torch.fx.GraphModule)
        with torch.no_grad():
            outs = gm(x)
        body_cmp = [stats_np(ref[f"body_out_{i}"], o.numpy()) for i, o in enumerate(outs)]
        record["torch_load"] = {"status": "PASS" if is_gm else "FAIL", "object": type(gm).__name__, "is_graph_module": is_gm, "parameter_count": sum(p.numel() for p in gm.parameters()),
                                "output_shapes": [list(o.shape) for o in outs], "body_outputs_vs_upstream_torch214": body_cmp,
                                "max_abs_diff": max(c["max_abs_diff"] for c in body_cmp), "bitwise_identical": all(c["bitwise_identical"] for c in body_cmp),
                                "meaning": "artifact loads and runs under torch 2.0.1 (target-side dependency of netspresso-trainer); not a server-acceptance claim"}
        print(f"[load torch {torch.__version__}] {record['torch_load']['status']} GraphModule={is_gm} outputs {record['torch_load']['output_shapes']} | max abs diff vs torch 2.14: {record['torch_load']['max_abs_diff']:.3e} bitwise={record['torch_load']['bitwise_identical']}")
    except Exception as exc:  # noqa: BLE001
        record["torch_load"] = {"status": "FAIL", "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
        print(f"[load torch {torch.__version__}] FAIL {record['torch_load']['error']}")
        _finish(out, record, ledger_before, suffix)
        return 4
    if args.torch_only:
        record["environment"] = environment_fingerprint(packages=("torch", "numpy"))
        _finish(out, record, ledger_before, suffix)
        return 0

    # ---- 2. official fork re-attach of the R1 body ----
    import ultralytics  # noqa: PLC0415
    import ultralytics.yolo.data.utils as _data_utils  # noqa: PLC0415
    import ultralytics.yolo.utils as _utils  # noqa: PLC0415
    from ultralytics import YOLO, YOLO_netspresso  # noqa: PLC0415
    from ultralytics.yolo.utils import SETTINGS  # noqa: PLC0415

    SETTINGS["datasets_dir"] = DATASETS.as_posix()
    _utils.DATASETS_DIR = DATASETS
    _data_utils.DATASETS_DIR = DATASETS
    record["fork"] = {"package": "ultralytics_nota", "version": ultralytics.__version__}
    try:
        net = YOLO_netspresso(compressed_model=fx_path.as_posix(), head_meta=meta_path.as_posix(), task="detect_retraining", meta_config="yolov8n.yaml").model.eval()
        with torch.no_grad():
            y = net(x)
        y = y[0] if isinstance(y, (list, tuple)) else y
        dec = stats_np(ref["decoded"], y.numpy())
        record["fork_reattach_r1_body"] = {"status": "PASS", "class": type(net).__name__, "decoded_shape": list(y.shape), "decoded_vs_upstream_pre_nms": dec,
                                           "boxes_vs_upstream": stats_np(ref["decoded"][:, :4], y[:, :4].numpy()), "scores_vs_upstream": stats_np(ref["decoded"][:, 4:], y[:, 4:].numpy())}
        print(f"[fork re-attach R1] decoded {list(y.shape)} vs upstream: cosine {dec['cosine']:.8f} max_abs {dec['max_abs_diff']:.3e} rel {dec['relative_l2_diff']:.3e}")
    except Exception as exc:  # noqa: BLE001
        record["fork_reattach_r1_body"] = {"status": "FAIL", "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
        print(f"[fork re-attach R1] FAIL {record['fork_reattach_r1_body']['error']}")

    # ---- 3. layer-by-layer: fork-loaded yolov8n.pt vs upstream dumps (R2 evidence) ----
    det_f = YOLO(args.weights.as_posix()).model.float().eval()
    sig_path = diag / "upstream_param_signature.json"
    weights_identical = None
    if sig_path.is_file():
        up_sig = json.loads(sig_path.read_text(encoding="utf-8"))
        mism = []
        for n, p in det_f.named_parameters():
            u = up_sig.get(n)
            if u is None or list(p.shape) != u[0] or abs(float(p.float().sum()) - u[1]) > 1e-3 * max(1.0, abs(u[1])):
                mism.append(n)
        weights_identical = not mism and len(up_sig) == sum(1 for _ in det_f.named_parameters())
        record["parameter_signature"] = {"identical": weights_identical, "mismatched": mism[:20], "upstream_count": len(up_sig), "fork_count": sum(1 for _ in det_f.named_parameters())}
    layer_stats, yv, outp = [], [], x
    with torch.no_grad():
        for m in det_f.model[:-1]:
            if m.f != -1:
                outp = yv[m.f] if isinstance(m.f, int) else [outp if j == -1 else yv[j] for j in m.f]
            outp = m(outp)
            yv.append(outp if m.i in det_f.save else None)
            key = f"layer_{m.i:02d}"
            st = stats_np(ref[key], outp.numpy()) if key in ref else {"status": "MISSING_REFERENCE"}
            layer_stats.append({"index": int(m.i), "layer": key, "layer_type": type(m).__name__, "stats": st})
        feats = [outp if j == -1 else yv[j] for j in det_f.model[-1].f]
        head = det_f.model[-1]
        # fork-contract maps computed in EVAL mode via the branch convs directly (a train-mode forward would update the BN running stats of cv2/cv3)
        maps = [torch.cat((head.cv2[i](feats[i]), head.cv3[i](feats[i])), 1) for i in range(head.nl)]
        fork_dec = det_f(x)
        fork_dec = fork_dec[0] if isinstance(fork_dec, (list, tuple)) else fork_dec
    head_stats = [stats_np(ref[f"body_out_{i}"], mp.numpy()) for i, mp in enumerate(maps)]
    layer_stats.append({"index": int(head.i), "layer": "detect_cv2_cv3_maps", "layer_type": type(head).__name__,
                        "stats": {"status": "COMPUTED", "cosine": min(s["cosine"] for s in head_stats), "max_abs_diff": max(s["max_abs_diff"] for s in head_stats),
                                  "mean_abs_diff": max(s["mean_abs_diff"] for s in head_stats), "relative_l2_diff": max(s["relative_l2_diff"] for s in head_stats),
                                  "bitwise_identical": all(s["bitwise_identical"] for s in head_stats), "count": sum(s["count"] for s in head_stats)}})
    dec_stats = stats_np(ref["decoded"], fork_dec.numpy())
    fd = first_divergence(layer_stats, max_relative_l2_diff=args.layer_tol)
    cls = classify_divergence(fd, input_identical=True, weights_identical=weights_identical, layer_count=len(det_f.model))
    record["fork_model_layer_diff"] = {"tolerance_relative_l2": args.layer_tol, "layers": layer_stats, "decoded_eval_vs_upstream": dec_stats, "first_divergence": fd, "classification": cls,
                                       "model_source": "fork YOLO(yolov8n.pt).model (same checkpoint bytes as upstream)"}
    print(f"[fork layer diff] first divergence: {fd['layer'] if fd else None} | decoded cosine {dec_stats['cosine']:.6f} rel {dec_stats['relative_l2_diff']:.3e} | class: {cls['category']} ({cls['confidence']})")
    for rec in layer_stats:
        s = rec["stats"]
        if s.get("status") == "COMPUTED":
            print(f"    {rec['layer']:<22} {rec['layer_type']:<12} rel {s['relative_l2_diff']:.3e} max_abs {s['max_abs_diff']:.3e} cos {s['cosine']:.8f}")

    # ---- 3b. root-cause probe: the fork's C2f.forward feeds every Bottleneck the SAME split half (out[-1] is a constant
    #          tuple element) instead of chaining them (upstream: y.extend(m(y[-1]))). Re-run the fork model with the
    #          chained semantics bound in-memory; if the divergence vanishes the mechanism is VERIFIED. ----
    from ultralytics.nn.modules.block import C2f as ForkC2f  # noqa: PLC0415

    def chained_forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    c2f_layers = [{"index": int(m.i), "bottlenecks": len(m.m)} for m in det_f.model if isinstance(m, ForkC2f)]
    orig_forward = ForkC2f.forward
    ForkC2f.forward = chained_forward
    try:
        patched_stats, yv2, outp2 = [], [], x
        with torch.no_grad():
            for m in det_f.model[:-1]:
                if m.f != -1:
                    outp2 = yv2[m.f] if isinstance(m.f, int) else [outp2 if j == -1 else yv2[j] for j in m.f]
                outp2 = m(outp2)
                yv2.append(outp2 if m.i in det_f.save else None)
                patched_stats.append({"index": int(m.i), "layer": f"layer_{m.i:02d}", "layer_type": type(m).__name__, "stats": stats_np(ref[f"layer_{m.i:02d}"], outp2.numpy())})
            dec2 = det_f(x)
            dec2 = dec2[0] if isinstance(dec2, (list, tuple)) else dec2
    finally:
        ForkC2f.forward = orig_forward
    fd2 = first_divergence(patched_stats, max_relative_l2_diff=args.layer_tol)
    dec2_stats = stats_np(ref["decoded"], dec2.numpy())
    verified = fd is not None and fd2 is None and dec2_stats["relative_l2_diff"] <= args.layer_tol
    probe = {"description": "fork C2f.forward (every Bottleneck on the same split half) replaced in-memory by the upstream chained semantics",
             "c2f_layers": c2f_layers, "affected_layers_n_ge_2": [c for c in c2f_layers if c["bottlenecks"] >= 2],
             "first_divergence_before": fd, "first_divergence_after": fd2, "decoded_vs_upstream_after": dec2_stats, "root_cause_verified": verified,
             "fork_source": "ultralytics_nota/nn/modules/block.py C2f.forward: out = cv1(x).split(...); y = [m(out[-1]) for m in self.m]",
             "upstream_source": "ultralytics/nn/modules/block.py C2f.forward: y = list(cv1(x).chunk(2, 1)); y.extend(m(y[-1]) for m in self.m)"}
    if verified:
        cls = {"category": "version difference", "confidence": "HIGH", "verified": True,
               "mechanism": ("fork C2f.forward applies each Bottleneck to the same split half instead of chaining them; identical to upstream only when n == 1. "
                             f"yolov8n has C2f blocks with n >= 2 at layers {[c['index'] for c in probe['affected_layers_n_ge_2']]} -> body outputs diverge from there; "
                             "binding the chained forward in-memory removes the divergence")}
        record["fork_model_layer_diff"]["classification"] = cls
    record["fork_c2f_root_cause_probe"] = probe
    print(f"[probe] first divergence before={fd['layer'] if fd else None} after={fd2['layer'] if fd2 else None} | decoded rel after {dec2_stats['relative_l2_diff']:.3e} | root cause verified: {verified}")

    # ---- 4. old fork-exported body vs upstream body outputs ----
    if args.old_fork_body.is_file():
        old_sha = compute_sha256(args.old_fork_body)
        if policy.is_trusted(old_sha) or args.allow_untrusted_body:
            with torch.no_grad():
                old = torch.load(args.old_fork_body.as_posix(), map_location="cpu").eval()
                old_outs = old(x)
            old_cmp = [stats_np(ref[f"body_out_{i}"], o.numpy()) for i, o in enumerate(old_outs)]
            record["old_fork_body_vs_upstream"] = {"path": portable(args.old_fork_body.as_posix()), "sha256": old_sha, "per_level": old_cmp,
                                                   "max_relative_l2_diff": max(c["relative_l2_diff"] for c in old_cmp), "min_cosine": min(c["cosine"] for c in old_cmp)}
            print(f"[old fork body] vs upstream body maps: min cosine {record['old_fork_body_vs_upstream']['min_cosine']:.6f} max rel {record['old_fork_body_vs_upstream']['max_relative_l2_diff']:.3e}")
        else:
            record["old_fork_body_vs_upstream"] = {"status": "NOT_VERIFIED", "reason": "untrusted SHA"}
    record["environment"] = environment_fingerprint(packages=("ultralytics", "torch", "torchvision", "numpy", "psutil"))
    _finish(out, record, ledger_before, suffix)
    return 0


def _finish(out: Path, record: dict, ledger_before: str, suffix: str) -> None:
    assert not any(m == "netspresso" or m.startswith("netspresso.") for m in sys.modules), "SDK must never be imported here"
    record["timestamp"] = datetime.now(UTC).replace(microsecond=0).isoformat()
    record["ledger_unchanged"] = hashlib.sha256(LEDGER.read_bytes()).hexdigest() == ledger_before
    (out / f"{suffix}.json").write_text(portable(json.dumps(record, indent=2, ensure_ascii=False, default=str)) + "\n", encoding="utf-8")
    print(f"ledger unchanged: {record['ledger_unchanged']} | NetsPresso API calls: 0 | record: {portable((out / f'{suffix}.json').as_posix())}")


if __name__ == "__main__":
    raise SystemExit(main())

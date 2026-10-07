#!/usr/bin/env python
"""Phase 5-E R1: UPSTREAM YOLOv8n traceable fx export + functional equivalence (0 NetsPresso credits).

Runs in the upstream environment (``.venv-yolo``: ultralytics 8.4.173, torch 2.14):

    env -u NETSPRESSO_API_KEY .venv-yolo/Scripts/python scripts/yolov8_r1_traceable_export.py

Pipeline (each step is recorded in reports/yolov8_baseline/<stamp>_r1/r1_result.json):

    1. baseline reproduction        upstream YOLO(yolov8n.pt).val(coco128)              TC-Y8-R1-001
    2. traceability analysis        upstream forward path, direct fx trace attempts      (evidence)
    3. traceability wrapper         TraceableYOLOv8Body around UNMODIFIED upstream code  TC-Y8-R1-002
    4. fx export (x2)               outputs/models/yolov8n_fx_r1/model_fx.pt + head meta TC-Y8-R1-003/012/013
    5. artifact validation          framework validate_pt + precheck_fx_bundle           TC-Y8-R1-004/006/007
    6. head re-attach               R1DecodeHead rebuilt from head meta (DFL/anchors)    TC-Y8-R1-005
    7. functional equivalence       pre-NMS raw + decoded tensors vs upstream model      TC-Y8-R1-008
    8. one-image diagnostic         post-NMS boxes / coords / conf / classes             TC-Y8-R1-009
    9. detection evaluation         upstream DetectionValidator on body + head           TC-Y8-R1-010
   10. cross-env dumps              per-layer upstream outputs for the fork cross-check  (R2 input)

The wrapper does not modify ultralytics; it defines the QA traceability boundary (framework.evaluation.fx_traceability).
The exported body has the SAME body/head split as the official fork's export_netspresso(). No netspresso SDK import.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from framework.evaluation.artifact_structure import validate_pt  # noqa: E402
from framework.evaluation.detection_accuracy import DetectionMetrics, compare_detection_accuracy  # noqa: E402
from framework.evaluation.detection_head import (  # noqa: E402
    expected_body_output_shapes,
    expected_decoded_output_shape,
    load_head_meta,
    precheck_fx_bundle,
    validate_output_schema,
)
from framework.evaluation.environment import environment_fingerprint  # noqa: E402
from framework.evaluation.fx_traceability import (  # noqa: E402
    PATCH_ID,
    TRACEABILITY_BOUNDARY,
    build_export_manifest,
    compare_detections,
    equivalence_stats,
    equivalence_verdict,
    summarize_graph,
)
from framework.validation.artifact import compute_sha256  # noqa: E402

LEDGER = ROOT / "reports" / "credit_usage.json"
DATASETS = ROOT / "datasets"
DISCLAIMER = ("R1 traceable fx export ran locally against upstream ultralytics. No NetsPresso API call was made. "
              "Equivalence here is between two local models (upstream vs exported body + rebuilt head); it is not a compression result.")
REG_MAX = 16


def portable(text: str) -> str:
    root_fwd = ROOT.as_posix() + "/"
    root_back = str(ROOT) + "\\"
    return text.replace(root_back.replace("\\", "\\\\"), "").replace(root_back, "").replace(root_fwd, "")


# --------------------------------------------------------------------------- #
# traceability wrapper (the "patch"): no ultralytics code is modified
# --------------------------------------------------------------------------- #
def make_wrapper_classes(torch):
    nn = torch.nn

    class TraceableYOLOv8Body(nn.Module):
        """Upstream DetectionModel layers 0..N-2 + Detect.cv2/cv3, tensor-only forward (fx-traceable).

        Reproduces ``BaseModel._predict_once`` (save/from indices) and the fork's train-mode Detect output
        ``cat(cv2[i](x_i), cv3[i](x_i))`` per level. No Python control flow depends on tensor values.
        """

        patch_id = PATCH_ID

        def __init__(self, det_model, c2f_cls=None):
            super().__init__()
            layers = list(det_model.model)
            detect = layers[-1]
            # traceability patch step 2: upstream C2f.forward iterates a Proxy (list(x.chunk(2, 1)) -> TraceError);
            # bind upstream's own split-based forward (identical chained semantics) on the instances, as the exporter does.
            self.patched_c2f = []
            if c2f_cls is not None:
                for m in det_model.modules():
                    if isinstance(m, c2f_cls):
                        m.forward = m.forward_split
                        self.patched_c2f.append({"index": getattr(m, "i", None), "bottlenecks": len(m.m)})
            self.layers = nn.ModuleList(layers[:-1])
            self.from_idx = [m.f for m in layers[:-1]]
            self.save = set(det_model.save)
            self.detect_from = list(detect.f)
            self.nl = int(detect.nl)
            self.cv2 = detect.cv2  # box branch (4*reg_max ch)
            self.cv3 = detect.cv3  # class branch (nc ch)

        def forward(self, x):
            y = []
            for i, m in enumerate(self.layers):
                f = self.from_idx[i]
                if f != -1:
                    x = y[f] if isinstance(f, int) else [x if j == -1 else y[j] for j in f]
                x = m(x)
                y.append(x if i in self.save else None)
            feats = [x if j == -1 else y[j] for j in self.detect_from]
            return [torch.cat((self.cv2[i](feats[i]), self.cv3[i](feats[i])), 1) for i in range(self.nl)]

    class R1DecodeHead(nn.Module):
        """Decode head rebuilt from netspresso_head_meta.json with upstream helpers (mirror of the fork's Detect_netspresso)."""

        def __init__(self, meta: dict, dfl_cls, make_anchors, dist2bbox):
            super().__init__()
            self.nc, self.nl = int(meta["nc"]), int(meta["nl"])
            self.reg_max = REG_MAX
            self.no = self.nc + 4 * self.reg_max
            self.register_buffer("stride", torch.tensor([float(s) for s in meta["stride"]]))
            self.dfl = dfl_cls(self.reg_max)
            self._make_anchors, self._dist2bbox = make_anchors, dist2bbox
            self.anchors = self.strides = None
            self.shape = None

        def forward(self, feats):
            shape = feats[0].shape
            if self.shape != shape:
                self.anchors, self.strides = (a.transpose(0, 1) for a in self._make_anchors(feats, self.stride, 0.5))
                self.shape = shape
            x_cat = torch.cat([f.view(shape[0], self.no, -1) for f in feats], 2)
            box, cls = x_cat.split((4 * self.reg_max, self.nc), 1)
            dbox = self._dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
            return torch.cat((dbox, cls.sigmoid()), 1)

    class R1Detector(nn.Module):
        """body (GraphModule) + decode head, with the attributes ultralytics' validator expects from a PyTorch model."""

        def __init__(self, body, head, names: dict, stride):
            super().__init__()
            self.body, self.head = body, head
            self.names = dict(names)
            self.stride = torch.tensor([float(s) for s in stride])
            self.task = "detect"
            self.yaml = {"channels": 3}

        def fuse(self, verbose=False):  # noqa: ARG002 - signature expected by AutoBackend
            return self  # no Conv+BN folding: the evaluated body stays bit-identical to the exported artifact

        def forward(self, x, augment=False, visualize=False, embed=None, **kwargs):  # noqa: ARG002
            feats = self.body(x)
            return self.head(feats), feats

    return TraceableYOLOv8Body, R1DecodeHead, R1Detector


def fx_node_records(gm) -> list[dict]:
    recs = []
    for n in gm.graph.nodes:
        rec = {"op": n.op, "target": n.target if isinstance(n.target, str) else getattr(n.target, "__name__", str(n.target)), "module_class": None}
        if n.op == "call_module":
            sub = gm.get_submodule(n.target)
            rec["module_class"] = f"{type(sub).__module__}.{type(sub).__name__}"
        recs.append(rec)
    return recs


def preprocess_image(path: Path, imgsz: int, letterbox_cls, np, torch):
    import cv2  # noqa: PLC0415

    im0 = cv2.imread(str(path))
    im = letterbox_cls(new_shape=(imgsz, imgsz), auto=False)(image=im0)
    im = im[..., ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW (ultralytics predictor pre_transform)
    im = np.ascontiguousarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(im)[None], im0.shape[:2]


def stats_of(a, b) -> dict:
    return equivalence_stats(a.detach().float().reshape(-1).tolist(), b.detach().float().reshape(-1).tolist())


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, default=ROOT / "outputs" / "models" / "yolov8n.pt")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "models" / "yolov8n_fx_r1")
    ap.add_argument("--report-dir", type=Path, default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--data", default="coco128.yaml")
    ap.add_argument("--images", type=int, default=8, help="number of COCO128 images for the pre-NMS equivalence check")
    ap.add_argument("--rep-image", default="000000000009.jpg", help="representative image for the one-image NMS diagnostic")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--equiv-min-cosine", type=float, default=0.999999, help="PROJECT-DEFINED EXAMPLE; decided after measuring the fused/unfused noise floor")
    ap.add_argument("--equiv-max-rel-diff", type=float, default=1e-4, help="PROJECT-DEFINED EXAMPLE relative L2 tolerance")
    ap.add_argument("--skip-baseline-val", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args(argv)
    if not args.weights.is_file():
        print(f"ABORT: weights not found: {args.weights}")
        return 2

    ledger_before = hashlib.sha256(LEDGER.read_bytes()).hexdigest()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report = args.report_dir or (ROOT / "reports" / "yolov8_baseline" / f"{stamp}_r1")
    report.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    diag = args.out_dir / "diag"
    diag.mkdir(exist_ok=True)
    print(DISCLAIMER)

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import torch.fx as fx  # noqa: PLC0415
    import ultralytics  # noqa: PLC0415
    from ultralytics import YOLO, settings  # noqa: PLC0415
    from ultralytics.data.augment import LetterBox  # noqa: PLC0415
    from ultralytics.models.yolo.detect import DetectionValidator  # noqa: PLC0415
    from ultralytics.nn.modules.block import DFL, C2f  # noqa: PLC0415
    from ultralytics.utils.nms import non_max_suppression  # noqa: PLC0415
    from ultralytics.utils.tal import dist2bbox, make_anchors  # noqa: PLC0415

    settings.update({"datasets_dir": DATASETS.as_posix(), "sync": False, "runs_dir": (report / "ultralytics_runs").as_posix()})
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    body_cls, head_cls, detector_cls = make_wrapper_classes(torch)  # TraceableYOLOv8Body, R1DecodeHead, R1Detector
    evaluator = f"ultralytics {ultralytics.__version__} DetectionValidator (torch {torch.__version__})"
    input_shape = [1, 3, args.imgsz, args.imgsz]
    weights_sha = compute_sha256(args.weights)
    record: dict = {"credit_consuming": False, "netspresso_api_calls": 0, "disclaimer": DISCLAIMER, "phase": "5-E R1", "patch_id": PATCH_ID,
                    "weights": {"path": portable(args.weights.as_posix()), "sha256": weights_sha, "size_bytes": args.weights.stat().st_size},
                    "evaluator": evaluator, "input_shape": input_shape}
    val_kwargs = dict(data=args.data, imgsz=args.imgsz, batch=8, device="cpu", conf=args.conf, iou=args.iou, plots=False, verbose=False, workers=0,
                      project=(report / "ultralytics_runs").as_posix(), half=False)

    # ---- 1. upstream baseline reproduction (TC-Y8-R1-001) ----
    yolo = YOLO(args.weights.as_posix())
    ckpt_meta = {k: str(v) for k, v in (yolo.ckpt or {}).items() if k in ("version", "date", "license", "docs")}
    record["checkpoint_metadata"] = ckpt_meta
    baseline = DetectionMetrics.not_available("baseline validation skipped (--skip-baseline-val)", dataset=args.data, imgsz=args.imgsz, evaluator=evaluator, model_label="yolov8n upstream baseline")
    if not args.skip_baseline_val:
        res = YOLO(args.weights.as_posix()).val(**{**val_kwargs, "name": "baseline"})
        baseline = _metrics(res, args, evaluator, "yolov8n upstream baseline (official weights)")
        print(f"[baseline] mAP50-95={baseline.map50_95} mAP50={baseline.map50} P={baseline.precision} R={baseline.recall}")
    prev = _previous_upstream_baseline()
    record["baseline_reproduction"] = {"current": baseline.to_dict(), "previous_record": prev,
                                       "reproduced": (prev is not None and baseline.status == "MEASURED" and abs(prev["map50_95"] - baseline.map50_95) < 5e-5)}

    # ---- 2. traceability analysis: where does upstream stop being traceable? ----
    det = deepcopy(yolo.model).float().eval()
    layers = [{"index": int(m.i), "type": f"{type(m).__module__}.{type(m).__name__}", "from": m.f} for m in det.model]
    detect = det.model[-1]
    analysis: dict = {"layers": layers, "detect_from": list(detect.f), "detect_attrs": {"nc": detect.nc, "nl": detect.nl, "reg_max": detect.reg_max, "stride": [float(s) for s in detect.stride],
                                                                                         "end2end": bool(getattr(detect, "end2end", False)), "legacy": bool(getattr(detect, "legacy", False))}}
    probe = deepcopy(det)  # throwaway copy: a train-mode forward updates BatchNorm running statistics
    with torch.no_grad():
        probe.train()
        tr_out = probe(torch.zeros(*input_shape))
    analysis["upstream_detect_train_output"] = {"type": type(tr_out).__name__, "keys": sorted(tr_out) if isinstance(tr_out, dict) else None,
                                                "shapes": {k: list(v.shape) for k, v in tr_out.items() if hasattr(v, "shape")} if isinstance(tr_out, dict) else None,
                                                "note": "upstream 8.4 Detect returns a dict {boxes, scores, feats} in train mode; the fork (8.0.108) returns nl maps [N, nc+4*reg_max, H, W]"}
    for mode in ("eval", "train"):
        try:
            getattr(probe, mode)()
            g = fx.Tracer().trace(probe)
            analysis[f"direct_trace_{mode}"] = {"status": "TRACED", "nodes": len(list(g.nodes)), "note": "upstream DetectionModel traced directly in this mode"}
        except Exception as exc:  # noqa: BLE001
            analysis[f"direct_trace_{mode}"] = {"status": "FAILED", "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    del probe
    det.eval()
    analysis["reference_model_note"] = "reference DetectionModel never ran in train mode (BN statistics untouched); train-mode/trace probes used a discarded deepcopy"
    analysis["boundary"] = TRACEABILITY_BOUNDARY
    record["traceability_analysis"] = analysis
    print(f"[analysis] direct trace eval={analysis['direct_trace_eval']['status']} train={analysis['direct_trace_train']['status']} | train output={analysis['upstream_detect_train_output']['type']}")

    # ---- 3/4. wrapper + export (twice, for reproducibility) ----
    def export_once(dest: Path) -> dict:
        model = deepcopy(YOLO(args.weights.as_posix()).model).float().eval()
        body = body_cls(model, c2f_cls=C2f)
        graph = fx.Tracer().trace(body)
        gm = fx.GraphModule(body, graph)
        gm.eval()
        dest.mkdir(parents=True, exist_ok=True)
        torch.save(gm, (dest / "model_fx.pt").as_posix())
        d = model.model[-1]
        feats = body(torch.zeros(*input_shape))
        anchors, strides = (a.transpose(0, 1) for a in make_anchors(feats, d.stride, 0.5))
        meta = {"nc": int(d.nc), "nl": int(d.nl), "anchors": anchors.tolist(), "stride": [float(s) for s in d.stride], "strides": strides.tolist(), "inplace": True}
        (dest / "netspresso_head_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return {"fx_sha256": compute_sha256(dest / "model_fx.pt"), "fx_size_bytes": (dest / "model_fx.pt").stat().st_size,
                "head_meta_sha256": compute_sha256(dest / "netspresso_head_meta.json"), "graph": summarize_graph(fx_node_records(gm)), "names": model.names,
                "patched_c2f": body.patched_c2f}

    first = export_once(args.out_dir)
    tmp = args.out_dir / "_repro_tmp"
    second = export_once(tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    fx_path, meta_path = args.out_dir / "model_fx.pt", args.out_dir / "netspresso_head_meta.json"
    repro = {"export_1_sha256": first["fx_sha256"], "export_2_sha256": second["fx_sha256"], "bitwise_identical": first["fx_sha256"] == second["fx_sha256"],
             "head_meta_identical": first["head_meta_sha256"] == second["head_meta_sha256"], "status": "PASS" if first["fx_sha256"] == second["fx_sha256"] else "FAIL",
             "note": "raw SHA preserved; no metadata normalization needed for the torch.save container when raw SHAs already match"}
    manifest = build_export_manifest(
        source_model_sha256=weights_sha, source_model_version=f"ultralytics checkpoint version={ckpt_meta.get('version')} date={ckpt_meta.get('date')}",
        source_code_version=f"ultralytics {ultralytics.__version__} (upstream, unmodified)", patch_identifier=PATCH_ID, input_shape=input_shape,
        fx_version=f"torch.fx (torch {torch.__version__})", torch_version=torch.__version__, python_version=platform.python_version(),
        timestamp=datetime.now(UTC).replace(microsecond=0).isoformat(), artifact_sha256=first["fx_sha256"], artifact_size_bytes=first["fx_size_bytes"],
        artifact_path=portable(fx_path.as_posix()), head_meta_sha256=first["head_meta_sha256"], head_meta_path=portable(meta_path.as_posix()),
        export_method="torch.fx.Tracer().trace(TraceableYOLOv8Body(upstream DetectionModel)) -> GraphModule -> torch.save", graph=first["graph"], reproducibility=repro,
        patched_c2f_instances=first["patched_c2f"],
    )
    (args.out_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    record["export"] = manifest
    print(f"[export] model_fx.pt {first['fx_size_bytes']} B sha {first['fx_sha256'][:16]} | graph nodes {first['graph']['node_count']} leaf_clean={first['graph']['leaf_clean']} | repro {repro['status']}")

    # ---- 5. artifact validation + precheck (TC-Y8-R1-004/006/007) ----
    def load_body():
        gm = torch.load(fx_path.as_posix(), map_location="cpu", weights_only=False)
        return gm.eval()

    def body_forward(shape):
        outs = load_body()(torch.zeros(*shape))
        return [list(o.shape) for o in outs]

    sv = validate_pt(fx_path, allow_full_unpickle=True, torch_module=torch).to_dict()
    pre = precheck_fx_bundle(fx_path, meta_path, input_shape=input_shape, expected_nc=80, allow_full_unpickle=True, torch_module=torch, forward=body_forward)
    record["artifact_validation"] = {"structural": sv, "precheck": pre.to_dict(),
                                     "status": "PASS" if sv["status"] == "PASS" and pre.status == "READY" and sv["details"].get("object_kind") == "graph_module" else "FAIL"}
    print(f"[artifact] structural {sv['status']} kind={sv['details'].get('object_kind')} params={sv['details'].get('parameter_count')} | precheck {pre.status}")
    if record["artifact_validation"]["status"] != "PASS":
        _finish(report, record, ledger_before)
        return 3

    # ---- 6. head re-attach (TC-Y8-R1-005) ----
    head_meta = load_head_meta(meta_path)
    meta_raw = json.loads(meta_path.read_text(encoding="utf-8"))
    body = load_body()
    head = head_cls(meta_raw, DFL, make_anchors, dist2bbox).eval()
    detector = detector_cls(body, head, first["names"], meta_raw["stride"]).eval()
    y0, feats0 = detector(torch.zeros(*input_shape))
    ok_dec, msg_dec = validate_output_schema([list(y0.shape)], [expected_decoded_output_shape(input_shape, head_meta)])
    ok_body, msg_body = validate_output_schema([list(f.shape) for f in feats0], expected_body_output_shapes(input_shape, head_meta))
    dfl_same = torch.equal(head.dfl.conv.weight, detect.dfl.conv.weight)
    record["reattach"] = {"status": "PASS" if ok_dec and ok_body else "FAIL", "decoded_output_shape": list(y0.shape), "body_output_shapes": [list(f.shape) for f in feats0],
                          "messages": [msg_body, msg_dec], "dfl_weights_equal_checkpoint": bool(dfl_same), "head_class": "R1DecodeHead (rebuilt from head meta; upstream DFL/make_anchors/dist2bbox)"}
    print(f"[reattach] {record['reattach']['status']}: body {record['reattach']['body_output_shapes']} -> decoded {list(y0.shape)} | DFL weights == checkpoint: {dfl_same}")
    if not (ok_dec and ok_body):
        _finish(report, record, ledger_before)
        return 4

    # ---- 7. functional equivalence, pre-NMS (TC-Y8-R1-008) ----
    img_dir = DATASETS / "coco128" / "images" / "train2017"
    images = sorted(img_dir.glob("*.jpg"))[: args.images]
    if (img_dir / args.rep_image).is_file() and (img_dir / args.rep_image) not in images:
        images.append(img_dir / args.rep_image)
    fused = deepcopy(det).fuse(verbose=False).eval()
    per_image = []
    for p in images:
        x, _ = preprocess_image(p, args.imgsz, LetterBox, np, torch)
        y_up, preds_up = det(x)
        y_r1, feats_r1 = detector(x)
        # raw tensors: fork-contract maps -> (boxes, scores) layout of upstream forward_head
        cat = torch.cat([f.view(1, head.no, -1) for f in feats_r1], 2)
        raw_box, raw_cls = cat.split((4 * REG_MAX, head.nc), 1)
        y_fused, _ = fused(x)
        per_image.append({"image": p.name,
                          "raw_boxes": stats_of(preds_up["boxes"], raw_box), "raw_scores": stats_of(preds_up["scores"], raw_cls),
                          "decoded_all": stats_of(y_up, y_r1), "decoded_boxes": stats_of(y_up[:, :4], y_r1[:, :4]), "decoded_scores": stats_of(y_up[:, 4:], y_r1[:, 4:]),
                          "noise_floor_fused_vs_unfused_decoded": stats_of(y_up, y_fused)})
    agg = {k: {"min_cosine": min(r[k]["cosine"] for r in per_image), "max_abs_diff": max(r[k]["max_abs_diff"] for r in per_image),
               "max_relative_l2_diff": max(r[k]["relative_l2_diff"] for r in per_image), "all_bitwise_identical": all(r[k]["bitwise_identical"] for r in per_image)}
           for k in ("raw_boxes", "raw_scores", "decoded_all", "decoded_boxes", "decoded_scores", "noise_floor_fused_vs_unfused_decoded")}
    worst = {"status": "COMPUTED", "cosine": agg["decoded_all"]["min_cosine"], "max_abs_diff": agg["decoded_all"]["max_abs_diff"], "relative_l2_diff": agg["decoded_all"]["max_relative_l2_diff"]}
    verdict = equivalence_verdict(worst, min_cosine=args.equiv_min_cosine, max_relative_l2_diff=args.equiv_max_rel_diff)
    record["functional_equivalence"] = {"images": len(per_image), "per_image": per_image, "aggregate": agg, "verdict_pre_nms_decoded": verdict,
                                        "comparison": "upstream DetectionModel (unfused, eval) vs R1 fx body + R1DecodeHead, identical preprocessed input, before NMS",
                                        "note": "noise_floor = upstream unfused vs upstream fused (same model, BN folding) - the magnitude of ordinary fp32 re-association noise"}
    print(f"[equivalence] decoded min cosine {agg['decoded_all']['min_cosine']:.8f} max_abs {agg['decoded_all']['max_abs_diff']:.3e} rel {agg['decoded_all']['max_relative_l2_diff']:.3e} "
          f"| raw bitwise identical: boxes={agg['raw_boxes']['all_bitwise_identical']} scores={agg['raw_scores']['all_bitwise_identical']} | noise floor rel {agg['noise_floor_fused_vs_unfused_decoded']['max_relative_l2_diff']:.3e} -> {verdict['verdict']}")

    # ---- 8. one-image diagnostic, post-NMS (TC-Y8-R1-009) ----
    rep = img_dir / args.rep_image
    x, hw0 = preprocess_image(rep, args.imgsz, LetterBox, np, torch)
    y_up, preds_up = det(x)
    y_r1, feats_r1 = detector(x)
    one: dict = {"image": rep.name, "original_hw": list(hw0), "settings": {}}
    for label, conf in (("predictor_default", 0.25), ("validator", args.conf)):
        d_up = non_max_suppression(y_up, conf_thres=conf, iou_thres=args.iou, max_det=300)[0].tolist()
        d_r1 = non_max_suppression(y_r1, conf_thres=conf, iou_thres=args.iou, max_det=300)[0].tolist()
        cmp = compare_detections(d_up, d_r1)
        one["settings"][label] = {"conf": conf, "iou": args.iou, "comparison": cmp,
                                  "upstream_top5": [[round(v, 3) for v in b] for b in d_up[:5]], "r1_top5": [[round(v, 3) for v in b] for b in d_r1[:5]]}
        print(f"[one-image {label} conf={conf}] upstream {cmp['ref_count']} boxes vs R1 {cmp['cand_count']} | matched {cmp['matched']} | max coord diff {cmp['max_coordinate_abs_diff']} | max conf diff {cmp['max_confidence_abs_diff']}")
    one["status"] = "PASS" if all(s["comparison"]["all_matched"] for s in one["settings"].values()) else "FAIL"
    record["one_image_diagnostic"] = one

    # ---- 10. cross-env dumps (consumed by scripts/yolov8_r1_fork_crosscheck.py in the fork / torch 2.0.1 envs) ----
    dumps = {"input": x.numpy()}
    yv, out = [], x
    for m in det.model[:-1]:
        if m.f != -1:
            out = yv[m.f] if isinstance(m.f, int) else [out if j == -1 else yv[j] for j in m.f]
        out = m(out)
        yv.append(out if m.i in det.save else None)
        dumps[f"layer_{m.i:02d}"] = out.numpy()
    for i, f in enumerate(feats_r1):
        dumps[f"body_out_{i}"] = f.numpy()
    dumps["detect_boxes"], dumps["detect_scores"], dumps["decoded"] = preds_up["boxes"].numpy(), preds_up["scores"].numpy(), y_up.numpy()
    np.savez(diag / "upstream_reference.npz", **dumps)
    (diag / "upstream_reference.json").write_text(json.dumps({"image": rep.name, "layers": layers[:-1], "detect_from": list(detect.f), "input_shape": input_shape,
                                                               "weights_sha256": weights_sha, "fx_sha256": first["fx_sha256"], "keys": sorted(dumps)}, indent=2), encoding="utf-8")
    param_sig = {n: [list(p.shape), float(p.float().sum()), float(p.float().abs().sum())] for n, p in det.named_parameters()}
    (diag / "upstream_param_signature.json").write_text(json.dumps(param_sig), encoding="utf-8")
    record["cross_env_dumps"] = {"dir": portable(diag.as_posix()), "keys": len(dumps), "note": "numpy arrays (float32) - no pickle; used by the fork/torch-2.0.1 cross-check"}

    # ---- 9. detection evaluation of body + head with the upstream validator (TC-Y8-R1-010) ----
    r1_metrics = DetectionMetrics.not_available("evaluation skipped (--skip-eval)", dataset=args.data, imgsz=args.imgsz, evaluator=evaluator, model_label="R1 fx body + rebuilt head")
    if not args.skip_eval:
        try:
            # YOLO.val() forces rect=True and mode=val; replicate them so baseline and R1 share the exact dataloader settings
            validator = DetectionValidator(args={**val_kwargs, "rect": True, "mode": "val", "task": "detect", "model": "r1_fx_body_plus_head", "name": "r1"})
            validator(model=detector)
            r1_metrics = _metrics(validator.metrics, args, evaluator, "R1 upstream-traced fx body + R1DecodeHead (rebuilt from head meta)")
            print(f"[val R1] mAP50-95={r1_metrics.map50_95} mAP50={r1_metrics.map50} P={r1_metrics.precision} R={r1_metrics.recall}")
        except Exception as exc:  # noqa: BLE001
            r1_metrics = DetectionMetrics.not_available(f"validation failed: {type(exc).__name__}: {str(exc)[:300]}", dataset=args.data, imgsz=args.imgsz, evaluator=evaluator, model_label="R1 fx body + rebuilt head")
            print(f"[val R1] N/A {r1_metrics.reason}")
    comparison = compare_detection_accuracy(baseline, r1_metrics)
    record["detection_evaluation"] = {"baseline": baseline.to_dict(), "r1": r1_metrics.to_dict(), "comparison": comparison.to_dict(),
                                      "conditions": {"data": args.data, "imgsz": args.imgsz, "conf": args.conf, "iou": args.iou, "rect": True, "batch": 8, "device": "cpu", "half": False},
                                      "note": "same evaluator build, same data/split/imgsz/conf/iou/rect; R1 body is unfused while the baseline validator fuses Conv+BN"}
    print(f"[compare] {comparison.status}: baseline {comparison.baseline_metric} vs R1 {comparison.candidate_metric} drop_pp={comparison.drop_pp} rel={comparison.relative_delta_percent}")

    # ---- verdict (R1 success criteria 1-10 that this script can observe) ----
    crit = {
        "baseline_reproduced": bool(record["baseline_reproduction"]["reproduced"]),
        "traceable_export_created": manifest["artifact_sha256"] is not None and first["graph"]["leaf_clean"],
        "head_reattach": record["reattach"]["status"] == "PASS",
        "schema_pass": pre.status == "READY" and ok_dec,
        "pre_nms_equivalence": verdict["verdict"] == "EQUIVALENT",
        "one_image_detection_match": one["status"] == "PASS",
        "coco128_map_equivalent": comparison.status == "MEASURED" and comparison.drop_pp is not None and abs(comparison.drop_pp) <= 0.05,
        "artifact_validation": record["artifact_validation"]["status"] == "PASS",
        "reproducibility": repro["status"] == "PASS",
    }
    record["r1_criteria"] = crit
    record["r1_status_this_env"] = "PASS" if all(crit.values()) else "FAIL"
    record["environment"] = environment_fingerprint(packages=("ultralytics", "torch", "torchvision", "numpy", "onnx", "psutil", "opencv-python"))
    _finish(report, record, ledger_before)
    print(f"\n[R1 upstream env] {record['r1_status_this_env']} | {json.dumps(crit)}")
    return 0 if record["r1_status_this_env"] == "PASS" else 5


def _metrics(res, args, evaluator: str, label: str) -> DetectionMetrics:
    box = res.box
    return DetectionMetrics(status="MEASURED", map50_95=round(float(box.map), 5), map50=round(float(box.map50), 5), map75=round(float(box.map75), 5),
                            precision=round(float(box.mp), 5), recall=round(float(box.mr), 5), dataset=args.data, dataset_version="ultralytics coco128.zip (COCO 2017 subset)",
                            split="val(train2017 subset)", images=128, instances=929, imgsz=args.imgsz, conf=args.conf, iou=args.iou, evaluator=evaluator, model_label=label,
                            extra={"speed_ms": {k: round(float(v), 3) for k, v in (getattr(res, "speed", {}) or {}).items()}})


def _previous_upstream_baseline() -> dict | None:
    runs = sorted(p for p in (ROOT / "reports" / "yolov8_baseline").glob("2*Z") if (p / "baseline.json").is_file())
    if not runs:
        return None
    b = json.loads((runs[-1] / "baseline.json").read_text(encoding="utf-8"))
    acc = b.get("accuracy", {})
    if acc.get("status") != "MEASURED":
        return None
    return {"record": portable((runs[-1] / "baseline.json").as_posix()), "map50_95": acc["metrics"]["mAP50-95"], "ultralytics_version": b["model"]["implementation"]["ultralytics_version"]}


def _finish(out: Path, record: dict, ledger_before: str) -> None:
    assert not any(m == "netspresso" or m.startswith("netspresso.") for m in sys.modules), "SDK must never be imported here"
    record["timestamp"] = datetime.now(UTC).replace(microsecond=0).isoformat()
    record["ledger_unchanged"] = hashlib.sha256(LEDGER.read_bytes()).hexdigest() == ledger_before
    (out / "r1_result.json").write_text(portable(json.dumps(record, indent=2, ensure_ascii=False, default=str)) + "\n", encoding="utf-8")
    print(f"ledger unchanged: {record['ledger_unchanged']} | NetsPresso API calls: 0 | record: {portable((out / 'r1_result.json').as_posix())}")


if __name__ == "__main__":
    raise SystemExit(main())

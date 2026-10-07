"""True detection accuracy (labeled COCO-format evaluation) and its comparison.

This is deliberately separate from :mod:`framework.evaluation.local_ort`'s
**output equivalence proxy**. Values here come from a labeled dataset and a detection
evaluator (mAP50-95 primary; mAP50, mAP75, precision, recall secondary). The proxy
(cosine similarity on raw outputs) is never an accuracy figure.

Status semantics
----------------
MEASURED        metric values exist for this dataset/split/evaluator
N/A             not measured (reason recorded) - never turned into PASS or FAIL
NOT_COMPARABLE  two measurements exist but their conditions differ (dataset, split,
                imgsz, evaluator) so a delta would be meaningless
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from framework.pipeline.result import Metrics

PRIMARY_METRIC = "mAP50-95"
SECONDARY_METRICS = ("mAP50", "mAP75", "precision", "recall")
MEASURED, NA, NOT_COMPARABLE = "MEASURED", "N/A", "NOT_COMPARABLE"
COMPARABILITY_FIELDS = ("dataset", "split", "imgsz", "evaluator", "conf", "iou")


@dataclass
class DetectionMetrics:
    status: str = NA
    reason: str | None = None
    map50_95: float | None = None
    map50: float | None = None
    map75: float | None = None
    precision: float | None = None
    recall: float | None = None
    dataset: str | None = None
    dataset_version: str | None = None
    split: str | None = None
    images: int | None = None
    instances: int | None = None
    imgsz: int | None = None
    conf: float | None = None
    iou: float | None = None
    evaluator: str | None = None  # e.g. "ultralytics 8.0.108 DetectionValidator"
    model_label: str | None = None  # e.g. "yolov8n baseline" / "identity fixture" - never "compressed" unless it is
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def not_available(cls, reason: str, **conditions: Any) -> DetectionMetrics:
        return cls(status=NA, reason=reason, **conditions)

    @property
    def primary(self) -> float | None:
        return self.map50_95

    def conditions(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in COMPARABILITY_FIELDS}

    def to_framework_metrics(self) -> Metrics:
        """Map the PRIMARY metric onto the common Result Model (``Metrics.accuracy``); others into ``extra``."""
        extra = {}
        for name, value in (("mAP50", self.map50), ("mAP75", self.map75), ("precision", self.precision), ("recall", self.recall)):
            if value is not None:
                extra[f"detection_{name}"] = float(value)
        return Metrics(accuracy=self.map50_95, extra=extra)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "primary_metric": PRIMARY_METRIC,
            "metrics": {"mAP50-95": self.map50_95, "mAP50": self.map50, "mAP75": self.map75, "precision": self.precision, "recall": self.recall},
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "split": self.split,
            "images": self.images,
            "instances": self.instances,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "evaluator": self.evaluator,
            "model_label": self.model_label,
            "is_output_equivalence_proxy": False,
            "extra": dict(self.extra),
        }


@dataclass
class AccuracyComparison:
    status: str  # MEASURED | N/A | NOT_COMPARABLE
    metric: str = PRIMARY_METRIC
    baseline_metric: float | None = None
    candidate_metric: float | None = None
    absolute_delta: float | None = None  # candidate - baseline (metric units, 0-1)
    drop_pp: float | None = None  # (baseline - candidate) * 100, positive = worse
    relative_delta_percent: float | None = None  # (candidate - baseline) / baseline * 100
    baseline_label: str | None = None
    candidate_label: str | None = None
    reason: str | None = None
    conditions: dict[str, Any] = field(default_factory=dict)
    secondary: dict[str, dict[str, float | None]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "metric": self.metric,
            "baseline_metric": self.baseline_metric,
            "candidate_metric": self.candidate_metric,
            "absolute_delta": self.absolute_delta,
            "drop_pp": self.drop_pp,
            "relative_delta_percent": self.relative_delta_percent,
            "baseline_label": self.baseline_label,
            "candidate_label": self.candidate_label,
            "reason": self.reason,
            "conditions": dict(self.conditions),
            "secondary": dict(self.secondary),
            "is_output_equivalence_proxy": False,
        }


def compare_detection_accuracy(baseline: DetectionMetrics | None, candidate: DetectionMetrics | None) -> AccuracyComparison:
    labels = {"baseline_label": baseline.model_label if baseline else None, "candidate_label": candidate.model_label if candidate else None}
    if baseline is None or baseline.status != MEASURED or baseline.primary is None:
        return AccuracyComparison(NA, reason="baseline accuracy not measured" + (f": {baseline.reason}" if baseline and baseline.reason else ""), **labels)
    if candidate is None or candidate.status != MEASURED or candidate.primary is None:
        return AccuracyComparison(NA, reason="candidate accuracy not measured" + (f": {candidate.reason}" if candidate and candidate.reason else ""),
                                  baseline_metric=baseline.primary, **labels)
    diffs = {k: (baseline.conditions()[k], candidate.conditions()[k]) for k in COMPARABILITY_FIELDS if baseline.conditions()[k] != candidate.conditions()[k]}
    if diffs:
        return AccuracyComparison(NOT_COMPARABLE, baseline_metric=baseline.primary, candidate_metric=candidate.primary,
                                  reason="evaluation conditions differ: " + ", ".join(f"{k} {b!r} vs {c!r}" for k, (b, c) in diffs.items()),
                                  conditions={"baseline": baseline.conditions(), "candidate": candidate.conditions()}, **labels)
    b, c = float(baseline.primary), float(candidate.primary)
    secondary = {}
    for name, attr in (("mAP50", "map50"), ("mAP75", "map75"), ("precision", "precision"), ("recall", "recall")):
        bv, cv = getattr(baseline, attr), getattr(candidate, attr)
        secondary[name] = {"baseline": bv, "candidate": cv, "drop_pp": round((bv - cv) * 100.0, 4) if bv is not None and cv is not None else None}
    return AccuracyComparison(
        MEASURED,
        baseline_metric=b,
        candidate_metric=c,
        absolute_delta=round(c - b, 6),
        drop_pp=round((b - c) * 100.0, 4),
        relative_delta_percent=round((c - b) / b * 100.0, 4) if b else None,
        conditions=baseline.conditions(),
        secondary=secondary,
        **labels,
    )

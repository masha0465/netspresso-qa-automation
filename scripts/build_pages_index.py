#!/usr/bin/env python
"""Build the GitHub Pages site: copy the COMMITTED HTML/JSON reports and generate an index with REAL / MOCK labels.

    python scripts/build_pages_index.py --out _site

Rules (enforced here, not just documented):
* Only files listed in REPORTS are published. Model binaries, datasets and SDK outputs are never copied.
* Every published file is scanned for secrets, local user names and absolute Windows/Unix home paths; a hit fails the build.
* Standard library only. No netspresso import, no network, 0 credits. Reads nothing from the environment.
* The index states the evidence label of each report: REAL (actual API run or actual model/data measurement) or MOCK
  (MockAdapter synthetic values that demonstrate the QA engine).
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/masha0465/netspresso-qa-automation"
FORBIDDEN = re.compile(r"np-[A-Za-z0-9]{20,}|NETSPRESSO_API_KEY=[A-Za-z0-9]|Bearer [A-Za-z0-9]{8,}|KIMSUNAH|[A-Za-z]:[\\/]Users[\\/]|/home/[a-z]|/Users/[a-z]")
ALLOWED_SUFFIXES = {".html", ".json", ".txt"}

# slug, label, title, directory under reports/, primary html, summary function name
REPORTS = [
    ("val2017", "REAL", "COCO val2017 독립 검증 (최종 판정)", "reports/yolov8_e5_1_finetune/20261005T053121Z_val2017", "val2017_summary.html", "val2017"),
    ("e5_1", "REAL", "E5-1 실 NetsPresso 압축 + 로컬 검증", "reports/yolov8_e5_1/20261005T040009Z", "e5_1_summary.html", "e5_1"),
    ("finetune_phaseC", "REAL", "로컬 fine-tuning Phase C (continuation, global epoch 41-80)", "reports/yolov8_e5_1_finetune/20261005T051007Z_phaseC", "finetune_summary.html", "finetune"),
    ("finetune_phaseB", "REAL", "로컬 fine-tuning Phase B (전층, 40 epoch)", "reports/yolov8_e5_1_finetune/20261005T043342Z_phaseB", "finetune_summary.html", "finetune"),
    ("finetune_phaseA", "REAL", "로컬 fine-tuning Phase A (detect 분기만, 30 epoch)", "reports/yolov8_e5_1_finetune/20261005T042904Z_phaseA", "finetune_summary.html", "finetune"),
    ("local_eval_run1", "REAL", "1차 실 압축 산출물 0-Credit 로컬 평가", "reports/local_eval/20261005T014650Z_20261005T010356Z_automatic_compression", "report.html", "local_eval"),
    ("mock_pairwise", "MOCK", "Mock QA: pairwise 전략", "reports/examples/mock_pairwise", "qa_report.html", "mock"),
    ("mock_pairwise_risk", "MOCK", "Mock QA: pairwise + risk 전략", "reports/examples/mock_pairwise_risk", "qa_report.html", "mock"),
    ("mock_risk", "MOCK", "Mock QA: risk-based 전략", "reports/examples/mock_risk", "qa_report.html", "mock"),
    ("mock_full", "MOCK", "Mock QA: full 매트릭스", "reports/examples/mock_full", "qa_report.html", "mock"),
    ("mock_full_regressed", "MOCK", "Mock QA: full 매트릭스 (회귀 시나리오)", "reports/examples/mock_full_regressed", "qa_report.html", "mock"),
    ("mock_full_regression", "MOCK", "Mock 회귀 비교 리포트 (baseline vs regressed)", "reports/examples/mock_full_regression", "regression_report.html", "regression"),
]

# JSON records published next to the HTML so a reader can verify numbers (text only, no binaries)
EXTRA_JSON_LINKS = {
    "val2017": ["val2017_summary.json", "dataset_manifest.json", "validation_results.json"],
    "e5_1": ["e5_1_result.json", "e5_1_execution.json"],
    "finetune_phaseC": ["finetune_result.json", "finetune_accuracy.json"],
    "finetune_phaseB": ["finetune_result.json"],
    "finetune_phaseA": ["finetune_result.json"],
    "local_eval_run1": ["local_eval_result.json", "quality_gate.json"],
    "mock_pairwise": ["qa_report.json"],
    "mock_full_regression": ["regression_report.json"],
}


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def summarize(kind: str, src: Path) -> str:
    """One line of key numbers read from the committed JSON record (never computed here)."""
    try:
        if kind == "val2017":
            r = _load(src / "val2017_summary.json")
            rows = {x["model"]: x["mAP50-95"] for x in r["accuracy"]["val2017"]}
            return (f"mAP50-95 baseline {rows['baseline']} / E5-1 {rows['e5_1']} / Phase C {rows['phase_c']} → {r['decision']}; "
                    f"Gate {r['quality_gate']['release']['overall']} (release); NetsPresso {r['netspresso_additional_execution']}")
        if kind == "e5_1":
            r = _load(src / "e5_1_result.json")
            p = r["model_metrics"]["params"]
            c = r["netspresso_execution"]["credit"]
            return (f"params {p['baseline_sdk']:,} → {p['candidate_sdk']:,} ({p['reduction_percent_sdk']} %); COCO128 mAP50-95 "
                    f"{r['accuracy']['baseline']['metrics']['mAP50-95']} → {r['accuracy']['candidate']['metrics']['mAP50-95']}; 계정 {c['account_total_before']} → {c['account_total_after']}; "
                    f"Gate {r['quality_gate']['compression']['overall']} / {r['quality_gate']['local_eval']['overall']} / {r['quality_gate']['release']['overall']}")
        if kind == "finetune":
            r = _load(src / "finetune_result.json")
            t = r["accuracy"]["three_way"]
            # the recovery decision is a COCO128 smoke classification (train == val); it is not a model-quality verdict
            line = (f"COCO128 smoke(train == val 128장) mAP50-95 {t['compressed']} → {t['fine_tuned']} (baseline {t['baseline']}); "
                    f"smoke 분류 {r['recovery']['decision']}, 모델 품질 회복 판정이 아님")
            if r["finetune_config"]["phase"] == "C":
                v = _load(ROOT / "reports/yolov8_e5_1_finetune/20261005T053121Z_val2017/val2017_summary.json")
                unseen = next(x["mAP50-95"] for x in v["accuracy"]["val2017"] if x["model"] == "phase_c")
                line += f"; 독립 val2017(unseen) mAP50-95 {unseen} → {v['decision']}"
            return line
        if kind == "local_eval":
            r = _load(src / "local_eval_result.json")
            g = r["quality_gate"]
            return (f"latency median {r['performance']['latency']['baseline']['median_ms']} → {r['performance']['latency']['optimized']['median_ms']} ms; "
                    f"proxy min cosine {r['output_equivalence_proxy']['min_cosine_similarity']}; Gate {g['compression']['overall']} / {g['local_eval']['overall']} / {g['release']['overall']}")
        if kind == "mock":
            r = _load(src / "qa_report.json")
            if r is None:
                return "MockAdapter 합성값 데모 (JSON은 크기 때문에 미추적, HTML만 공개)"
            s = r["summary"]
            return (f"전략 {r['strategy']}: 선택 {s['selected_combinations']} / 실행 {s['tested_combinations']} · PASS {s['PASS']} FAIL {s['FAIL']} BLOCKED {s['BLOCKED']} "
                    f"NOT_TESTED {s['NOT_TESTED']} · pair coverage {r['selection'].get('pair_coverage_percent', 'n/a')} %")
        if kind == "regression":
            r = _load(src / "regression_report.json")
            s = r["summary"]
            return f"overall {r['overall']}: PASS {s['PASS']} / REGRESSION {s['REGRESSION']} / NOT_COMPARABLE {s['NOT_COMPARABLE']}"
    except (KeyError, TypeError, ValueError) as exc:  # record schema drift must be visible, not hidden
        return f"(summary unavailable: {type(exc).__name__})"
    return ""


def scan(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return sorted({m.group(0) for m in FORBIDDEN.finditer(text)})


def build(out: Path) -> int:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rows, problems, published = [], [], 0
    for slug, label, title, rel_dir, primary, kind in REPORTS:
        src = ROOT / rel_dir
        if not (src / primary).is_file():
            problems.append(f"missing primary report: {rel_dir}/{primary}")
            continue
        dest = out / slug
        dest.mkdir()
        files = [primary] + [j for j in EXTRA_JSON_LINKS.get(slug, []) if (src / j).is_file()]
        for name in files:
            if Path(name).suffix not in ALLOWED_SUFFIXES:
                problems.append(f"refused non-text file: {rel_dir}/{name}")
                continue
            hits = scan(src / name)
            if hits:
                problems.append(f"forbidden pattern in {rel_dir}/{name}: {hits}")
                continue
            shutil.copyfile(src / name, dest / name)
            published += 1
        links = " · ".join(f'<a href="{slug}/{html.escape(n)}">{html.escape(n)}</a>' for n in files)
        rows.append(f"<tr><td><span class=\"label {label}\">{label}</span></td><td>{html.escape(title)}</td><td>{html.escape(summarize(kind, src))}</td>"
                    f"<td>{links}</td><td><a href=\"{REPO_URL}/tree/main/{rel_dir}\">{html.escape(rel_dir)}</a></td></tr>")
    if problems:
        for p in problems:
            print("ERROR:", p, file=sys.stderr)
        return 1
    index = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"><title>NetsPresso QA Automation — reports</title>
<style>
body{{font-family:-apple-system,"Segoe UI",Roboto,"Noto Sans KR",sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#1f2933;line-height:1.45}}
table{{border-collapse:collapse;width:100%;font-size:.9rem}} th,td{{border:1px solid #d9dee3;padding:.4rem .5rem;vertical-align:top;text-align:left}} th{{background:#f5f7fa}}
.label{{font-weight:600;padding:.1rem .45rem;border-radius:3px}} .REAL{{background:#e3f6ec;color:#0f5132}} .MOCK{{background:#fff1cc;color:#6b4e00}}
.note{{background:#fff8e6;border-left:3px solid #d4a72c;padding:.5rem .75rem;margin:1rem 0;font-size:.9rem}} code{{background:#f5f7fa;padding:0 .25rem;border-radius:3px}}
</style></head><body>
<h1>NetsPresso QA Automation Framework — 리포트</h1>
<p>저장소: <a href="{REPO_URL}">{REPO_URL}</a> · 이 페이지는 저장소에 커밋된 HTML/JSON 리포트를 그대로 게시합니다. 모델 바이너리와 데이터셋은 포함하지 않습니다.</p>
<div class="note"><strong>REAL</strong> = 실제 NetsPresso API 실행 또는 실제 모델·데이터로 로컬에서 측정한 값. <strong>MOCK</strong> = MockAdapter의 결정적 합성값으로 QA 엔진 동작을 보이는 데모(실제 모델 측정이 아님).
모든 임계값은 PROJECT-DEFINED EXAMPLE이며 Nota 공식 기준이 아닙니다. 본 사이트는 공개 NetsPresso SDK만 사용한 개인 포트폴리오 프로젝트의 산출물입니다.</div>
<table><tr><th>라벨</th><th>리포트</th><th>핵심 수치 (JSON 레코드에서 읽음)</th><th>파일</th><th>원본 위치</th></tr>
{''.join(rows)}
</table>
<p>최종 판정: <strong>VAL2017_OVERFIT · Release FAIL · NetsPresso 추가 실행 HOLD</strong> (상세는 첫 행의 val2017 리포트와 README §10).</p>
</body></html>
"""
    (out / "index.html").write_text(index, encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    print(f"published {published} files in {len(rows)} report groups -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "_site")
    args = ap.parse_args(argv)
    return build(args.out)


if __name__ == "__main__":
    raise SystemExit(main())

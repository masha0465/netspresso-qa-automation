# Portfolio Summary — NetsPresso QA Automation Framework

> Personal portfolio project using publicly available NetsPresso APIs/SDKs and technical resources. Not affiliated with or endorsed by Nota Inc.
> 수치 라벨: **REAL** = 실제 실행/실측, **MOCK** = MockAdapter 합성값, **DESIGN** = 설계/정책.

## Project
NetsPresso QA Automation Framework — AI 모델 최적화 파이프라인(압축 → 산출물 → accuracy → performance → reproducibility → release)에 대한 E2E Quality Gate. 저장소: `netspresso-qa-automation`.

## Role
QA Engineer / QA Automation (1인 설계·구현·실험·판단)

## Period
2026 (Phase 0 → 6)

## Objective
AI Model Optimization Pipeline의 E2E Quality Gate 구축 — "최적화가 완료되었다"가 아니라 "릴리스해도 되는가"에 코드와 실측 증거로 답하는 체계.

## Key Contributions
- QA automation framework architecture: Adapter 추상화(Mock / NetsPresso), 공통 Result Model(`ExecutionResult` → `CaseResult`), 설정 주도(YAML), 최소 의존성, Python 3.14 프레임워크와 3.11 SDK/평가 환경의 분리(JSON 통신).
- NetsPresso adapter: DRY_RUN 기본, 명시 승인(`mode: real` + `--confirm-credit-use`), API 키는 환경변수만(값 비노출), 실 작업 1종(`automatic_compression`) 구현, 서버 응답 → Result Model 매핑, 재시도 없음.
- Mock/Real execution boundary: CI·pytest는 `netspresso`를 import할 수 없음(assert), 실 호출은 별도 스크립트에서만.
- Model × Device × Runtime × Backend × Optimization matrix: 324 조합(126 feasible + 198 UNSUPPORTED, 근거 있을 때만) [DESIGN+MOCK].
- Pairwise / risk-based selection: 결정적 greedy pairwise 15 케이스·feasible pair coverage 100 % 실측 보고, 가중치 기반 리스크 선택과 선택 사유 기록 [MOCK 데모].
- Artifact validation: 존재·SHA-256·추적성 + 구조(fx GraphModule·ONNX)·forward·shape, 신뢰 SHA 목록 기반 안전 로딩, normalized ONNX checksum [REAL].
- Task-level accuracy validation: ultralytics 평가기로 mAP50-95/mAP50/mAP75/P/R, 동일 조건 강제(evaluator·data·imgsz·conf/IoU·rect), 조건 불일치는 NOT_COMPARABLE [REAL].
- Performance validation: 프로세스 격리 ONNX Runtime latency median/p95·peak RSS(개발 머신, 타깃 디바이스 아님) [REAL].
- Reproducibility tracking: bitwise / functional / not / NOT_VERIFIED 4단계, baseline registry(candidate → baseline → retired, 승격 정책) [REAL: NOT_VERIFIED 유지].
- Defect classification: 13 카테고리 + UNCLASSIFIED, Gate 결과·명시적 error kind만 증거, 의심 원인은 "suspected"로만.
- Quality Gate: compression / local_eval / release 프로파일, 결측 ≠ PASS, 실행 성공 ≠ PASS [DESIGN thresholds].
- Credit-safe real API execution: exactly-one guard(입력 SHA·precheck·장부 prior-ops·승인), 계정 잔액 before/after 관측 장부, 0-Credit 스크립트의 키 존재 시 abort.
- Independent validation: 학습에 쓰지 않은 COCO val2017(5,000장, 오염 검사 0건)로 일반화 검증 → Release 차단.

## Real Validation [REAL]
- E5-1: R1 YOLOv8n fx body(upstream 8.4.173 기반 traceable export, upstream과 bitwise 동일 확인)에 대해 NetsPresso `automatic_compression`(ratio 0.5, PR_L2) **1회**, 25 Credit. params 3,157,184 → 882,996(−72 %), FLOPs 8.80 → 3.87 G(−56 %), 구조 PASS, 로컬 지연 −30 %, **COCO128 mAP50-95 0.44369 → 0.0**.
- 그 이전 1회(공식 샘플 `graphmodule.pt`, 25 Credit)는 어댑터 E2E 경로 검증.
- Total: **50 Credits / 2 real operations** (잔여 450). 이후 모든 작업(로컬 검증·fine-tuning·val2017)은 0 Credit.

## Key Result [REAL]
COCO128(train == val, smoke)에서는 로컬 fine-tuning(Phase A → B → C, CPU, 0 Credit) 후 mAP50-95 **0.46688**까지 회복(baseline 0.44369 초과).

하지만 independent **COCO val2017**(unseen, 5,000장)에서는 **0.00917**(baseline 0.36804, drop 35.9 pp)로 붕괴. COCO128 회복은 128장 암기였다.

따라서:
- **VAL2017_OVERFIT**
- **Release FAIL** (accuracy, checksum 미검증, reproducibility NOT_VERIFIED)
- **NetsPresso additional execution HOLD**

| Stage | COCO128 (smoke) | COCO val2017 (unseen) |
|---|---:|---:|
| Baseline YOLOv8n | 0.44369 | 0.36804 |
| E5-1 compressed | 0.0 | 0.0 |
| Phase C fine-tuned | 0.46688 | 0.00917 |

## QA Insight
"Smoke PASS does not imply Release PASS."

"Output equivalence proxy does not replace task-level accuracy." (E5-1: cosine 0.984 / mAP 0.0 · Phase C: cosine 0.990 / val2017 mAP 0.009)

부가 발견: SDK `completed` ≠ 품질, FLOPs 감소 ≠ 지연 개선(모델별 상이), 측정 프로세스 격리 없이는 memory 비교가 뒤집힘, archived 공식 포크 코드의 의미 차이(C2f 비연쇄)를 레이어별 diff로 확정.

## Engineering Outcome
Release decision을 자동화 가능한 Quality Gate(compression / local_eval / release)로 연결하고, 독립 validation에서 generalization failure를 탐지하여 release를 차단하는 검증 체계를 구현했다. 전 체인은 SHA-256과 JSON 레코드로 추적 가능하며(모델 바이너리 미추적), 198개 오프라인 테스트와 CI가 엔진·장부·시크릿 부재를 강제한다.

## Limitations (요약)
실 NetsPresso 실행 2회(압축만), 실측 모델 1종(YOLOv8n), 타깃 디바이스 미측정, 재현성 NOT_VERIFIED, fine-tuning은 COCO128로만 수행(데이터 전략이 복구 실패의 원인 후보이나 미검증), 임계값은 project-defined example.

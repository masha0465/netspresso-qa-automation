# NetsPresso QA Automation Framework

**AI 모델 최적화 파이프라인을 위한 QA Automation Framework — Optimization → Artifact → Accuracy → Performance → Reproducibility → Release Gate**

> **This is a personal QA automation portfolio project using publicly available NetsPresso APIs/SDKs and technical resources.**
> 본 프로젝트는 공개된 NetsPresso SDK/API 및 기술 자료만을 사용한 **개인 포트폴리오 프로젝트**입니다. Nota의 내부 QA 프레임워크·내부 시스템·내부 소스 코드·고객 환경·비공개 모델과는 어떤 관련도 없으며, Nota의 내부 QA 프로세스를 대변하지 않습니다.

[![qa](https://img.shields.io/badge/CI-GitHub_Actions-informational)](.github/workflows/qa.yml)
![python](https://img.shields.io/badge/python-3.14_(framework)_|_3.11_(SDK_env)-blue)
![credits](https://img.shields.io/badge/NetsPresso_Credits_used-50_/_500_(2_real_ops)-success)
![release](https://img.shields.io/badge/Final_Release_Decision-FAIL_(VAL2017__OVERFIT)-critical)

표기 규칙 — 이 문서의 모든 수치·판정에는 출처 라벨을 붙입니다.
**REAL** = 실제 NetsPresso API 실행 또는 실제 모델·데이터로 로컬에서 측정한 값 · **MOCK** = MockAdapter의 결정적 합성값(엔진 검증용) · **DESIGN** = 설계·정책(임계값 등, 실측 아님) · **NOT_VERIFIED** = 확인하지 않은 것.

---

## 목차

1. [Project Overview](#1-project-overview)
2. [QA Problem](#2-qa-problem)
3. [Architecture](#3-architecture)
4. [Test Strategy](#4-test-strategy)
5. [Quality Gate](#5-quality-gate)
6. [Real NetsPresso Validation](#6-real-netspresso-validation)
7. [E5-1 Result](#7-e5-1-result)
8. [Fine-tuning Recovery](#8-fine-tuning-recovery)
9. [Independent Validation](#9-independent-validation)
10. [Final Release Decision](#10-final-release-decision)
11. [Key QA Findings](#11-key-qa-findings)
12. [Automation & CI](#12-automation--ci)
13. [Reproducibility / Traceability](#13-reproducibility--traceability)
14. [Credit Safety](#14-credit-safety)
15. [Limitations](#15-limitations)
16. [Future Work](#16-future-work)
17. [Quick Start / Repository Map](#17-quick-start--repository-map)

---

## 1. Project Overview

AI 모델 최적화 플랫폼(압축·양자화·변환·프로파일링)에서 "최적화가 완료되었다"는 사실은 릴리스 근거가 되지 못합니다. 최적화 후에도 **산출물이 구조적으로 유효한지, task-level accuracy가 허용 범위 안에 있는지, 지연·메모리가 실제로 줄었는지, 같은 조건에서 같은 결과가 재현되는지**를 정량적으로 확인하고, 그 판단이 Model × Device × Runtime × Backend × Optimization 조합에서 일관되어야 합니다.

이 저장소는 그 문제를 다루는 Python QA 자동화 프레임워크이며, NetsPresso 공개 SDK를 **참조 대상**으로 사용해 다음 흐름을 코드·설정·실측 증거로 끝까지 연결했습니다.

```
테스트 전략 설계 → 자동화 → 결과 정량화 → Quality Gate → 독립 Validation → Release 판단
```

| 구분 | 내용 | 라벨 |
|---|---|---|
| 프레임워크 | Python 3.14, pytest, PyYAML, Jinja2 (최소 의존성), 198 tests, coverage 93 % | REAL |
| 실 SDK 환경 | Python 3.11 별도 venv, netspresso 1.17.0 — 실 작업 **2회**(automatic_compression), Credit **50 / 500** 사용 | REAL |
| 평가 환경 | Python 3.11 + upstream ultralytics 8.4.173 + torch 2.14 (CPU) — COCO128 / COCO val2017 평가, 로컬 fine-tuning | REAL |
| 최종 결론 | 압축 후보는 로컬 fine-tuning으로 **COCO128(train == val) smoke 0.46688**까지 회복했지만 **독립 COCO val2017에서 0.00917**로 붕괴 → **VAL2017_OVERFIT, Release FAIL, NetsPresso 추가 실행 HOLD** | REAL |

이 프로젝트의 가장 중요한 QA 판단은 "COCO128에서 baseline을 넘은 모델을 Release하지 않고, 독립 val2017에서 일반화 여부를 검증해 Release를 차단한 것"입니다.

---

## 2. QA Problem

일반 소프트웨어 QA와 달리 모델 최적화 QA에는 다음 특성이 있고, 프레임워크의 구성 요소는 각 특성에 하나씩 대응합니다.

| 모델 최적화 QA의 특성 | 왜 문제인가 | 대응 |
|---|---|---|
| 결과가 이진이 아니라 **연속값**(accuracy, latency, memory) | "조금 나빠짐"의 허용 기준이 필요 | 임계값 기반 **Quality Gate** 프로파일 (`configs/quality_gate.yaml`) |
| 조합 폭발(5차원) | 전수 실행은 비용·시간상 불가 | **Pairwise + Risk-based** 선택, `NOT_TESTED` vs `UNSUPPORTED` 분리 |
| 지원 여부가 조합마다 다르고 문서로 확정되지 않음 | 미지원을 실패로, 미검증을 성공으로 오판 | 5단계 상태 모델, 근거 있을 때만 `UNSUPPORTED` |
| 산출물이 곧 배포물 | 파일 존재 ≠ 무결·유효 | **Artifact / Structural Validation**(SHA-256, GraphModule/ONNX 구조, forward·shape) |
| **실행 성공 ≠ 품질** | SDK `completed`가 accuracy를 보장하지 않음 | Execution / Compression / Local-eval / Release를 **다른 질문**으로 분리 |
| **smoke 결과 ≠ 일반화** | train == val 데이터의 회복은 암기일 수 있음 | **독립 validation**(COCO val2017) 없이는 Release 판단 금지 |
| proxy ≠ accuracy | 출력 유사도(cosine)는 검출 품질을 대변하지 않음 | output-equivalence **proxy**를 별도 기준으로 두고 accuracy와 혼용 금지 |
| 비결정성 | 같은 설정, 다른 결과 → 회귀 판단 불가 | Reproducibility 4단계(bitwise / functional / not / **not verified**) |
| 클라우드 최적화는 **유료(Credit)** | 디버깅 중 반복 호출 = 예산 소진 | Mock-first, 인터프리터 분리, `--confirm-credit-use`, exactly-one guard, Credit Ledger |

---

## 3. Architecture

```
                 QA Automation Framework (Python 3.14, MockAdapter·CI)   /   SDK & eval envs (Python 3.11)
                                              |
        +-- Adapter Layer ---------------------+-------------------------------------------+
        |     +-- MockAdapter        결정적 합성값, 0 Credit, pytest·CI 전용                [MOCK]
        |     +-- NetsPressoAdapter  DRY_RUN 기본 / REAL_RUN은 명시 승인 + 키 + 1회 가드     [REAL 2회]
        |
        +-- Matrix Engine
        |     +-- Cartesian  (324 조합 = 126 feasible + 198 UNSUPPORTED)                   [DESIGN+MOCK]
        |     +-- Pairwise   (15 cases, feasible pair coverage 100 %)                       [MOCK]
        |     +-- Risk-based (가중치 → HIGH/MEDIUM/LOW, 선택 사유 기록)                     [MOCK]
        |
        +-- Pipeline  Optimize → Validate → Report   (ExecutionResult → CaseResult)
        |
        +-- Validation
        |     +-- Artifact        존재·SHA-256·추적성 / 구조(GraphModule·ONNX)·forward·shape  [REAL]
        |     +-- Accuracy        task-level mAP50-95 (COCO128 smoke / COCO val2017 unseen)    [REAL]
        |     +-- Performance     process-isolated ONNX Runtime latency·peak RSS (개발 머신)   [REAL]
        |     +-- Reproducibility bitwise / functional / not / NOT_VERIFIED                   [REAL: NOT_VERIFIED]
        |     +-- Proxy           output equivalence cosine — accuracy가 아님                  [REAL]
        |
        +-- Quality Gate   profiles: compression / local_eval / release  (결측 ≠ PASS)       [DESIGN thresholds]
        +-- Defect Classification   13 categories + UNCLASSIFIED, 증거 기반, suspected_cause는 "의심"
        +-- Regression   baseline run vs current run, 임계값 비교
        +-- Reporting    JSON(기계) + HTML(사람) — mock / real / local-eval / E5-1 / fine-tune / val2017
        +-- Credit Ledger   simulated vs estimated vs actual, 계정 잔액 before/after 관측
```

설계 원칙: (1) **Adapter 추상화** — QA 엔진은 `BaseAdapter` 계약만 알고 벤더 SDK 내부를 모른다. (2) **실행 사실과 QA 판정의 분리** — `ExecutionResult`(무슨 일이 있었나) → `CaseResult`(그래서 PASS인가). (3) **설정 주도** — 매트릭스·임계값·리스크·시나리오가 모두 YAML. (4) **최소 의존성** — 메인 프레임워크는 PyYAML + Jinja2만; torch/ultralytics/netspresso는 별도 3.11 환경에만 존재하며 JSON으로만 통신한다.

### Python 환경 분리 [REAL]

| 환경 | 역할 |
|---|---|
| `.venv` (3.14) | 프레임워크, MockAdapter, pytest, CI. `netspresso`·torch를 import하지 않음(테스트·CI assert) |
| `.venv-netspresso` (3.11) | netspresso 1.17.0 (torch 2.0.1 고정). 실 API 실행 전용 |
| `.venv-yolo` (3.11) | upstream ultralytics 8.4.173 + torch 2.14 + onnxruntime. 평가·fine-tuning·val2017 |
| `.venv-yolo-nota` (3.11) | archived Nota 포크 `ultralytics_nota` 8.0.108 (비교·원인 분석용) |

netspresso 1.17.0 → netspresso-trainer 1.3.1 → `torch<=2.0.1`(Python 3.11까지만 wheel 제공)이라는 의존성 제약 때문에 Python 3.12+에서는 SDK 설치가 실패합니다. ENVIRONMENT/CONFIGURATION 이슈(Medium)로 분류했습니다([`docs/netspresso_sdk_research.md`](docs/netspresso_sdk_research.md)).

---

## 4. Test Strategy

### Matrix [DESIGN + MOCK]

`Model × Device × Runtime × Backend × Optimization`를 YAML로 선언하고 전체 조합을 생성합니다 — 현재 설정 **324 조합 = 126 feasible + 198 UNSUPPORTED**(구조적/문서 근거). 상태 모델 `PASS / FAIL / BLOCKED / NOT_TESTED / UNSUPPORTED`는 서로 다른 의미를 절대 합치지 않으며, `UNSUPPORTED`는 출처가 인용된 규칙이 있을 때만 부여합니다.

### Pairwise [MOCK]

126 feasible 조합 대신 모든 차원 쌍이 최소 1회 등장하도록 결정적 greedy set-cover로 **15 케이스**를 선택(축소율 88.1 %), feasible 92쌍 **커버리지 100 %**를 프레임워크가 **계산해서** 보고합니다. infeasible 10쌍은 별도 목록으로 남기고, 선택되지 않은 111 셀은 `NOT_TESTED`로 정직하게 남습니다. 최소성(최적해)은 주장하지 않습니다.

### Risk-based [MOCK]

`configs/risk.yaml` 가중치(new_model/new_device/previous_regression/known_compatibility_issue 3, customer_facing·new_runtime·new_backend·new_method 2, lossy_precision 1)로 점수·등급을 만들고, 케이스마다 `"model.previous_regression (+3)"` 같은 선택 사유를 리포트에 남깁니다. 전략: `full`, `pairwise`, `risk`, `pairwise_risk`.

| 전략 (MOCK 데모) | 선택 | 실행 | PASS / FAIL / BLOCKED | NOT_TESTED |
|---|---|---|---|---|
| full | 126 | 123 | 112 / 8 / 3 | 0 |
| pairwise | 15 | 14 | 9 / 4 / 1 | 111 |
| risk (HIGH) | 34 | 34 | 27 / 4 / 3 | 92 |
| pairwise_risk | 43 | 42 | 34 / 5 / 3 | 83 |

위 수치는 MockAdapter 합성값으로 **엔진 동작을 보여주는 데모**이며 실제 모델의 측정이 아닙니다. 같은 회귀 세트를 `full`로 비교하면 3건, `pairwise`로 비교하면 1건만 검출되는 트레이드오프도 데모에서 수치로 드러납니다.

### Release Test Plan [REAL]

실제 모델(YOLOv8n)에 대해서는 TC-Y8-001 … 061(모델·데이터·baseline·최적화·산출물·재현성·릴리스) + CG-/R1-/E5-/FT-/VAL- 시리즈로 테스트 케이스를 정의하고 상태(DONE/PASS/FAIL/PENDING/CREDIT)를 추적했습니다 — [`docs/phase5e_release_test_plan.md`](docs/phase5e_release_test_plan.md). 전체 전략은 [`docs/test_strategy.md`](docs/test_strategy.md).

---

## 5. Quality Gate

[DESIGN] 임계값은 모두 **PROJECT-DEFINED EXAMPLE**이며 Nota 공식 기준이 아닙니다. 세 프로파일은 서로 다른 질문에 답하며, 하위 프로파일 PASS는 상위 프로파일 PASS를 함의하지 않습니다.

| Profile | 질문 | 필수 기준(예) |
|---|---|---|
| `compression` | 최적화 산출물이 쓸 수 있는 파일인가 | model_size 증가 0 %, artifact 존재(checksum optional), structural validity |
| `local_eval` | 로컬에서 기능적으로 동작하는가 | output-equivalence proxy ≥ 0.99, latency ≤ +10 %, memory ≤ +15 %, size ≤ 0 % (accuracy는 optional) |
| `release` | 릴리스 가능한가 | **accuracy drop ≤ 1.0 pp**, latency, memory, artifact checksum(promoted baseline 대비), **reproducibility**(2회 이상 증거) |

규칙: `required` 기준이 하나라도 FAIL → Overall FAIL, 사유 전부 기록. **필수 기준의 데이터가 없으면(NOT_APPLICABLE) FAIL** — 결측 위에서 PASS를 선언하지 않습니다. 실행이 COMPLETED가 아니면 Gate를 평가하지 않습니다. 결함 분류는 Gate 결과와 어댑터의 명시적 `error.kind`만 증거로 사용하고, 근거가 없으면 `UNCLASSIFIED`입니다([`docs/release_quality_gate.md`](docs/release_quality_gate.md)).

---

## 6. Real NetsPresso Validation

[REAL] 실 API 실행은 일반 테스트·CI와 완전히 분리되어 있고, 두 번만 수행했습니다.

| 원칙 | 구현 |
|---|---|
| dry-run default | `configs/netspresso.yaml`: `mode: dry_run`, `confirm_credit_use: false`. DRY_RUN은 SDK import 없이 실행 계획만 생성 |
| explicit real-run authorization | `mode: real` + `--confirm-credit-use` 둘 다 필요(`REAL_RUN_AUTHORIZED`), 미승인은 SDK 접근 전 거부 |
| API key | 환경변수 `NETSPRESSO_API_KEY`(.env, gitignored)만; 값은 저장·로그·렌더링하지 않음(테스트) |
| CI | `netspresso` 미설치 assert, 키 미참조, 네트워크 없음 |
| exactly-one guard | 입력 SHA 정확 일치 + precheck READY + 장부 prior-ops 수 일치 + 승인 없이는 호출 불가(`framework/evaluation/input_gate.py`), 재시도 없음 |
| ledger | `reports/credit_usage.json` — 계정 잔액 before/after를 SDK 조회로 관측, 클라이언트 상수(25)와 비교 |
| traceability | 입력 SHA → 실행 레코드 → 산출물 SHA → 로컬 검증 → Gate → 장부 entry 체인 |

| # | 날짜 | 작업 | 입력 | 결과 | Credit(관측) |
|---|---|---|---|---|---|
| 1 | 2026-10-05 | `automatic_compression` ratio 0.5 | 공식 샘플 `graphmodule.pt` (224×224) | completed, 28.49 → 7.46 MB; accuracy N/A(라벨 없음) | 500 → 475 (25) |
| 2 | 2026-10-05 | `automatic_compression` ratio 0.5 (**E5-1**) | R1 YOLOv8n fx body `d8e761da…` (640×640) | completed, PR_L2 | 475 → 450 (25) |

1회차는 E2E 어댑터 경로 검증(SDK `completed` ≠ Release PASS: 릴리스 질문 6개 중 크기 1개만 답함)이었고, 0-Credit 로컬 검증(구조·params/FLOPs·ORT·proxy 0.825)을 통해 재학습 없는 pruning 산출물이 기능적으로 동등하지 않음을 확인했습니다([`docs/quality_gap_analysis.md`](docs/quality_gap_analysis.md)).

---

## 7. E5-1 Result

[REAL] 입력 준비(0 Credit): upstream ultralytics 8.4.173을 수정하지 않고 2단계 traceability wrapper(`r1-upstream-body-wrapper-v1`)로 fx body를 export해 upstream 모델과 **pre-NMS bitwise 동일**, COCO128 mAP50-95 0.44369 = 0.44369임을 확인한 뒤에만 압축 입력으로 인정했습니다. 그 과정에서 archived Nota 포크(`ultralytics_nota` 8.0.108)의 `C2f.forward`가 Bottleneck을 연쇄하지 않아 n ≥ 2 블록에서 upstream과 달라지는 문제(포크 평가 0.135 vs 0.444)를 레이어별 diff로 **원인 확정**했습니다([`docs/phase5e_conditional_go_resolution.md`](docs/phase5e_conditional_go_resolution.md)).

| 항목 | E5-1 (NetsPresso 생성 후보, sha `21d8cbd1…`) |
|---|---|
| params | 3,157,184 → **882,996** (−72.0 %; SDK 보고 = 독립 torch 계산) |
| FLOPs | 8.80 G → **3.87 G** (−56.0 %; 로컬 ONNX 2×MACs 3.83 G와 일치) |
| 모델 크기 | 12.25 MB → 3.52 MB (SDK), 파일 12,846,691 → 3,693,813 B |
| 구조/forward/head 재부착 | GraphModule PASS, `[1,144,80/40/20]` → `[1,84,8400]` PASS |
| 로컬 ORT (개발 머신) | latency median 23.8 → 16.5 ms, peak RSS 107 → 93 MB |
| **COCO128 mAP50-95** | baseline **0.44369** → E5-1 **0.0** (drop 44.37 pp) |
| output proxy | cosine 0.984 — accuracy 0.0과 함께 "proxy ≠ accuracy"의 실측 사례 |
| Quality Gate | compression **PASS** / local_eval FAIL / release **FAIL** → `ACCURACY_REGRESSION` [HIGH] |

**압축 성공 ≠ Release 성공.** 상세: [`docs/phase5e_e5_1_experiment.md`](docs/phase5e_e5_1_experiment.md).

---

## 8. Fine-tuning Recovery

[REAL, 0 Credit] 추가 Credit 없이 압축 후보(body + R1 decode head)를 로컬 CPU에서 fine-tuning했습니다(upstream 데이터로더 + `v8DetectionLoss`, AdamW 1e-3 cosine, seed 0, batch 16, imgsz 640). 평가 조건은 전 단계와 동일(ultralytics 8.4.173, COCO128, conf 0.001, IoU 0.7, rect=True, batch 8).

| Phase | 학습 범위 | epochs | COCO128 mAP50-95 | 판정(smoke) |
|---|---|---|---|---|
| E5-1 | — | — | 0.0 | — |
| A | detect 분기(cv2/cv3)만, backbone BN 고정 | 30 | 0.0348 | RECOVERY_FAILED |
| B | 전층 | 40 | 0.28965 | PARTIAL_RECOVERY (65 %) |
| C | 전층 continuation (global 41–80) | 40 | **0.46688** | smoke 임계값 band 도달 → 독립 검증 필요 |

Phase C는 COCO128에서 baseline(0.44369)을 **넘었지만**, COCO128은 train == val(같은 128장)이므로 이 값은 일반화 증거가 아닙니다. 그래서 Release 판단 대신 **독립 validation**으로 넘겼습니다. 아키텍처는 불변(882,996 params / 3.83 G)이었고 E5-1 원본 산출물은 수정하지 않았습니다. 상세: [`docs/phase5e_finetune_recovery.md`](docs/phase5e_finetune_recovery.md) §1–§12.

---

## 9. Independent Validation

[REAL, 0 Credit, 학습 없음] 학습에 쓰지 않은 **COCO val2017(5,000장, 36,781 annotations)**로 세 모델을 독립 프로세스에서 평가했습니다. 오염 검사: COCO128 학습 128장 vs val2017 — 파일명 겹침 **0**, COCO image id 겹침 **0**, 내용 SHA-256 겹침 **0**.

| Model | 종류 | COCO128 (train==val) | **COCO val2017 (unseen)** | gap (pp) |
|---|---|---|---|---|
| Baseline (R1 YOLOv8n) | upstream, 불변 | 0.44369 | **0.36804** | 7.6 |
| E5-1 | NetsPresso 생성 후보 | 0.0 | **0.0** | 0.0 |
| Phase C | 로컬 fine-tuned 파생물 | **0.46688** | **0.00917** | **45.8** |

baseline 0.368은 ultralytics 공개 YOLOv8n val2017 37.3과 0.5 pp 이내 → 평가 파이프라인 정상. Phase C의 gap은 baseline 자체 gap보다 38.2 pp 크고 unseen 회복률은 2.5 %에 그침 → COCO128 성능 회복은 **암기(overfitting)**였습니다. 상세: [`docs/phase5e_finetune_recovery.md`](docs/phase5e_finetune_recovery.md) §13.

---

## 10. Final Release Decision

| Stage | Dataset | mAP50-95 | Decision |
|---|---|---:|---|
| Baseline | COCO128 (smoke) | 0.44369 | reference |
| E5-1 compressed | COCO128 (smoke) | 0.0 | ACCURACY_REGRESSION |
| Phase C fine-tuned | COCO128 (smoke) | 0.46688 | smoke threshold band reached → requires unseen validation |
| Baseline | COCO val2017 (unseen) | 0.36804 | reference (public 37.3 ±0.5 pp) |
| E5-1 compressed | COCO val2017 (unseen) | 0.0 | ACCURACY_REGRESSION |
| Phase C fine-tuned | COCO val2017 (unseen) | **0.00917** | **VAL2017_OVERFIT** |

```
COCO128:      Phase C = 0.46688
COCO val2017: Phase C = 0.00917
Final:        VAL2017_OVERFIT
Release:      FAIL
```

| 판정 축 | 결과 |
|---|---|
| Execution (NetsPresso) | COMPLETED (2/2) |
| Compression Gate | PASS |
| Local Evaluation Gate | PASS (latency/memory/proxy; accuracy optional) |
| **Release Gate** | **FAIL** — accuracy drop 35.89 pp > 1.0 pp (unseen), checksum 미검증(promoted baseline 없음), reproducibility NOT_VERIFIED |
| Defect | `VAL2017-release-ACCURACY_REGRESSION` [HIGH] (역사적 `E5-1-release-ACCURACY_REGRESSION` 유지) |
| Baseline promotion | 수행하지 않음 |
| **NetsPresso additional execution** | **HOLD** |

---

## 11. Key QA Findings

1. **Smoke Test PASS ≠ Release PASS.** COCO128 0.46688(baseline 초과)이 val2017 0.00917로 붕괴했다. train == val 데이터의 회복은 일반화 증거가 아니며, 독립 validation이 Release Gate의 필수 입력이어야 한다.
2. **Output Equivalence Proxy ≠ Task-level Accuracy.** E5-1 cosine 0.984 / mAP 0.0, Phase C cosine 0.990 / val2017 mAP 0.009. proxy는 보조 신호일 뿐이다.
3. **SDK `completed` ≠ 품질.** 1회차·2회차 모두 compression Gate PASS였지만 릴리스 질문(accuracy·재현성·체크섬)에는 답하지 못했다.
4. **FLOPs 감소 ≠ 지연 개선.** 1회차 샘플은 FLOPs −52 %에도 CPU 지연이 줄지 않았고(+2.5 %), YOLOv8n에서는 −30 %가 나왔다 — 모델별로 측정해야 한다.
5. **측정 방법이 결과를 뒤집는다.** 같은 프로세스에서 두 모델을 순서대로 측정하면 memory 비교가 뒤집혀(+86 % → 격리 측정 −23.5 %) 평가기를 프로세스 격리로 바꿨다. 참조 모델을 train 모드로 한 번 forward하면 BN 통계가 오염되어 가짜 FAIL이 났던 사례도 기록했다.
6. **공식 참조 코드도 검증 대상이다.** archived Nota 포크의 `C2f.forward`가 upstream과 다른 의미(Bottleneck 비연쇄)를 가져 yolov8n을 0.135로 평가했다; 레이어별 diff로 원인을 확정하고 upstream 기반 traceable export로 대체했다.
7. **결측은 PASS가 아니다.** 첫 산출물의 체크섬·단일 실행 재현성을 결함으로 오분류하던 로직을 "증거 없음(NOT_APPLICABLE)"으로 고쳤고, 그래도 필수 기준은 Gate를 막는다.
8. **Credit은 공학적 제약이다.** 500 Credit 중 50만 쓰고도 압축 → 회복 → 독립 검증 → Release 차단까지 끝낸 것은 Mock-first, 0-Credit 로컬 검증, exactly-one guard 설계 덕분이다.

---

## 12. Automation & CI

[REAL] `.github/workflows/qa.yml`(Python 3.14, ubuntu): 프레임워크 설치 → `netspresso` 미설치 assert → `ruff check` → unit / integration / regression → 전체 스위트 + coverage(xml) + JUnit → Mock QA 워크플로우 리포트 생성 → **Credit 장부 무결성 assert**(CI는 Credit을 쓰지 않으며 모든 entry는 confirmation=true인 실 작업이어야 함) → 리포트 아티팩트 업로드. CI는 API 키를 참조하지 않고 NetsPresso에 접속하지 않습니다.

로컬 자동화 스크립트(모두 0 Credit, 실 호출 스크립트 2종만 예외):

| 스크립트 | 환경 | 역할 |
|---|---|---|
| `scripts/run_mock_qa.py`, `compare_regression.py` | 3.14 | Mock QA 워크플로우(매트릭스/pairwise/risk), 회귀 비교 |
| `scripts/netspresso_dry_run.py` | 3.11 | 실행 계획·Credit 추정만(API 호출 없음) |
| `scripts/run_real_netspresso.py`, `run_e5_1_experiment.py` | 3.11 | **실 호출**(`--confirm-credit-use` 필수, exactly-one guard) |
| `scripts/run_local_eval.py` | 3.11 | 실 산출물의 0-Credit 로컬 검증(구조·params/FLOPs·ORT·proxy·Gate) |
| `scripts/yolov8_baseline_prep.py`, `yolov8_r1_traceable_export.py`, `yolov8_r1_fork_crosscheck.py` | 3.11 | YOLOv8n baseline, traceable fx export, 포크 원인 분석 |
| `scripts/yolov8_e5_1_local_validation.py`, `yolov8_e5_1_finetune.py`, `yolov8_val2017_eval.py` | 3.11 | E5-1 후보 검증, 로컬 fine-tuning(A/B/C), COCO val2017 독립 평가 |

테스트: **198 tests, coverage 93 %** — 모두 오프라인, 실 NetsPresso 호출 테스트 없음. 실 실행 레코드 기반 테스트는 저장된 JSON 레코드의 일관성(SHA 체인, 산술, Gate 의미론, 장부 불변, 시크릿 부재)을 검증합니다.

---

## 13. Reproducibility / Traceability

[REAL] 전 체인의 SHA-256과 레코드가 저장소의 텍스트 증거로 연결됩니다(모델 바이너리는 미추적).

```
yolov8n.pt (f59b3d83…, ultralytics assets)
  → R1 fx body  d8e761da…  (upstream 8.4.173 + wrapper; 2회 export bit-identical)       reports/yolov8_baseline/*_r1/
  → E5-1 automatic_compression (NetsPresso, 2026-10-05, 475→450)                         reports/real_runs/20261005T040009Z_*/
  → E5-1 candidate  21d8cbd1…                                                            reports/yolov8_e5_1/20261005T040009Z/
  → local fine-tune  A 0a203872… → B 73a8f76f… → C dba5cd21…                             reports/yolov8_e5_1_finetune/*_phase{A,B,C}/
  → COCO val2017 evaluation (baseline / E5-1 / C)                                        reports/yolov8_e5_1_finetune/*_val2017/
  → Quality Gate (compression PASS / local_eval PASS / release FAIL) → VAL2017_OVERFIT → HOLD
```

재현성 상태는 정직하게 **NOT_VERIFIED**입니다: 실 압축·fine-tuning·val2017 평가 모두 1회 실행이며, 레지스트리(`reports/baselines/registry.json`)의 모든 항목은 `candidate`로만 등록되어 있고 승격(promotion)은 수행하지 않았습니다. 재현성이 확인된 것은 R1 fx export(2회 bit-identical)와 COCO128 baseline 재측정(0.44369 = 0.44369)뿐입니다. 환경 fingerprint(OS·CPU·Python·torch·ultralytics·onnxruntime, 시크릿 필터링)는 모든 레코드에 포함됩니다.

---

## 14. Credit Safety

| 층 | 장치 |
|---|---|
| 구조 | 실 API 코드는 3.11 인터프리터에만; 3.14 프레임워크·pytest·CI는 `netspresso`를 import할 수 없음 |
| 런타임 | `credit_consuming` 어댑터는 `allow_credit_consuming=True` 없이 `PipelineRunner`가 거부; NetsPressoAdapter는 DRY_RUN 기본 |
| 승인 | `mode: real` + `--confirm-credit-use` + 입력 SHA 일치 + precheck READY + 장부 prior-ops 일치 (exactly-one guard), 재시도 없음 |
| 0-Credit 스크립트 가드 | `NETSPRESSO_API_KEY` 존재 시 abort(값 미독), `netspresso` import 감지 시 abort, 장부 SHA 전후 비교 |
| 장부 | `simulated` / `estimated` / `actual` 분리; `actual`은 계정 잔액 before/after 관측값만 |

**최종 상태 [REAL]: 사용 50 / 잔여 450 / 실 작업 2회** (각 25 Credit, 클라이언트 상수와 관측 차감 일치). 2회차 이후의 모든 작업(로컬 검증·fine-tuning A/B/C·val2017)은 API 호출 0, Credit 0입니다.

---

## 15. Limitations

- **실 NetsPresso 실행은 `automatic_compression` 2회뿐**이며 quantize/convert/profile/graph-optimize의 실 경로는 미구현(인가되어도 `RealExecutionNotImplementedError`). 매트릭스·pairwise·risk 데모 수치는 MOCK입니다.
- 실측 모델은 YOLOv8n 1종, 단일 configuration(`yolov8n | intel_xeon_w2233 | onnxruntime | default | automatic_compression`). 타깃 디바이스 성능은 미측정(로컬 CPU 개발 머신만, GPU 없음).
- **로컬 fine-tuning은 COCO128(128장, train == val)로만 수행**했고 그 결과는 unseen 데이터에서 붕괴했습니다. 적절한 규모의 학습 데이터와 holdout 기반 early stopping이 없었다는 점이 복구 실패의 직접 원인 후보이지만, 그 자체를 검증하지는 않았습니다(NOT_VERIFIED).
- 재현성은 NOT_VERIFIED(각 단계 1회 실행), baseline promotion 없음.
- Quality Gate 임계값·리스크 가중치는 PROJECT-DEFINED EXAMPLE이며 Nota 공식 기준이 아닙니다. 회귀 판정은 임계값 비교로 통계적 유의성을 주장하지 않습니다.
- SDK 조사(`docs/netspresso_sdk_research.md`)는 설치 패키지 introspection과 공개 저장소 열람에 기반하며 공식 API 문서가 아닙니다. 서버 측 지원 조합·Credit 차감 규칙은 관측한 2건 외에는 NOT_VERIFIED입니다.
- 공식 Nota YOLOv8 워크플로우 저장소(ModelZoo-YOLOv8, ultralytics_nota)는 2024-02 이후 archived 상태입니다.

---

## 16. Future Work

- 복구 전략 재설계: 수천 장 이상의 학습 데이터 + val과 분리된 holdout + early stopping으로 fine-tuning을 다시 설계하고, val2017에서 재평가 — 그 결과가 나오기 전까지 NetsPresso 추가 실행은 HOLD.
- 재현성 Level 2+: 동일 조건 2회차 실행(압축·학습·평가)으로 bitwise/functional 재현성 판정, 그 후에만 baseline promotion.
- 실 경로 확장: quantize/convert/profile 어댑터 구현은 설계되어 있으나 Credit 비용(50/50/25) 대비 효용을 각 단계마다 별도 승인으로 판단.
- 타깃 디바이스 측정(Profiler 또는 실 디바이스)과 개발 머신 측정의 분리 리포팅.
- Gate 임계값을 도메인 데이터로 보정하고, 반복 측정 분산을 회귀 판정에 반영.

---

## 17. Quick Start / Repository Map

```bash
# 3.14 메인 환경 — 프레임워크·테스트·Mock 리포트 (오프라인, 0 Credit)
python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev]"
pytest -q                                   # 198 tests
ruff check .
python scripts/run_mock_qa.py --strategy pairwise        # -> reports/examples/mock_pairwise/
python scripts/compare_regression.py reports/examples/mock_full/qa_report.json reports/examples/mock_full_regressed/qa_report.json

# 3.11 SDK 환경 (선택; 실 호출은 명시 승인 시에만)
py -3.11 -m venv .venv-netspresso && .venv-netspresso/Scripts/python -m pip install -r requirements/netspresso-py311.lock.txt
cp .env.example .env                        # NETSPRESSO_API_KEY (커밋되지 않음)
.venv-netspresso/Scripts/python scripts/netspresso_dry_run.py --operation automatic_compression   # 계획만, API 호출 없음
```

```
configs/     models/devices/runtimes/backends/optimizations · quality_gate (profiles) · risk · mock_scenarios · netspresso (dry-run policy) · local_eval (trust list, ORT policy)
framework/   adapters/ (base, mock, netspresso, factory) · matrix/ (generator, pairwise, risk) · pipeline/ (result model, runner)
             validation/ (accuracy, performance, artifact, reproducibility, structural, equivalence)
             evaluation/ (artifact_structure, model_stats, local_ort, environment, onnx_checksum, detection_head, detection_accuracy,
                          fx_traceability, input_gate, recovery) · quality_gate · defects · regression · reporter · credit_ledger · baselines · config
tests/       unit / integration / regression — 198 tests, 모두 오프라인
scripts/     §12 표 참조
reports/     credit_usage.json (장부) · baselines/registry.json · real_runs/ (실 실행 2회: 메타데이터·레코드, 바이너리 미추적)
             local_eval/ · yolov8_baseline/ (baseline, R1) · yolov8_e5_1/ (E5-1) · yolov8_e5_1_finetune/ (phase A/B/C, val2017) · examples/ (MOCK)
docs/        netspresso_sdk_research · quality_gap_analysis · test_strategy · release_quality_gate · historical_yolov8_regression ·
             phase5e_yolov8_strategy · yolov8_baseline_strategy · phase5e_release_test_plan · phase5e_conditional_go_resolution ·
             phase5e_e5_1_experiment · phase5e_finetune_recovery · portfolio_summary
```

문서 읽기 순서(채용 검토용): [`docs/portfolio_summary.md`](docs/portfolio_summary.md) → 이 README §6–§11 → [`docs/phase5e_finetune_recovery.md`](docs/phase5e_finetune_recovery.md) §13(최종 증거) → [`docs/test_strategy.md`](docs/test_strategy.md) / [`docs/release_quality_gate.md`](docs/release_quality_gate.md).

---

*Personal portfolio project using publicly available NetsPresso APIs/SDKs and technical resources. Not affiliated with or endorsed by Nota Inc.*

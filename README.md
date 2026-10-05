# AI 모델 최적화 플랫폼을 위한 QA Automation Framework

**NetsPresso-based AI Model Optimization QA Automation Framework**

> **This is a personal QA automation portfolio project using publicly available NetsPresso SDK/API resources.**
> 본 프로젝트는 공개된 NetsPresso SDK/API 및 기술 자료만을 사용한 **개인 포트폴리오 프로젝트**입니다.
> Nota의 내부 QA 프레임워크, 내부 시스템, 내부 소스 코드, 고객 환경, 비공개 모델과는 어떤 관련도 없으며, Nota의 내부 QA 프로세스를 대변하지 않습니다.

[![qa](https://img.shields.io/badge/CI-GitHub_Actions-informational)](.github/workflows/qa.yml)
![python](https://img.shields.io/badge/python-3.14_(framework)_|_3.11_(SDK_env)-blue)
![credits](https://img.shields.io/badge/NetsPresso_Credits_used-25_/_500-success)

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [AI 모델 최적화 QA와의 관계](#2-ai-모델-최적화-qa와의-관계)
3. [QA Engineer 직무 요구사항과의 연결](#3-qa-engineer-직무-요구사항과의-연결)
4. [Architecture](#4-architecture)
5. [Python 환경 분리](#5-python-환경-분리)
6. [Matrix Testing](#6-matrix-testing)
7. [Pairwise Testing](#7-pairwise-testing)
8. [Risk-based Testing](#8-risk-based-testing)
9. [Quality Gate](#9-quality-gate)
10. [Result Model](#10-result-model)
11. [Defect Classification](#11-defect-classification)
12. [Regression Testing](#12-regression-testing)
13. [Artifact Validation](#13-artifact-validation)
14. [Reproducibility Validation](#14-reproducibility-validation)
15. [Mock-first 전략](#15-mock-first-전략)
16. [향후 NetsPressoAdapter](#16-향후-netspressoadapter)
17. [Credit-safe 설계](#17-credit-safe-설계)
18. [GitHub / CI 구조](#18-github--ci-구조)
19. [현재 한계 (Limitations)](#19-현재-한계-limitations)
20. [Quick Start](#20-quick-start)
21. [QA 설계 문서 (5-C) 와 0-Credit 로컬 검증 (5-D)](#21-qa-설계-문서-phase-5-c-와-0-credit-로컬-검증-phase-5-d)

---

## 1. 프로젝트 개요

AI 모델 최적화 플랫폼(압축·양자화·변환·프로파일링)은 "모델이 돌아간다"는 사실만으로 릴리스를 판단할 수 없습니다. 최적화 후에도 **정확도가 허용 범위 안에 있는지, 지연 시간과 메모리가 실제로 줄었는지, 배포 산출물이 무결한지, 같은 입력으로 같은 결과가 재현되는지**를 정량적으로 확인해야 하고, 그 결과가 **수많은 Model × Device × Runtime × Backend × Optimization 조합**에서 일관되는지 확인해야 합니다.

이 프로젝트는 그 문제를 다루는 QA 자동화 프레임워크입니다. 핵심은 "테스트를 실행하는 것"이 아니라 아래 흐름을 코드와 설정으로 명시하는 것입니다.

```
단순 테스트 실행  →  테스트 전략 설계  →  자동화  →  결과 정량화  →  Quality Gate  →  Release 판단 근거
```

| 구분 | 내용 |
|---|---|
| 대상 도메인 | AI 모델 최적화 파이프라인 (NetsPresso 공개 SDK를 참조 대상으로 삼음) |
| 프레임워크 언어/환경 | Python 3.14, pytest, PyYAML, Jinja2 (최소 의존성) |
| 실 SDK 환경 | Python 3.11 별도 venv (`.venv-netspresso`, netspresso 1.17.0) — 현재 조사 완료, 실 호출은 미실행 |
| 현재 단계 | 2–4(Mock QA 엔진 + CI) → 5-A(안전 경계) → 5-B(실 작업 1회, 25 Credit) → 5-C(Quality Gap 분석) → **5-D 완료: 0 Credit 로컬 검증**(Gate 프로파일·레지스트리·구조 검증·독립 검증·로컬 ORT 평가). 실 API 작업 1회, Credit 사용 **25 / 500**(잔액 475) |

---

## 2. AI 모델 최적화 QA와의 관계

일반 소프트웨어 QA와 달리 모델 최적화 QA에는 다음 특성이 있습니다. 이 프레임워크의 각 구성 요소는 그 특성 하나씩에 대응합니다.

| 모델 최적화 QA의 특성 | 왜 문제인가 | 이 프레임워크의 대응 |
|---|---|---|
| 결과가 이진(성공/실패)이 아니라 **연속값**(정확도, 지연, 메모리) | "조금 나빠짐"을 허용할지 판단 기준이 필요 | 임계값 기반 **Quality Gate** (`configs/quality_gate.yaml`) |
| 조합 폭발 (모델 × 디바이스 × 런타임 × 백엔드 × 최적화) | 전체 실행은 비용·시간상 불가 | **Pairwise + Risk-based** 선택, 상태 분리(NOT_TESTED vs UNSUPPORTED) |
| 지원 여부가 조합마다 다르고 문서로 확정되지 않음 | 미지원을 실패로, 미검증을 성공으로 오판하기 쉬움 | 5단계 상태 모델, 지원 지식은 **명시적 근거가 있을 때만** UNSUPPORTED |
| 산출물(모델 파일)이 곧 배포물 | 파일 존재만으로는 무결성·추적성 보장 불가 | **Artifact Validation** (존재·크기·SHA-256·메타데이터) |
| 최적화 과정에 비결정성이 개입 가능 | 같은 설정인데 다른 결과 → 회귀 판단 불가 | **Reproducibility** 4단계 판정 (bitwise / functional / not / not verified) |
| 실패 원인이 다양 (인증, 입력, 호환성, 서버, 회귀…) | "실패"만 기록하면 분석이 시작되지 않음 | **구조화된 Defect Classification**, 근거 부족 시 UNCLASSIFIED |
| 클라우드 최적화 서비스는 **유료(Credit)** | 디버깅 중 반복 호출 = 예산 소진 | **Mock-first**, 인터프리터 분리, `--confirm-credit-use`, Credit Ledger |

---

## 3. QA Engineer 직무 요구사항과의 연결

AI 최적화 플랫폼 QA Engineer 포지션에서 일반적으로 요구되는 역량과, 이 저장소에서 그 역량을 확인할 수 있는 위치입니다.

| 역량 | 이 저장소에서의 근거 |
|---|---|
| 테스트 전략 설계 | §6–§9, `configs/` 전체가 "무엇을 왜 테스트하는가"의 선언 |
| 정형 테스트 설계 (조합 테스트) | `framework/matrix/pairwise.py` — 결정적 pairwise, 커버리지 **실측** 보고 |
| Risk-based testing | `framework/matrix/risk.py`, `configs/risk.yaml` — 선택 이유를 케이스마다 문자열로 기록 |
| Python 자동화 / CLI·API 지향 검증 | `framework/pipeline/runner.py`, `scripts/run_mock_qa.py`, `scripts/compare_regression.py` |
| Quality Gate 설계 | `framework/quality_gate.py` — 기준별 결과 + 실패 사유를 절대 숨기지 않음 |
| Regression testing | `framework/regression.py` — baseline vs current, 임계값 기반, 통계적 유의성 주장 없음 |
| Artifact / Reproducibility 검증 | `framework/validation/artifact.py`, `reproducibility.py` |
| 구조화된 결함 분류 / 근본원인 지향 분석 | `framework/defects.py` — 증거 기반 매핑, `suspected_cause`는 "의심"으로만 표기 |
| 리포팅 | `framework/reporter.py` — JSON(기계용) + HTML(사람용) |
| CI 자동화 | `.github/workflows/qa.yml` — lint → unit → integration → regression → coverage → 리포트 아티팩트 |
| 환경/설정 분석 | `docs/netspresso_sdk_research.md` — Python 3.14 vs SDK 의존성 충돌을 ENVIRONMENT/CONFIGURATION 이슈로 문서화 |
| 자격 증명 안전 처리 | `.env.example`, `.gitignore`, 시크릿 스캔 테스트(`tests/unit/test_reporter_and_safety.py`) |

---

## 4. Architecture

```
                     QA Automation Framework (Python 3.14)
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
              MockAdapter                   NetsPressoAdapter
              Python 3.14                   Python 3.11 (.venv-netspresso)
              0 Credit, 결정적              실 API — 승인 후에만, 향후 Phase 9+
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                         Common Result Model (JSON)
                    framework/pipeline/result.py
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
       Matrix / Pairwise      Validation +           Regression
       / Risk selection       Quality Gate         (baseline vs current)
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    ▼
                       Defect Classification
                                    ▼
                        JSON / HTML Report
```

설계 원칙:

- **Adapter 추상화** (`framework/adapters/base.py`): QA 엔진은 `BaseAdapter` 계약만 알고, 벤더 SDK의 내부 구조를 모릅니다. 어댑터가 무엇을 하든 결과는 동일한 `ExecutionResult`로 귀결됩니다.
- **실행 사실과 QA 판정의 분리**: `ExecutionResult`(무슨 일이 있었나) → `CaseResult`(그래서 PASS인가). 실행 성공 ≠ PASS, 실행 실패 ≠ 분류된 결함.
- **설정 주도**: 매트릭스 차원, 임계값, 리스크 가중치, 목 시나리오가 모두 YAML. 코드 수정 없이 QA 정책을 바꿀 수 있고, 잘못된 설정은 `ConfigurationError`로 조기에 드러납니다.
- **최소 의존성**: 메인 프레임워크는 PyYAML + Jinja2만 사용합니다. torch/tensorflow는 SDK 환경에만 존재합니다.

디렉터리:

```
configs/            models / devices / runtimes / backends / optimizations / quality_gate / risk / mock_scenarios / netspresso (provider·execution policy)
framework/
  adapters/         base.py, mock_adapter.py, netspresso_adapter.py (dry-run only), factory.py
  matrix/           generator.py (full product, support), pairwise.py, risk.py
  pipeline/         result.py (Result Model), runner.py
  validation/       accuracy.py, performance.py, artifact.py, reproducibility.py
  quality_gate.py   defects.py   regression.py   reporter.py   credit_ledger.py   config.py
  templates/        run_report.html.j2, regression_report.html.j2
tests/              unit / integration / regression (89 tests, 모두 오프라인)
scripts/            run_mock_qa.py, compare_regression.py, netspresso_dry_run.py
reports/            credit_usage.json (0/500), examples/ (생성된 예시 리포트)
docs/               netspresso_sdk_research.md (Phase 1 SDK 조사, source of truth)
.github/workflows/  qa.yml
```

---

## 5. Python 환경 분리

Phase 1 조사에서 검증된 사실(자세한 내용은 [`docs/netspresso_sdk_research.md`](docs/netspresso_sdk_research.md)):

- 설치된 `netspresso==1.17.0`은 `netspresso-trainer==1.3.1`을 고정하고, 이는 `torch<=2.0.1`, `torchvision<=0.15.2`를 요구합니다. 해당 torch 버전의 wheel은 **Python 3.11까지만** 제공됩니다.
- 따라서 Python 3.12+ 환경에서는 SDK 설치 자체가 실패합니다. 이 프로젝트의 메인 환경(3.14)이 그 경우입니다.

이 발견은 **ENVIRONMENT / CONFIGURATION 이슈, 심각도 Medium**으로 분류했습니다. SDK 기능 결함이 아니라 패키징·문서 정합성 이슈이며 우회가 가능하므로 과장하지 않습니다.

| 환경 | 역할 |
|---|---|
| **Python 3.14** (`.venv`) | QA 프레임워크, MockAdapter, pytest, 매트릭스/pairwise/리스크, Quality Gate, 회귀, 리포팅, CI |
| **Python 3.11** (`.venv-netspresso`) | NetsPresso SDK 1.17.0, 향후 NetsPressoAdapter 및 실 API 실험 **전용** |

두 환경은 **JSON 결과 파일**(`ExecutionResult` 스키마)로만 통신합니다. 메인 프레임워크는 `netspresso`를 import하지 않으며, 이 사실은 테스트(`test_framework_never_imports_netspresso`)와 CI(설치 여부 assert)로 강제됩니다.

---

## 6. Matrix Testing

`Model × Device × Runtime × Backend × Optimization` 다섯 차원을 YAML로 선언하고 전체 조합을 생성합니다.

| 차원 | 현재 값 | 비고 |
|---|---|---|
| Model | yolov8n, mobilenet_v2, pidnet_s | 공개 모델 기준. `mock_baseline` 수치는 **합성값**이며 실측이 아님 |
| Device | intel_xeon_w2233, raspberry_pi_4b, jetson_orin_nano | SDK `DeviceName` enum의 실제 값(`Intel-Xeon`, `RaspberryPi4B`, `Jetson-Orin-Nano`)에 매핑. SDK에 범용 "CPU"는 없음 |
| Runtime | onnxruntime, tflite, tensorrt, openvino | |
| Backend | default, xnnpack(tflite 전용), openvino_ep(onnxruntime 전용) | 런타임별 적용 가능 백엔드는 구조적 제약 |
| Optimization | fp16_conversion, int8_quantization, automatic_compression | SDK 서비스에 대응 |

**상태 모델** — 서로 다른 의미를 절대 합치지 않습니다.

| 상태 | 의미 |
|---|---|
| `PASS` | 실행 완료 + Quality Gate 통과 |
| `FAIL` | 실행 오류 **또는** 실행 완료 후 Gate 실패 |
| `BLOCKED` | 외부 선행 조건 미충족으로 실행 불가 (예: 디바이스 팜 슬롯 없음) |
| `NOT_TESTED` | 지원 가능성은 있으나 **아직 검증하지 않음** (선택되지 않은 셀의 초기 상태) |
| `UNSUPPORTED` | 명시적 근거로 **미지원임이 알려진** 조합. 실패가 아님 |

지원 판정 정책: `UNSUPPORTED`는 (a) 백엔드-런타임 구조적 제약 또는 (b) `devices.yaml`의 `known_unsupported`에 **출처가 인용된** 규칙에 해당할 때만 부여합니다. 그 외는 모두 `UNKNOWN` → `NOT_TESTED`에서 시작합니다. 지원된다고 "가정"하는 코드 경로는 없습니다.

현재 설정의 전체 매트릭스: **324 조합** = 126 feasible + 198 UNSUPPORTED(구조적/문서 근거).

---

## 7. Pairwise Testing

126개 feasible 조합을 모두 실행하는 대신, 모든 차원 쌍(2-wise)의 값 조합이 최소 1회 등장하도록 케이스를 고릅니다. 구현은 결정적 greedy set-cover(AETG/IPOG 계열의 단순화)이며, 동률은 입력 순서로 해소하므로 동일 입력 → 동일 출력입니다.

프레임워크가 **계산해서** 보고하는 값 (주장이 아니라 실측):

| 항목 | 값 (현재 설정) |
|---|---|
| Feasible 조합 | 126 |
| 선택된 조합 | 15 |
| 축소율 | 88.1 % |
| 전체 매트릭스의 값 쌍 | 102 |
| Feasible 쌍 | 92 |
| 커버된 쌍 / 커버리지 | 92 / **100.0 %** |
| Infeasible 쌍 (UNSUPPORTED 조합에만 등장) | 10 — 별도 목록으로 보고, 조용히 버리지 않음 |

수학적 최적성(최소 케이스 수)은 주장하지 않습니다. 리포트가 보여주는 것은 "선택된 셀이 feasible 쌍을 100 % 덮는다"는 검증 가능한 사실입니다. 선택되지 않은 111개 셀은 정직하게 `NOT_TESTED`로 남습니다.

---

## 8. Risk-based Testing

pairwise가 "넓게"라면 risk-based는 "위험한 곳을 깊게"입니다. 각 차원 항목에 불리언 리스크 플래그를 두고, `configs/risk.yaml`의 가중치를 더해 점수와 등급(HIGH/MEDIUM/LOW)을 만듭니다.

| 리스크 요인 | 가중치 | 근거 |
|---|---|---|
| new_model / new_device | 3 | 검증 이력이 없는 항목 |
| previous_regression | 3 | 과거 회귀가 기록된 항목은 재발 가능성 |
| known_compatibility_issue | 3 | 알려진 호환성 문제 |
| customer_facing | 2 | 결함 시 영향 범위가 큼 |
| new_runtime / new_backend / new_method | 2 | 신규 실행 경로 |
| lossy_precision (INT8) | 1 | 손실 최적화는 정확도 리스크 내재 |

케이스마다 `risk.factors`에 `"model.previous_regression (+3)"` 같은 문자열이 남아, **왜 선택되었는지** 리포트에서 그대로 읽을 수 있습니다. 전략은 `full`, `pairwise`, `risk`(HIGH만), `pairwise_risk`(pairwise ∪ HIGH) 네 가지입니다.

| 전략 | 선택 | 실행 | PASS / FAIL / BLOCKED | NOT_TESTED |
|---|---|---|---|---|
| full | 126 | 123 | 112 / 8 / 3 | 0 |
| pairwise | 15 | 14 | 9 / 4 / 1 | 111 |
| risk (HIGH) | 34 | 34 | 27 / 4 / 3 | 92 |
| pairwise_risk | 43 | 42 | 34 / 5 / 3 | 83 |

(실행 수가 선택 수보다 적은 것은 실행 시점에 UNSUPPORTED로 판명된 케이스가 있기 때문입니다. 이 역시 실패로 집계하지 않습니다.)

---

## 9. Quality Gate

Quality Gate는 "릴리스 판단 근거"를 코드로 고정한 것입니다. 각 기준은 임계값과 `required` 플래그를 가집니다.

```yaml
quality_gate:
  accuracy:        { max_drop_percent: 1.0,      required: true }   # 절대 퍼센트포인트
  latency:         { max_increase_percent: 10.0, required: true }   # baseline 대비 상대 증가율
  memory:          { max_increase_percent: 15.0, required: true }
  model_size:      { max_increase_percent: 10.0, required: false }  # 실패 시 WARN
  artifact:        { required: true }
  reproducibility: { required: true }
```

규칙:
- `required: true` 기준이 하나라도 FAIL → Overall FAIL. 사유는 `reasons` 목록에 **전부** 기록됩니다.
- `required: false` 기준의 실패는 WARN으로 표시되고 판정을 바꾸지 않습니다.
- 필수 기준을 평가할 데이터가 없으면(NOT_APPLICABLE) **FAIL**입니다. 결측 데이터 위에서 PASS를 선언하지 않습니다.
- 실행이 COMPLETED가 아니면 Gate는 평가되지 않고 그 사실이 사유로 남습니다.

실제 출력 예 (MockAdapter, 시나리오 SCN-001):

```
QUALITY GATE
--------------------------------
Accuracy            FAIL
Latency             PASS
Memory              PASS
Model Size          PASS
Artifact Integrity  PASS
Reproducibility     PASS
--------------------------------
Overall             FAIL

Reasons:
  - Accuracy: accuracy drop 3.15 pp exceeds threshold 1.00 pp
DEFECT ACCURACY_REGRESSION [HIGH]
  suspected cause: INT8 quantization precision loss / calibration coverage (suspected, not confirmed)
```

---

## 10. Result Model

`framework/pipeline/result.py`가 두 환경이 공유하는 계약입니다. NetsPresso 내부 구조에 결합되지 않은 일반 필드만 사용합니다.

**`ExecutionResult`** (어댑터 출력 — 사실): `operation`, `configuration{model, device, runtime, backend, optimization}`, `execution_status(COMPLETED|ERROR|BLOCKED|UNSUPPORTED|NOT_EXECUTED)`, `execution_time_s`, `baseline_metrics`, `metrics{accuracy, latency_ms, memory_mb, model_size_mb, extra}`, `artifact{path, exists, size_bytes, format, checksum_sha256, expected_checksum_sha256, checksum_valid, metadata}`, `logs`, `error{kind, message, details}`, `environment{adapter, adapter_version, sdk_name, sdk_version, python_version, platform}`, `reproducibility{level, runs, checksums}`, `timestamp`, `credit{usage_type: simulated|estimated|actual|none, estimated, actual}`, `scenario_id`

**`CaseResult`** (QA 판정): `status`, `support`, `selected`, `selection_reason`, `risk{score, level, factors}`, `execution`, `quality_gate{overall, criteria[], reasons[]}`, `defect`

`baseline_metrics`와 `metrics`(optimized/current)는 항상 분리되어 있으며, 모든 객체는 `to_dict()/from_dict()`로 무손실 JSON 라운드트립이 가능합니다(테스트로 보장). Credit 필드는 기본값이 `none`이고, `actual`은 실제 계정 조회값이 있을 때만 기록됩니다. MockAdapter는 항상 `simulated`를 씁니다.

---

## 11. Defect Classification

**테스트 결과와 결함 분류는 다른 질문에 답합니다.** 결과(`Status`)는 "릴리스 기준을 충족했는가", 분류(`Defect`)는 "증거가 어떤 종류의 문제를 가리키는가"입니다.

카테고리: `AUTH_ERROR`, `INPUT_VALIDATION_ERROR`, `MODEL_COMPATIBILITY_ERROR`, `OPTIMIZATION_FAILURE`, `CONVERSION_FAILURE`, `RUNTIME_ERROR`, `ACCURACY_REGRESSION`, `PERFORMANCE_REGRESSION`, `MEMORY_REGRESSION`, `ARTIFACT_ERROR`, `REPRODUCIBILITY_ERROR`, `CONFIGURATION_ERROR`, `ENVIRONMENT_ERROR`, 그리고 **`UNCLASSIFIED`**.

분류에 사용하는 증거는 두 가지뿐입니다.
1. Quality Gate의 **기준별 결과** (accuracy 실패 → ACCURACY_REGRESSION 등; 복수 실패 시 우선순위로 primary/secondary 구분)
2. 어댑터가 넘긴 **명시적 `error.kind`** — 카테고리 이름 또는 Phase 1에서 확인한 SDK 예외 클래스 이름의 매핑 테이블 (`NotSupportedModelException → MODEL_COMPATIBILITY_ERROR`, `NotEnoughCreditException → ENVIRONMENT_ERROR` 등)

자유 텍스트 메시지에서 카테고리를 "추측"하지 않습니다. 근거가 없으면 `UNCLASSIFIED`이고 `suspected_cause`는 `None`입니다. `suspected_cause`가 채워지는 경우에도 문구에 "(suspected, not confirmed)"를 붙여 근본원인 확정으로 읽히지 않게 합니다.

각 결함 레코드: `category`, `severity`, `message`, `evidence[]`, `suspected_cause`, `reproducibility(deterministic|unknown)`, `affected_configuration`, `secondary_categories[]`.

---

## 12. Regression Testing

두 실행(baseline run, current run)의 JSON 리포트를 케이스 ID로 매칭해 비교합니다 (`framework/regression.py`, `scripts/compare_regression.py`).

| 판정 | 정의 |
|---|---|
| **REGRESSION** | accuracy 하락 > 임계값, latency/memory/model_size 증가 > 임계값, 상태 하락(PASS→FAIL/BLOCKED), artifact checksum이 유효→무효, reproducibility가 허용 집합 밖으로 하락 — 중 하나 이상 |
| **PASS** | 위 조건 중 아무것도 해당하지 않음 (개선·불변 포함) |
| NEW / MISSING | 한쪽 실행에만 존재 |
| NOT_COMPARABLE | 비교할 메트릭이 없음 |

임계값은 기본적으로 `quality_gate.yaml`과 동일한 값을 재사용해 Gate와 회귀 판단이 어긋나지 않게 합니다. 통계적 유의성은 주장하지 않으며, 양쪽 모두 실행되지 않은 셀은 `not_executed_in_both`로 집계해 비교 대상에서 제외합니다. 비교는 케이스 ID 정렬 순으로 결정적입니다.

데모: `configs/scenarios/mock_scenarios_regressed.yaml`(현재 실행용)은 baseline 대비 3건의 회귀(지연, 정확도, 재현성)와 1건의 개선을 심어 두었고, `full` 전략 비교에서 정확히 그 3건만 REGRESSION으로 검출됩니다(`tests/regression/test_regression.py`). 흥미로운 부수 관찰: 같은 회귀를 `pairwise` 전략으로 비교하면 1건만 잡힙니다 — 조합 축소가 회귀 탐지 범위를 줄인다는 트레이드오프가 리포트 수치로 드러납니다.

---

## 13. Artifact Validation

최적화 산출물은 곧 배포물입니다. 검증 항목:

- 파일 존재, 크기 > 0
- SHA-256 체크섬 (실제 파일은 **바이너리 모드**로 계산 → Windows CRLF 변환 영향 없음), 기대값과 비교
- 포맷(확장자) 기록
- 추적성 메타데이터: `source_model`, `optimization`, `runtime`, `device`, `backend`, `adapter`/`sdk_version`, `configuration_key`, 생성 주체

`checksum_valid`가 `None`(기대값 없어 검증 불가)이면 필수 기준에서는 FAIL로 처리합니다. "검증 못 함"을 "정상"으로 보고하지 않기 위해서입니다. 실 파일용 헬퍼(`artifact_from_file`, `compute_sha256`, `verify_file_checksum`)는 향후 NetsPressoAdapter가 그대로 사용합니다.

---

## 14. Reproducibility Validation

같은 모델·설정·SDK 버전·파라미터로 반복 실행했을 때의 결과를 네 단계로 구분합니다.

| 등급 | 조건 |
|---|---|
| `BITWISE_REPRODUCIBLE` | 반복 실행의 artifact checksum이 모두 동일 |
| `FUNCTIONALLY_REPRODUCIBLE` | checksum은 다르지만 모든 실행의 메트릭이 첫 실행 대비 허용 오차 이내 |
| `NOT_REPRODUCIBLE` | 메트릭이 허용 오차를 벗어남 (또는 checksum 상이 + 메트릭 증거 없음) |
| `NOT_VERIFIED` | 반복 실행이 2회 미만 |

bit-for-bit 재현성은 checksum 증거가 있을 때만 주장합니다. `NOT_VERIFIED`는 `required: true`인 Gate에서 FAIL로 처리되어, "확인하지 않았음"이 "재현됨"으로 승격되는 일을 막습니다.

---

## 15. Mock-first 전략

MockAdapter는 편의 도구가 아니라 **QA 엔진의 정확성을 검증하는 기준 장치**입니다.

- **결정적**: 모든 수치는 `mock_baseline` × 최적화 프로파일 × 설정 키의 SHA-256 해시로 유도됩니다. `random` 모듈은 사용하지 않으며 테스트가 이를 확인합니다.
- **시나리오 주입** (`configs/mock_scenarios.yaml`): 정확도 회귀, 지연 회귀, artifact checksum 불일치, 비재현, 변환 실패, 호환성 오류, 근거 없는 실패(→ UNCLASSIFIED), BLOCKED, 런타임 UNSUPPORTED — QA 엔진의 모든 경로를 재현 가능하게 트리거합니다.
- **0 Credit, 0 네트워크**: pytest·CI가 여기에만 의존합니다.

Mock 수치는 실제 모델의 측정값이 아니며, HTML 리포트 상단에도 그 사실을 명시합니다.

---

## 16. 향후 NetsPressoAdapter

### 16.1 현재 상태: 안전 경계 + Dry-run + 실 작업 1종(`automatic_compression`) 구현·1회 실행 검증

`framework/adapters/netspresso_adapter.py`는 MockAdapter와 동일한 `BaseAdapter` 계약을 구현합니다. 실 실행 경로는 `automatic_compression` **한 가지만** 구현되어 있고, 2026-10-05에 정확히 1회 실행해 E2E 흐름(QA Engine → NetsPressoAdapter → SDK → 실 결과 → 공통 Result Model → Quality Gate → Ledger)을 검증했습니다(§16.3). 다른 작업(convert/quantize/profile/graph_optimize)의 실 경로는 미구현이며 인가되어도 `RealExecutionNotImplementedError`로 거부됩니다.

| ExecutionMode | 조건 | 동작 |
|---|---|---|
| `DRY_RUN` (기본) | `execution.mode: dry_run` | 구조화된 실행 계획(`DryRunPlan`)만 생성. SDK import 없음, 네트워크 없음, Credit 0 |
| `REAL_RUN_UNAUTHORIZED` | `mode: real`이지만 `confirm_credit_use: false` | 어떤 작업도 `RealExecutionNotAuthorizedError`로 즉시 거부 (SDK 접근 전) |
| `REAL_RUN_AUTHORIZED` | `mode: real` + `confirm_credit_use: true` (`--confirm-credit-use`) | `REAL_IMPLEMENTED_OPERATIONS`(현재 `automatic_compression`)만 실행. 순서: 정책 → 구현 여부 → 키 존재 확인 → SDK 지연 import → 인증(잔액 before) → **서비스 호출 1회** → 잔액 after → Result Model 매핑. 재시도 없음. 미구현 작업은 `RealExecutionNotImplementedError` |

- SDK는 `importlib.import_module("netspresso")`로 **지연 로드**되며, 인가된 실 실행에서만 호출됩니다. 3.14 프레임워크 어디에도 `import netspresso` 문이 없음을 테스트가 강제합니다.
- API 키는 환경변수 이름(`NETSPRESSO_API_KEY`)만 알고 **값은 저장·로그·렌더링하지 않습니다** (테스트: 가짜 키가 어떤 출력에도 나타나지 않음).
- 객체 생성 시 인증하지 않습니다. 인증은 인가된 실 실행 경로에서만 명시적 단계로 수행됩니다.
- 작업 모델: `validate_connection()`, `optimize()`(automatic_compression), `quantize()`, `convert()`, `profile()`. 각 작업은 Phase 1에서 검증된 SDK 서비스/메서드 이름(`compressor_v2().automatic_compression`, `converter_v2().convert_model`, `quantizer().automatic_quantization`, `profiler().profile_model`, `graph_optimizer().optimize_model`)에 매핑되며, 테스트가 이 이름들을 `docs/research_cache/sdk_surface.json`과 대조합니다.
- 설정: `configs/netspresso.yaml` (`provider.type: mock`, `execution.mode: dry_run`, `confirm_credit_use: false`가 기본). YAML에 키 값이 들어오면 `ConfigurationError`.

Dry-run 예시:

```
$ python scripts/netspresso_dry_run.py --operation profile --device jetson_orin_nano --runtime tensorrt
Provider:          NetsPresso
Execution:         DRY_RUN
Operation:         profile
SDK call:          NetsPresso.profiler().profile_model
Model:             mobilenet_v2
Estimated Credit:  25 (CLIENT_SIDE_PRE_CHECK_CONSTANT ...; actual server-side deduction NOT verified)
Actual Credit:     NOT MEASURED
API call:          NOT EXECUTED
Credit consumed:   0
```

`--matrix pairwise`를 주면 선택된 전체 조합의 계획과 클라이언트 측 Credit 추정 합계를 장부(`reports/credit_usage.json`)의 잔량·예비분과 대조해 예산 초과 여부를 보여줍니다. 이 숫자는 SDK의 클라이언트 측 사전 체크 상수(Automatic Compression 25, Advanced Compression 50, Convert 50, Profile 25, Quantize 50, Graph Optimize 50)이며 **서버 실제 차감이 검증된 값이 아닙니다**.

### 16.3 첫 실 실험 결과 (Phase 5-B, 2026-10-05) — `reports/real_runs/20261005T010356Z_automatic_compression/`

| 항목 | 값 |
|---|---|
| 실행 명령 | `.venv-netspresso/Scripts/python scripts/run_real_netspresso.py --confirm-credit-use` (Python 3.11) |
| 작업 | `compressor_v2().automatic_compression` × **1회**, 재시도 0 |
| 입력 모델 | 공식 샘플 `graphmodule.pt` (v1.17.0 태그, 28,485,001 bytes, 입력 `[1,3,224,224]`, ratio 0.5 — 공식 예제와 동일 조건) |
| SDK / 결과 상태 | netspresso 1.17.0 / `completed`, method `PR_L2`, 레이어 50개 |
| 모델 크기 | 28.485 MB → **7.456 MB** (파일 기준); 서버 보고 size 27.17 → 7.11, FLOPs 1.957 G → 0.943 G, 파라미터 7.06 M → 1.81 M |
| 산출물 | `sdk_output.pt` (7,456,333 bytes, SHA-256 `bcb3162d…9da5`) + SDK가 로컬 변환한 `sdk_output.onnx`. 바이너리는 git 미추적, 메타데이터·체크섬만 추적 |
| Credit | estimated 25 (CLIENT_SIDE_PRE_CHECK_CONSTANT) / **actual 25** — 계정 잔액 before 500 → after 475를 SDK 사용자·크레딧 조회로 관찰. 이 작업에 대해 클라이언트 상수와 실제 차감이 일치함을 확인 |
| accuracy / latency / memory | **NOT_AVAILABLE** — 압축 작업은 측정하지 않음. 수치를 만들어내지 않았음 |
| 기본 Quality Gate | **FAIL** — 사유 전부 "데이터 없음"(accuracy·latency·memory NOT_APPLICABLE, 기준 체크섬 없음, 단일 실행으로 재현성 NOT_VERIFIED). 결함 분류: `UNCLASSIFIED: required criteria could not be evaluated (missing data)` — 회귀로 오분류하지 않음 |
| 압축 단계 Gate (model_size 필수 + artifact) | **PASS** (크기 −73.8 %) |
| 케이스 상태 | `FAIL` (릴리스 기준 충족 증거 부족; 압축 자체는 성공). 이 구분이 바로 "실행 성공 ≠ PASS" 원칙 |

이 실험이 드러낸 프레임워크 결함 2건은 실행 전 수정했습니다: 기준 체크섬이 없는 첫 산출물과 단일 실행 NOT_VERIFIED가 각각 `ARTIFACT_ERROR`/`REPRODUCIBILITY_ERROR`로 분류되던 문제 → 증거 부족(`NOT_APPLICABLE`)으로 처리. 서버 응답에서 SDK enum에 없는 값(`MIX` data type, Jetpack `6.2.1+b38`, `dlc` framework)도 관찰되어 연구 문서 §11에 기록했습니다.

### 16.2 실 실행 경로가 지켜야 할 제약

Phase 1 조사에서 확인한 사실을 바탕으로 어댑터가 지키는 제약입니다.

- `NetsPresso(api_key=...)`를 사용합니다. email/password 경로는 SDK 코드상 deprecated입니다.
- **객체 생성 자체가 네트워크 동작**입니다: PyPI 버전 확인(구버전이면 `sys.exit(1)`), 로그인, 사용자/Credit 조회가 생성자에서 수행됩니다. 어댑터는 생성을 명시적 단계로 취급하고 `dev_mode` 사용 여부를 기록합니다.
- SDK 서비스 메서드는 예외를 잡아 `status=ERROR`인 metadata를 **정상 반환**합니다. 어댑터는 `COMPLETED` 외의 상태를 명시적 실패로 승격합니다.
- SDK 호출은 analytics 이벤트를 전송합니다(소스 확인). 어댑터는 관련 환경변수(`GA_DISABLE_ANALYTICS`)를 문서화·제어합니다.
- 결과는 SDK `metadata.json` → `ExecutionResult` JSON으로 매핑되어 3.14 프레임워크가 오프라인으로 소비합니다.
- 실행 스크립트는 `scripts/run_real_netspresso.py --confirm-credit-use`이며, 플래그 없이는 즉시 중단합니다.

---

## 17. Credit-safe 설계

NetsPresso Credit은 이 프로젝트의 소모성 자원(500)입니다. 안전장치는 여러 층으로 겹쳐 있습니다.

| 층 | 장치 |
|---|---|
| 구조 | 실 API 코드는 별도 인터프리터(3.11)에만 존재. 3.14 프레임워크·pytest·CI에서는 import 자체가 불가 |
| 런타임 | `BaseAdapter.credit_consuming=True`인 어댑터는 `allow_credit_consuming=True` 없이는 `PipelineRunner`가 생성 단계에서 `CreditSafetyError`. NetsPressoAdapter는 `DRY_RUN`에서만 `credit_consuming=False` |
| 실행 정책 | `ExecutionMode` 3단계(§16.1). 기본 `DRY_RUN`; 실 실행은 `mode: real` + `confirm_credit_use` 두 조건을 모두 요구하고, Phase 5-A에서는 그래도 실행되지 않음 |
| 스크립트 | `--confirm-credit-use` 없이는 중단 (Phase 9) |
| 장부 | `reports/credit_usage.json` — `simulated`는 실 잔액을 건드리지 않고 `simulated_total`에만 누적, `actual`은 계정 조회값 + 명시적 confirmation 없이는 기록 불가 (`CreditLedgerError`) |
| 테스트 | 저장소 장부가 0 / 500, operations 비어 있음을 확인하는 가드 테스트. CI도 동일 assert |
| 비용 근거 | SDK 소스의 클라이언트 측 상수(Automatic Compression 25, Advanced Compression 50, Convert 50, Profile 25, Quantize 50, Graph Optimize 50)를 **예산 계획용**으로만 사용. 서버 실제 차감은 미검증. Simulator는 상수 테이블에 없으나 무료로 간주하지 않음 |

현재 상태: **Credit 사용 25 / 500 (잔액 475, 계정 조회로 관찰), 실 API 작업 1회, 인증 세션 1회.** 장부: `reports/credit_usage.json` (usage_type `actual`, before/after 잔액과 추정 근거 기록).

---

## 18. GitHub / CI 구조

`.github/workflows/qa.yml` (Python 3.14, ubuntu):

1. 메인 프레임워크만 설치 → `netspresso`가 설치되지 않았음을 assert
2. `ruff check framework tests scripts`
3. `pytest tests/unit`, `tests/integration`, `tests/regression`
4. 전체 스위트 + coverage(xml) + JUnit 리포트
5. `scripts/run_mock_qa.py --strategy pairwise --regression-demo` → JSON/HTML 리포트 생성
6. `reports/credit_usage.json` 무결성 assert (CI는 Credit을 쓰지 않으며, 모든 항목은 confirmation=true인 실 작업이고 used = Σ 관찰/추정 차감이어야 함)
7. 리포트 아티팩트 업로드

CI는 API 키를 참조하지 않고, NetsPresso에 접속하지 않으며, Python 3.11 SDK 환경을 사용하지 않습니다. 실 API 실험은 CI 밖에서 수동으로만 실행됩니다.

Git 위생: `.gitignore`가 `.env`, `netspresso.env`, `.venv-netspresso/`, 모델 바이너리를 제외하고, `.gitattributes`가 LF 개행과 바이너리 파일을 고정합니다. 시크릿 패턴 스캔이 테스트에 포함되어 있습니다.

---

## 19. 현재 한계 (Limitations)

- **실 NetsPresso 실행은 1회(automatic_compression)뿐입니다.** 매트릭스·리포트 예시의 수치는 MockAdapter 합성값이며, 실 결과는 §16.3의 단일 실행만 존재합니다. 정확도·지연·메모리의 실측값은 아직 없습니다.
- `NetsPressoAdapter`의 실 경로는 `automatic_compression` 1종만 구현되어 있습니다. 실 SDK 호출 코드는 3.14 pytest에서 직접 실행할 수 없어, 3.14에서는 가짜 SDK 모듈 주입으로 매핑·오류·BLOCKED 경로를 테스트하고 3.11에서는 수동 1회 실행으로 검증하는 이원 구조입니다.
- 매트릭스의 지원 지식은 공식 예제 주석 수준의 출처만 반영했습니다. 특정 Framework × Device × DataType 조합의 실제 지원 여부는 서버 응답으로만 확정할 수 있습니다.
- Pairwise는 greedy 휴리스틱으로 최소 케이스 수를 보장하지 않습니다. 커버리지 100 %만 보장·보고합니다.
- 리스크 가중치와 Quality Gate 임계값은 데모용 초기값입니다. 실제 제품에서는 도메인 데이터로 보정해야 합니다.
- 회귀 판정은 임계값 비교이며 통계적 유의성 검정이 아닙니다. 환경 노이즈와 실제 회귀를 구분하려면 반복 측정과 분산 정보가 추가로 필요합니다.
- 로컬 GPU가 없어 GPU 대상 로컬 평가는 불가합니다.

---

## 20. Quick Start

```bash
# Python 3.14 메인 환경
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"      # Windows
# source .venv/bin/activate && pip install -e ".[dev]"  # macOS/Linux

# 테스트 (오프라인, 0 Credit)
pytest -q
pytest --cov=framework --cov-report=term-missing

# Mock QA 워크플로우 + 리포트
python scripts/run_mock_qa.py --strategy pairwise
python scripts/run_mock_qa.py --strategy full --regression-demo
#  -> reports/examples/mock_full/qa_report.{json,html}
#  -> reports/examples/mock_full_regression/regression_report.{json,html}
#  (전수(full) 매트릭스 JSON은 1 MB를 넘어 로컬에서만 생성되고 git에는 추적되지 않습니다.
#   저장소에는 HTML 리포트와 소형 JSON 예시만 포함되어 있습니다.)

# 저장된 두 리포트 비교 (REGRESSION이면 exit code 1)
python scripts/compare_regression.py reports/examples/mock_full/qa_report.json \
                                     reports/examples/mock_full_regressed/qa_report.json
```

실 NetsPresso 환경(선택, Phase 9 이후에만 필요):

```bash
py install 3.11
python3.11 -m venv .venv-netspresso
.venv-netspresso/Scripts/python -m pip install -r requirements/netspresso-py311.lock.txt
cp .env.example .env    # NETSPRESSO_API_KEY 입력. .env는 커밋되지 않음
```

---

## 21. QA 설계 문서 (Phase 5-C) 와 0-Credit 로컬 검증 (Phase 5-D)

첫 실 실험(§16.3)은 "SDK `completed` ≠ Release PASS"를 실 데이터로 보여주었습니다. 압축은 성공했지만 릴리스 판단에 필요한 6개 질문 중 크기 1개만 답할 수 있었고, 나머지는 **증거 미수집**(NOT_APPLICABLE)이었습니다. 그 결과를 기반으로 "다음에 무엇을 검증해야 실제 Release Quality Gate가 되는가"를 세 문서로 분리해 설계했습니다 (API 호출 없이 작성, Credit 0).

| 문서 | 답하는 질문 | 핵심 내용 |
|---|---|---|
| [`docs/quality_gap_analysis.md`](docs/quality_gap_analysis.md) | 지금 **무엇이 부족한가** | 실측값/SDK 보고값/QA 계산값/미측정값을 구분한 Quality Gap 표, Gate FAIL의 구조적 이유 5가지, Gap TOP 10, 0-Credit vs 실 API 분류, 다음 실험 4회·약 225 Credit 권고 |
| [`docs/test_strategy.md`](docs/test_strategy.md) | **무엇을 어떻게** 테스트하는가 | API Contract(Layer 1) vs QA Validation(Layer 2) 분리와 기존 111개 테스트의 재분류, 파이프라인 8단계 최소 Release Test Suite(TC-M01 … TC-R02), Accuracy/Latency/Memory 측정 설계, Artifact 5단계 진술, Reproducibility Level 0–4, Roadmap 5-D/5-E/5-F |
| [`docs/release_quality_gate.md`](docs/release_quality_gate.md) | **어떤 조건이면** Release PASS인가 | 6원칙, 기준 정의, Gate 프로파일(`compression_stage` / `local_eval` / `release`), Release PASS 조건 R1–R10과 현재 상태, 상태 의미론, 임계값 거버넌스 |

### Phase 5-D — 0 Credit 로컬 검증 (구현 완료)

5-C에서 식별한 구조적 Gap을 NetsPresso API 호출 없이 프레임워크에 반영하고, 5-B 산출물에 적용했습니다 (`reports/local_eval/`, `reports/baselines/registry.json`).

| 구현 | 위치 | 5-B 산출물 적용 결과 |
|---|---|---|
| Gate 프로파일 `compression` / `local_eval` / `release` | `configs/quality_gate.yaml`, `QualityGate(config, profile=…)` | compression **PASS** · local_eval **FAIL** · release **FAIL** — 각각 사유가 다름(아래) |
| Baseline 레지스트리 (`candidate → baseline → retired`, 승격 정책 명시) | `framework/baselines.py` | 압축 산출물을 `candidate`로 등록. 2회차 재현성 증거 없이는 기대 체크섬으로 쓰이지 않음 |
| 산출물 구조 검증 (PT: `weights_only` 우선, 신뢰 SHA 목록에서만 전체 unpickle; ONNX: checker + I/O/노드/initializer) | `framework/evaluation/artifact_structure.py` | `sdk_output.pt`·`graphmodule.pt` GraphModule 로드 PASS, `sdk_output.onnx` 261 노드 PASS |
| params / FLOPs 독립 검증 (`sdk_reported` / `independently_verified` / `delta` / `verification_status` 분리) | `framework/evaluation/model_stats.py` | params **정확 일치**(7,060,084 → 1,814,928). FLOPs는 ONNX MAC 추정의 **2×MACs**와 0.6–1.0 % 이내 일치 → SDK "flops" 관례 관찰 |
| 프로세스 격리 로컬 ONNX Runtime 평가 (warm-up 10 / 100회, median·p95, peak RSS) | `framework/evaluation/local_ort.py`, `scripts/run_local_eval.py` (3.11) | latency median 3.49 → 3.57 ms(+2.5 %, 개발 머신), peak RSS 76.0 → 58.1 MB. **FLOPs −52 %가 CPU 지연 개선으로 이어지지 않음** |
| Output equivalence **proxy** (정확도 아님) | `framework/validation/equivalence.py` | 3개 출력 min cosine **0.825** < 0.99 → FAIL. 재학습 없는 pruning 결과로 원본과 기능적 동등성이 깨짐 |
| Accuracy | — | **NOT_APPLICABLE** (라벨 평가셋 없음). proxy와 절대 혼용하지 않음 |
| 환경 fingerprint (시크릿 키/값 필터) | `framework/evaluation/environment.py` | OS·CPU·Python·ORT/torch 버전·EP·스레드 기록 |

측정 중 발견한 QA 교훈 두 가지: (1) 같은 프로세스에서 두 모델을 순서대로 측정하면 첫 모델이 런타임의 1회성 할당을 떠안아 memory 비교가 뒤집힌다(+86 % → 격리 측정 시 −23.5 %) — 평가기는 모델별 새 인터프리터를 사용하도록 수정했습니다. (2) SDK `completed`·크기 −74 %·params 일치라는 "성공" 신호가 모두 참이어도, 출력 동등성 proxy는 이 산출물이 그대로 배포 가능한 모델이 아님을 보여줍니다 — Gate 프로파일 분리의 이유입니다.

현재 실측 범위: 실 NetsPresso 작업 1회 + 로컬 평가. 정확도·재현성(Level 2+)·타깃 디바이스 성능은 **아직 측정되지 않았습니다**. 모든 임계값은 project-defined example이고 Nota 공식 기준이 아닙니다.

---

*Personal portfolio project using publicly available NetsPresso APIs/SDKs and technical resources. Not affiliated with or endorsed by Nota Inc.*

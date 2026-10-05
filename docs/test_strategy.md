# Test Strategy — AI Model Optimization Pipeline QA

> 질문: "**무엇을 어떻게** 테스트해서 Release 판단 근거를 만드는가?"
> 부족한 것의 목록은 [`quality_gap_analysis.md`](quality_gap_analysis.md), PASS 조건은 [`release_quality_gate.md`](release_quality_gate.md).
>
> 모든 threshold는 **project-defined example policy**다. Nota 공식 기준이 아니며, 공식 기준이 확인되면 `configs/quality_gate.yaml`에서 교체한다.

---

## 1. 두 계층: API Contract Test vs QA Validation Test

같은 "테스트"라도 묻는 질문이 다르다.

| 계층 | 질문 | 실패의 의미 | 기준의 출처 |
|---|---|---|---|
| **Layer 1 — API Contract** | "SDK/서비스가 문서·시그니처대로 동작하는가?" | 플랫폼 또는 통합 결함 (`INPUT_VALIDATION_ERROR`, `MODEL_COMPATIBILITY_ERROR`, `RUNTIME_ERROR`, `CONFIGURATION_ERROR`) | SDK 소스/시그니처(`sdk_surface.json`), 서버 응답 스키마 |
| **Layer 2 — QA Validation** | "그 결과가 **릴리스할 만한 품질**인가?" | 품질 회귀 또는 증거 부족 (`ACCURACY_REGRESSION`, `PERFORMANCE_REGRESSION`, `MEMORY_REGRESSION`, `ARTIFACT_ERROR`, `REPRODUCIBILITY_ERROR`, `UNCLASSIFIED(missing data)`) | QA가 정의한 정책(`quality_gate.yaml`) |

Layer 1이 PASS여도 Layer 2가 PASS인 것은 아니다. Phase 5-B가 정확히 그 상태다.

### 1.1 기존 테스트(111개)의 계층 분류

| 파일 | Layer 1 (Contract) | Layer 2 (QA Validation) | 기타(안전/인프라) |
|---|---|---|---|
| `test_netspresso_adapter.py` (22) | SDK 이름·시그니처 대조(`test_dry_run_plan_is_structured_and_uses_verified_sdk_names`), 작업 테이블 vs 설정 상수, 실 경로 metadata 매핑 3종(completed/error/SystemExit), SDK 예외명→카테고리 | 실 결과의 Gate/결함 판정(`UNCLASSIFIED missing data`), 장부 무변경 | 정책·키 비노출·거부·CLI 10여 개 |
| `test_quality_gate_and_defects.py` (14) | — | 전부 (Gate PASS/FAIL/WARN/NOT_APPLICABLE, 결함 분류·UNCLASSIFIED·UNSUPPORTED 비결함) | — |
| `test_validation.py` (6) | — | 전부 (accuracy/latency/memory/artifact/reproducibility 의미론) | — |
| `test_regression.py` (5) | — | 전부 | — |
| `test_credit_ledger.py` (6) | — | Credit 회계 규칙, 저장소 장부 무결성 | — |
| `test_matrix.py` (8) | — | 매트릭스·pairwise 커버리지·risk 선택 | — |
| `test_mock_adapter.py` (10) | (Mock 계약: 결정성·시나리오) | — | — |
| `test_mock_pipeline.py` (21) | — | 상태 회계·판정 규칙·결정성 | credit-safety |
| `test_config.py` (9), `test_result_model.py` (7), `test_reporter_and_safety.py` (3) | — | 리포트 내용 | 설정 검증, 모델 라운드트립, import 가드, 시크릿 스캔 |

관찰: Layer 1 테스트는 전부 **오프라인 대조**(introspection 기록·가짜 SDK)다. 실 서버 계약은 Phase 5-B 1회 관찰이 유일한 증거다. 이는 의도된 비용 구조다 — Layer 1을 실 호출로 반복 검증하면 Credit만 소모되고 새 정보는 거의 없다.

---

## 2. Pipeline 단계별 최소 Release Test Suite

파이프라인: Model → Optimization → Quantization → Conversion → Evaluation → Deployment Artifact → Quality Gate → Release

설계 원칙: 단계마다 "계약 1개 + 품질 1~2개"만 둔다. 전수 테스트가 아니라 **릴리스 판단에 필요한 최소 증거**를 정의한다.

필드 약어: Credit = 실 API Credit 필요 여부(추정은 클라이언트 상수), Mock = MockAdapter/가짜 SDK로 로직 검증 가능 여부, Gate = Release Gate 영향도(Blocker / Required-evidence / Informational).

### Stage 1 — Model (입력 모델 무결성)

**TC-M01 입력 모델 무결성과 입력 shape 계약** · Layer 1
- Objective: 최적화에 넣는 모델이 로드 가능하고 선언한 입력 shape와 일치하는가
- Preconditions: 모델 파일, 선언 shape(`[1,3,224,224]`), 3.11 venv(torch/onnx)
- Input: `graphmodule.pt` (또는 ONNX)
- Operation: `torch.load` / `onnx.checker`, 입력 텐서 shape 확인, SHA-256 기록
- Expected: 로드 성공, shape 일치, 체크섬이 baseline 레지스트리에 기록됨
- Evidence: `artifact_from_file` 결과, 레지스트리 항목
- Failure: 로드 실패 / shape 불일치 → `INPUT_VALIDATION_ERROR`
- Credit: 아니오 · Mock: 예 · Priority: **P0** · Gate: Blocker (잘못된 입력은 뒤의 모든 증거를 무효화)

### Stage 2 — Optimization (compression)

**TC-O01 automatic_compression 호출 계약** · Layer 1
- Objective: 검증된 시그니처로 1회 호출 → `completed` metadata와 산출물
- Preconditions: 인가(`--confirm-credit-use`), 키, 예산 ≥ 예비+25
- Input: TC-M01 통과 모델, ratio 0.5, framework pytorch
- Operation: `compressor_v2().automatic_compression(...)`
- Expected: status `completed`, `compressed_model_path` 존재, `compression_info.ratio == 0.5`, `model_info.input_shapes == 요청값`, 레이어 목록 비어 있지 않음
- Evidence: `execution_result.json`, `sdk_output/metadata.json`
- Failure: status ≠ completed / 스키마 불일치 → SDK error_detail 매핑 또는 `CONFIGURATION_ERROR`
- Credit: **예 (25, 실측 일치)** · Mock: 예(가짜 SDK) · Priority: P0 · Gate: Required-evidence
- 현황: **Phase 5-B에서 1회 PASS**

**TC-O02 압축 효과 — 크기·FLOPs·파라미터** · Layer 2
- Objective: 파일 크기가 줄었고, SDK 보고 FLOPs/params가 로컬 재계산과 일치하는가
- Input: 원본·압축 산출물, SDK metadata
- Operation: 파일 크기 측정; `torch.load` 후 파라미터 합산; FLOPs 카운터
- Expected: size 증가 0 % 이하(압축 단계 Gate), params 정확히 일치, FLOPs 오차 ≤ 예시 5 %
- Evidence: `metrics.model_size_mb`, `metrics.extra.sdk_*` vs `local_*`
- Failure: 증가 / 불일치 → `PERFORMANCE_REGRESSION`(크기) / `CONFIGURATION_ERROR`(보고값 불일치)
- Credit: 아니오 · Mock: 예 · Priority: P1 · Gate: Required-evidence
- 현황: 크기 PASS(−73.8 %); **5-D에서 교차 검증 완료** — params 정확 일치(torch·ONNX), FLOPs는 2×MACs 관례로 0.6–1.0 % 이내 일치

**TC-O03 압축 재현성(동일 조건 2회차)** · Layer 2
- Objective: 같은 모델·ratio·SDK 버전으로 다시 압축하면 같은 산출물이 나오는가
- Preconditions: TC-O01 산출물과 체크섬이 레지스트리에 있음
- Operation: TC-O01 재실행(별도 승인) → 체크섬·metadata·레이어 비교
- Expected: Level 3(체크섬 동일) 또는 Level 4(출력 동일) — 어느 쪽인지 **기록**
- Evidence: `Reproducibility.level`, 두 실행의 metadata diff
- Failure: 출력 불일치(Level 4 실패) → `REPRODUCIBILITY_ERROR`; 체크섬만 다르면 FUNCTIONALLY_REPRODUCIBLE로 기록(결함 아님)
- Credit: **예 (25)** · Mock: 예 · Priority: P1 · Gate: Required-evidence

### Stage 3 — Quantization

**TC-Q01 automatic_quantization 호출 계약** · Layer 1
- Objective: INT8 양자화가 검증된 시그니처로 완료되고 산출물·metadata 스키마가 맞는가
- Preconditions: ONNX 입력(압축 산출물 `sdk_output.onnx`), 캘리브레이션 npy(공식 `pickle_calibration_dataset_224x224.npy`, 3.6 MB), 인가·예산
- Operation: `quantizer().automatic_quantization(..., weight/activation INT8, metric SNR, threshold 0)`
- Expected: status completed, `quantized_model_path` 존재, `quantize_info.weight_precision == int8`
- Failure: error_detail 매핑 / 폴링 TIMEOUT → `OPTIMIZATION_FAILURE` 또는 `RUNTIME_ERROR`
- Credit: **예 (50)** · Mock: 예 · Priority: P1 · Gate: Required-evidence (양자화를 릴리스 범위에 넣는 경우)

**TC-Q02 양자화 손실 proxy** · Layer 2
- Objective: INT8 모델의 출력이 FP32 모델과 얼마나 다른가
- Operation: 로컬 ORT로 두 모델에 동일 입력 → top-1 agreement, cosine
- Expected: agreement ≥ 예시 95 % (proxy; 정확도 아님)
- Failure: 미달 → `ACCURACY_REGRESSION`(suspected cause: INT8 precision loss, 미확정)
- Credit: 아니오 · Mock: 예 · Priority: P1 · Gate: Required-evidence

### Stage 4 — Conversion

**TC-C01 convert_model 호출 계약(타깃 1개)** · Layer 1
- Objective: 선택한 타깃(OpenVINO / Intel-Xeon / FP16)으로 변환이 완료되는가
- Preconditions: 서버 `available_options` 스냅샷에서 지원 확인(Phase 5-B 응답에 존재), 인가·예산
- Operation: `converter_v2().convert_model(input=ONNX, target_framework=openvino, target_device_name=Intel-Xeon, target_data_type=FP16, wait_until_done=True)`
- Expected: status completed, `converted_model_path` 존재, `convert_task_info.framework/device_name/data_type == 요청값`
- Failure: TIMEOUT/ERROR → `CONVERSION_FAILURE`
- Credit: **예 (50)** · Mock: 예 · Priority: P1 · Gate: Required-evidence

**TC-C02 변환 산출물 포맷 유효성** · Layer 2
- Objective: 산출물이 타깃 포맷 규약에 맞는가(OpenVINO: .xml+.bin 쌍; TFLite: 플랫버퍼 로드)
- Operation: 포맷별 로컬 로더로 open (OpenVINO runtime 미설치 시 구조 검사만 수행하고 `NOT_VERIFIED` 기록)
- Expected: 로드/구조 검사 통과
- Failure: → `ARTIFACT_ERROR`
- Credit: 아니오 · Mock: 예 · Priority: P1 · Gate: Required-evidence

### Stage 5 — Evaluation

**TC-E01 Accuracy (baseline vs optimized)** · Layer 2 — 설계 상세 §3.1
- Preconditions: task 정의 + 라벨 평가셋 + 동일 전처리. **현 샘플 모델로는 불가**(task 불명)
- Expected: drop ≤ 예시 1.0 pp (project-defined)
- Failure: → `ACCURACY_REGRESSION`; 데이터 없음 → `NOT_APPLICABLE`(필수면 Gate FAIL, 결함은 UNCLASSIFIED)
- Credit: 모델 교체 시 25 · Mock: 예 · Priority: **P0** · Gate: **Blocker**

**TC-E02 Latency (baseline vs optimized, 동일 조건)** · Layer 2 — §3.2
- Expected: 증가 ≤ 예시 10 % (개선이 기대값이지만 Gate는 "악화 금지"만 강제)
- Credit: 로컬 아니오 / 타깃 convert 50 + profile 25 (모델당) · Priority: **P0** · Gate: Blocker

**TC-E03 Memory (peak, baseline vs optimized)** · Layer 2 — §3.3
- Expected: 증가 ≤ 예시 15 %
- Credit: 로컬 아니오 / 타깃 profile 결과에 포함 · Priority: P1 · Gate: Required-evidence

### Stage 6 — Deployment Artifact

**TC-A01 산출물 무결성·추적성** · Layer 2 — §4
- Objective: 배포 산출물이 존재·비어 있지 않음·구조 유효·체크섬이 기대값과 일치하며, 어떤 입력·설정·SDK로 만들어졌는지 추적 가능한가
- Expected: `checksum_valid == True`(레지스트리 기대값 대비), 메타데이터 11키 + 환경 fingerprint
- Failure: 불일치 → `ARTIFACT_ERROR`; 기대값 없음 → `NOT_APPLICABLE`
- Credit: 아니오 · Mock: 예 · Priority: **P0** · Gate: Blocker

**TC-A02 산출물 재현성 레벨 판정** · Layer 2 — §5
- Expected: Level ≥ 3 또는 Level 4 + 사유 기록
- Credit: 2회차 산출물이 있으면 아니오 · Priority: P1 · Gate: Required-evidence

### Stage 7 — Quality Gate

**TC-G01 결측 데이터는 절대 PASS가 아니다** · Layer 2
- Operation: 필수 기준 중 하나의 메트릭을 `None`으로 → Gate
- Expected: Overall FAIL, 사유에 "not available", 결함 `UNCLASSIFIED (missing data)` — 회귀로 분류되지 않음
- Credit: 아니오 · Mock: 예 · Priority: P0 · Gate: Blocker (정책 자체의 무결성)
- 현황: **구현·테스트 완료**, Phase 5-B 실 데이터로도 확인

**TC-G02 단계 프로파일 적용** · Layer 2
- Operation: 압축 단계 결과에 `compression_stage` 프로파일, 전체 증거에 `release` 프로파일 적용
- Expected: 압축 단계 PASS가 release PASS로 승격되지 않음; 리포트가 어느 프로파일인지 명시
- Credit: 아니오 · Priority: P0 · Gate: Blocker (판정 오해 방지) · 현황: 프로파일 미구현(Phase 5-D)

### Stage 8 — Release

**TC-R01 Baseline vs Current 회귀 비교(실 데이터)** · Layer 2
- Operation: 레지스트리 baseline 실행 vs 현재 실행 → `RegressionComparator`
- Expected: REGRESSION 0건, NEW/MISSING 설명됨
- Credit: 2회차 데이터 필요(TC-O03과 공유) · Priority: P1 · Gate: Required-evidence

**TC-R02 Credit 회계 일치** · Layer 2
- Operation: 각 실 작업 전후 잔액 조회 → 장부 actual, 추정과 차이 보고
- Expected: 장부 used == Σactual, 추정과의 차이는 기록(불일치 자체는 결함이 아니라 **문서 수정 대상**)
- Credit: 작업에 포함 · Priority: P1 · Gate: Informational (예산 통제)
- 현황: compression 1건 PASS(25 = 25)

---

## 3. Accuracy / Latency / Memory 검증 설계

### 3.1 Accuracy

| 질문 | 설계 |
|---|---|
| baseline 결과는 무엇이어야 하는가 | **같은 평가셋·같은 전처리·같은 런타임**으로 측정한 원본 모델의 task metric. SDK가 제공하지 않으므로 QA가 측정·기록한다. 측정 조건(데이터셋 식별자·샘플 수·전처리 해시·런타임 버전)을 메트릭과 함께 저장해야 비교가 성립한다 |
| optimized와의 비교 | `drop_pp = (baseline − optimized) × 100` (절대 퍼센트포인트, 구현됨). 개선은 항상 PASS |
| threshold 정의 위치 | `configs/quality_gate.yaml` — **project-defined example** `max_drop_percent: 1.0`. Nota 공식 기준 미확인. task별 override 허용(`accuracy.by_task: {classification: 1.0, detection: 1.5, segmentation: 1.0}` 형태 제안) |
| task 차이 | metric 이름을 모델 설정에 둠(`accuracy_metric: top1 / mAP50 / mIoU`, 이미 `models.yaml`에 존재). 비교는 항상 **같은 metric 이름** 간에만 허용하고, 다르면 `NOT_COMPARABLE` |
| 실측값이 없을 때 | `metrics.accuracy = None` → 기준 `NOT_APPLICABLE` → 필수면 Gate FAIL, 결함 `UNCLASSIFIED (missing data)`. **추정값·Mock값으로 채우지 않는다** (구현됨) |
| 현 샘플 모델의 한계 | `graphmodule.pt`는 task/라벨이 불명. 가능한 것은 **출력 일치도 proxy**(원본 vs 압축: top-1 agreement, cosine similarity, max |Δ|)이며, 리포트에는 `metrics.extra.output_agreement_top1` 등 **별도 이름**으로 기록하고 accuracy 필드에는 넣지 않는다 |
| 진짜 정확도를 얻는 경로 | 라벨셋이 공개된 소형 분류 모델(예: torchvision 사전학습 모델 + 공개 검증 subset)로 TC-M01→TC-O01(25 Credit)→TC-E01 재실행. 데이터셋 라이선스·용량은 실행 전 확인 |

### 3.2 Latency

| 질문 | 설계 |
|---|---|
| 동일 조건 필요성 | **필수**. 같은 머신·같은 런타임 버전·같은 스레드 수·같은 입력 텐서·같은 배치. baseline과 optimized를 **같은 프로세스에서 교대로** 측정해 시스템 상태 차이를 줄인다 |
| warm-up | 예: 10회 (JIT/메모리 할당 안정화). 측정에서 제외 |
| 측정 횟수 | 예: 100회 (수치는 example policy). 너무 적으면 노이즈, 너무 많으면 시간 |
| 대표값 | **median**을 Gate 기준으로, **p95**를 함께 기록(꼬리 지연은 배포 품질 지표). mean은 이상치에 취약해 보조로만 |
| 고정 요소 | `device`(로컬: CPU 모델명), `runtime`(onnxruntime 버전), `backend`(EP = CPUExecutionProvider), `intra_op_num_threads`, 입력 shape → 전부 `Environment.extra`에 기록해야 비교 유효 |
| 로컬 vs 타깃 | 로컬 ORT 측정은 **개발 머신 수치**로 라벨링(`latency_source: local_onnxruntime_cpu`). 타깃 디바이스 수치는 Profiler(`profile_result.latency`, int, 단위 미검증)로 별도 필드. 둘을 섞어 비교하지 않는다 |
| 노이즈 vs 회귀 | 반복 측정 분산(p95−median, 또는 반복 실행 간 median 변동)을 기록하고, 임계값 위반이 분산 범위 안이면 `WARN`으로 격하하는 규칙을 **제안**(현재 미구현; 통계적 유의성 주장 없음) |
| 실측값이 없을 때 | `None` → `NOT_APPLICABLE` (구현됨) |

### 3.3 Memory

| 질문 | 설계 |
|---|---|
| peak vs average | **peak**. 배포 가능 여부는 최대치가 결정한다. average는 보조 기록 |
| CPU/GPU 분리 | 분리. 로컬은 CPU RSS만(GPU 없음, `memory_gpu_mb = None`). Profiler는 `memory_footprint_cpu/gpu`를 제공하므로 필드 분리 유지 |
| 동일 환경 비교 | 필수. 측정 방법(RSS 증가분 vs 할당기 통계)을 기록. 프로세스 기준선(모델 로드 전 RSS)을 빼서 모델 귀속 메모리만 비교 |
| 실측값이 없을 때 | `None` → `NOT_APPLICABLE` |
| threshold | `max_increase_percent: 15.0` — project-defined example |

---

## 4. Artifact Validation 설계

### 4.1 Artifact 레코드 (기존 `Artifact` + 레지스트리 메타데이터)

| 필드 | Baseline Artifact | Optimized Artifact | 비고 |
|---|---|---|---|
| path, exists, size_bytes, format | ✓ | ✓ | 구현됨 |
| sha256 | ✓ (입력 모델: `source_model_sha256` 기록됨) | ✓ (`checksum_sha256` 기록됨) | 구현됨 |
| expected_sha256 | 레지스트리에서 | 레지스트리에서 | **미구현** — Phase 5-D |
| model format 구조 검증 결과 | torch.load / onnx.checker | 동일 | 미구현 |
| sdk_version, operation, configuration_key | ✓ | ✓ | 구현됨 |
| input_shape, compression_ratio | ✓ (metadata) | ✓ | 구현됨 |
| timestamp | ✓ | ✓ | 구현됨 |
| environment fingerprint | python/platform/sdk + (추가) torch/ort/cpu/threads | 동일 | 부분 구현 |

### 4.2 다섯 가지 서로 다른 진술

| 진술 | 의미 | 현재 상태(Phase 5-B) |
|---|---|---|
| Artifact **exists** | 경로에 파일이 있고 크기 > 0 | ✓ 확인 |
| Artifact **readable** | 포맷 로더가 열 수 있음 | 간접 증거만(SDK가 ONNX export에 성공) — 미검증 |
| Artifact **structurally valid** | 그래프/텐서가 포맷 규약에 맞고 I/O shape가 선언과 일치 | 미검증 |
| Artifact **checksum recorded** | SHA-256이 기록됨 | ✓ 기록 |
| Artifact **reproducibility verified** | 동일 조건 재실행 산출물과 체크섬(또는 출력)이 일치 | **미검증** (runs = 1) |

"파일이 생성되었다"(1·4)와 "동일 조건에서 재현된다"(5)는 다른 진술이며, 리포트는 각각 별도 상태로 표기한다.

### 4.3 Baseline 레지스트리 승격 규칙 (Phase 5-D 구현 예정)

1. 어떤 `(operation, configuration_key, sdk_version)`에 대해 첫 COMPLETED 산출물의 체크섬·메타데이터·메트릭은 `reports/baselines/`에 **후보(candidate)** 로 저장된다 — 이 시점에는 여전히 `checksum_valid = None`.
2. 2회차 실행이 Level 3 또는 4를 만족하면 후보는 **baseline**으로 승격되고, 이후 실행은 `expected_sha256`을 자동 조회한다.
3. 승격 전에는 어떤 리포트도 artifact 기준을 PASS로 표기하지 않는다.

---

## 5. Reproducibility Test 설계 (Level 0–4)

| Level | 진술 | 비용 | 의미와 한계 | Phase 5-B 상태 |
|---|---|---|---|---|
| 0 | 산출물이 존재한다 | 0 | 가장 약한 증거. 내용에 대해 아무것도 말하지 않음 | ✓ |
| 1 | metadata/config가 요청과 동일하다 (ratio, input_shapes, method, sdk_version) | 0 | 서버가 요청을 바꾸지 않았음. 두 실행 간 비교의 전제 조건 | ✓ (육안) → 자동 검사 예정 |
| 2 | 동일 조건 재실행이 **동일한 구조의 output**을 만든다 (레이어 수·채널 수·파라미터 수 일치) | 실 재실행 1회(compression 25) | 알고리즘이 결정적인지. 체크섬이 달라도 구조가 같을 수 있음 | ✗ |
| 3 | 체크섬이 동일하다 (bitwise) | Level 2에 포함 | 가장 강한 파일 수준 증거. 서버 직렬화에 타임스탬프 등이 섞이면 결정적 알고리즘이어도 실패할 수 있음 → 실패 시 Level 4로 판정 | ✗ |
| 4 | semantic equivalence — 두 산출물이 동일 입력에 **동일(허용 오차 내) 출력** | 2회차 산출물 존재 시 0 (로컬 ORT) | 배포 관점에서 실제로 중요한 것. 체크섬이 달라도 여기서 통과하면 `FUNCTIONALLY_REPRODUCIBLE` | ✗ |

프레임워크의 `ReproducibilityLevel`과의 대응: Level 3 → `BITWISE_REPRODUCIBLE`, Level 4(3 실패) → `FUNCTIONALLY_REPRODUCIBLE`, Level 4 실패 → `NOT_REPRODUCIBLE`, 2회차 없음 → `NOT_VERIFIED`(현재).

Phase 5-B는 Level 0–1까지 검증했고(5-D에서 Level 1을 `configuration_consistency`로 자동화), **Level 2부터 실 재실행(E1, 25 Credit)이 필요**하다. Level 4의 비교 자체는 로컬에서 0 Credit이며, 5-D의 출력 비교기(`compare_outputs`)가 그대로 쓰인다.

---

## 6. Credit 그룹과 실행 계획

| 그룹 | 내용 | Credit |
|---|---|---|
| **A. 0 Credit** | 스키마·설정 검증, 산출물 구조 검증, 체크섬 재계산, FLOPs/params 재계산, 로컬 ORT 평가(일치도·latency·memory), Gate 프로파일, 레지스트리, 회귀 로직, Mock E2E, 리포트 | 0 |
| **B. 최소화 가능** | 서버 지원 조합(기존 `available_options` 스냅샷 재사용), 라벨셋 있는 모델로 교체(모델 1개만), 환불 정책(자연 발생 시 기록) | 0~25 |
| **C. 실 API 필수** | E1 재현성 2회차(25), E2 압축 모델 convert+profile(75), E3 원본 convert+profile(75), E4 INT8 quantization(50) | 225 |

권고: **4회, 약 225 Credit, 잔여 250**. 실험마다 별도 승인·예산 점검·단일 실행. 상세 근거는 `quality_gap_analysis.md` §6.

---

## 7. 우선순위와 Roadmap

| 우선순위 | 정의 | 해당 TC |
|---|---|---|
| **P0** Release blocker | 없으면 릴리스 판단 자체가 불가 | TC-M01, TC-O01(완료), TC-E01, TC-E02, TC-A01, TC-G01(완료), TC-G02 |
| **P1** High-value | 판단의 신뢰도를 크게 올림 | TC-O02, TC-O03, TC-Q01, TC-Q02, TC-C01, TC-C02, TC-E03, TC-A02, TC-R01, TC-R02 |
| **P2** Regression/robustness | 반복 실행·노이즈·드리프트 대응 | latency 분산 규칙, 서버 옵션 드리프트 감시, error/stopped 실 관찰 |
| **P3** Nice-to-have | 포트폴리오 가독성 | HTML에 단계 프로파일 표시, 레지스트리 뷰어 |

| Phase | 내용 | Credit |
|---|---|---|
| **5-C** (본 문서) | READ-ONLY Quality Gap Analysis, 전략·Gate 문서 | 0 |
| **5-D** 0-Credit local validation — **완료** | Gate 프로파일(`compression`/`local_eval`/`release`), baseline 레지스트리(`candidate` 등록), 산출물 구조 검증(PT 안전 로드 우선 + 신뢰 목록 기반 전체 로드, ONNX checker), params·FLOPs 독립 검증(params 정확 일치, FLOPs = 2×MACs 관례 관찰), 프로세스 격리 로컬 ORT 평가기(latency median/p95, peak RSS, 출력 동등성 proxy), 환경 fingerprint(시크릿 필터), `reports/local_eval/` 리포트. 결과: `compression` PASS, `local_eval` FAIL(proxy cosine 0.825), `release` FAIL(증거 부족) | 0 |
| **5-E** 최소 실 실험 | E1 → (승인) E2 → (승인) E3 → (승인) E4. 각 실행 후 레지스트리 승격·회귀 비교·장부 기록 | ≤ 225 |
| **5-F** E2E / Release Gate | 실 baseline·optimized 데이터로 `release` 프로파일 평가, 회귀 리포트, README에 "실측 범위" 명시. Accuracy는 모델 교체 결정에 따라 Blocker 유지 또는 라벨셋 모델로 재실행 | 0 (추가 모델 시 +25) |

Phase 명칭은 기존 계획(5-A/5-B)을 잇되, 원래 사양의 Phase 12(Profiler)·13(Artifact)·14(Reproducibility)·15(Regression) 내용이 5-D/5-E/5-F에 흡수된다.

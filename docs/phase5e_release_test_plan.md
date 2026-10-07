# Phase 5-E Release Test Plan — YOLOv8n + COCO

> 범위: YOLOv8n 단일 모델, 단일 configuration(`yolov8n | intel_xeon_w2233 | onnxruntime | default | automatic_compression`), 공식 Nota fx 경로. 임계값은 전부 **PROJECT-DEFINED EXAMPLE THRESHOLD**. Credit 열의 숫자는 SDK 클라이언트 사전 체크 상수(automatic_compression 25는 5-B에서 실 차감 일치 확인).
> 상태 표기: **DONE**(5-E 준비에서 0 Credit으로 완료), **READY**(즉시 실행 가능), **PENDING**(선행 조건 있음), **CREDIT**(실 API 필요).

---

## 1. Test Matrix (5-E 범위)

| 축 | 값 | 비고 |
|---|---|---|
| Model | yolov8n (v8.4.0 weights, sha `f59b3d83…`) | fx body `model_fx.pt` + head meta |
| Device | intel_xeon_w2233 (nominal; 로컬 측정은 개발 머신 CPU) | 타깃 Profiler는 E5-3 선택 |
| Runtime | onnxruntime 1.30.0 (local) | |
| Backend | default (CPUExecutionProvider) | |
| Optimization | automatic_compression ratio 0.5 | 단일 |
| Dataset | COCO128(smoke) / COCO val2017(release, pending) | |

## 2. Test Cases

### MODEL

| TC | Name | Objective | Precondition | Input | Operation | Expected | Evidence | Failure → Defect | Credit | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| TC-Y8-001 | YOLOv8n model acquisition | 공식 출처·버전 식별 가능한 가중치 확보 | 네트워크 | assets v8.4.0 URL | 다운로드, 크기 검증 | 6,549,796 B, provenance.json 기록, git 미추적 | `outputs/models/yolov8n.provenance.json` | 크기/출처 불일치 → INPUT_VALIDATION_ERROR | 0 | **DONE** |
| TC-Y8-002 | model fingerprint | SHA-256·크기·출처·라이선스 기록, 신뢰 목록 등록 | 001 | yolov8n.pt | `compute_sha256`, 신뢰 목록 대조 | sha `f59b3d83…`, trusted=True | baseline.json `model.weights` | 신뢰 목록 불일치 → 전체 unpickle 금지(NOT_APPLICABLE) | 0 | **DONE** |
| TC-Y8-003 | baseline structural validation | 체크포인트 로드·구조 인식, fx export 산출물 구조 | 002 | yolov8n.pt, model_fx.pt | `validate_pt`(checkpoint / graph_module) | PASS, params 3,157,200 / 3,157,184 | baseline.json, fx_export_precheck.json | FAIL → ARTIFACT_ERROR | 0 | **DONE** |

### DATA

| TC | Name | Objective | Precondition | Input | Operation | Expected | Evidence | Failure → Defect | Credit | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| TC-Y8-010 | dataset preparation | 평가셋 확보·버전·split·경로 기록 | 네트워크 | coco128.yaml / coco.yaml | ultralytics 자동 다운로드 → `datasets/`(gitignore) | coco128: 128 images/929 instances ✓; val2017: 5,000 images | baseline.json `accuracy.dataset_*` | 누락/손상 → ENVIRONMENT_ERROR | 0 | coco128 **DONE** / val2017 **PENDING(승인)** |
| TC-Y8-011 | preprocessing validation | baseline·compressed가 동일 전처리를 쓰는지 | 010 | val 설정 | imgsz 640 letterbox, conf 0.001, IoU 0.7 고정; 설정 해시 기록 | 두 평가의 설정 동일 | 결과 JSON `preprocessing`, `conf_iou_settings` | 불일치 → CONFIGURATION_ERROR | 0 | **READY**(압축본 평가 시 재확인) |

### BASELINE

| TC | Name | Objective | Precondition | Input | Operation | Expected | Evidence | Failure → Defect | Credit | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| TC-Y8-020 | baseline accuracy | mAP50-95(primary), mAP50/mAP75/P/R | 010 | yolov8n.pt + data | `model.val(imgsz 640, cpu)` | 측정값 기록(판정 없음) | coco128: 0.4437 / 0.6011 / 0.4722 / 0.6286 / 0.5290 | 측정 실패 → N/A(PASS 아님) | 0 | coco128 **DONE** / val2017 **PENDING** |
| TC-Y8-021 | baseline latency | 격리 프로세스 ORT median/p95 | ONNX export | yolov8n_640_opset13.onnx | warm-up 10, 100회 | 기록 | median 15.53 ms, p95 29.56 ms | — | 0 | **DONE** |
| TC-Y8-022 | baseline memory | peak RSS 증가분 | 동일 | 동일 | psutil RSS | 기록; GPU N/A | 107.13 MB | — | 0 | **DONE** |
| TC-Y8-023 | baseline params/FLOPs | 구현 보고값 vs 독립 계산 | ONNX | 동일 | `verify`, `verify_flops` | params Δ 0.17 % PASS; FLOPs 8.855 G vs 2×MACs 8.744 G PASS; 미커버 op 기록 | baseline.json `params_flops` | 불일치 → CONFIGURATION_ERROR | 0 | **DONE** |

### OPTIMIZATION

| TC | Name | Objective | Precondition | Input | Operation | Expected | Evidence | Failure → Defect | Credit | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| TC-Y8-030 | NetsPresso compatibility precheck | 업로드 전 입력이 compressor 계약(fx GraphModule .pt)에 맞는지 — 2025 실패 유형 제거 | 002 | yolov8n.pt → 포크 `export_netspresso` | `model_fx.pt` + head meta 생성, `validate_pt` graph_module, forward shape | **READY**: 12,846,561 B, [1,144,80/40/20] — 단, 이 산출물은 R1에서 **기능 비동등**으로 확인되어 압축 입력에서 **제외** | fx_export_precheck.json | 비-fx → INPUT_VALIDATION_ERROR(업로드 차단) | 0 | **DONE (READY, 입력 대체됨 → TC-Y8-R1)** |
| TC-Y8-030b | **R1 upstream traceable fx export + functional equivalence** | 압축 입력이 upstream YOLOv8n과 같은 함수인지 | 002, 030 분석 | yolov8n.pt → upstream 8.4.173 + wrapper(`r1-upstream-body-wrapper-v1`) | `yolov8_r1_traceable_export.py` + `yolov8_r1_fork_crosscheck.py` | **PASS**: `yolov8n_fx_r1/model_fx.pt` 12,846,691 B sha `d8e761da…`; pre-NMS bitwise; 6=6 boxes; mAP 0.44369 = 0.44369; torch 2.0.1 load PASS; 2회 export 동일 | `*_r1/r1_result.json`, `*_r1_torch201_load/`, `*_r1_fork_crosscheck/` | 비동등 → INPUT_VALIDATION_ERROR(압축 입력 거부) | 0 | **DONE (PASS)** |
| TC-Y8-031 | automatic compression | 공식 경로 산출물로 1회 압축 | **030b(R1 PASS)**, 승인, 예산(≥ 125 = 100 예비 + 25) | **`outputs/models/yolov8n_fx_r1/model_fx.pt`**, input_shapes [{1,3,[640,640]}], ratio 0.5, framework pytorch | `run_e5_1_experiment.py --confirm-credit-use` → `run_real_netspresso.py …yolov8n_fx_r1/model_fx.pt --input-shape 1,3,640,640` | status completed, metadata ratio 0.5, 장부 actual 기록 | `reports/real_runs/20261005T040009Z_automatic_compression/`, `reports/yolov8_e5_1/20261005T040009Z/` | error_detail 매핑(MODEL_COMPATIBILITY_ERROR 등); **재시도 금지** | **25 (실 차감 25 확인, 475→450)** | **DONE (COMPLETED, E5-1)** |

### OPTIMIZED MODEL

| TC | Name | Objective | Precondition | Input | Operation | Expected | Evidence | Failure → Defect | Credit | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| TC-Y8-040 | artifact integrity | 존재·크기·SHA·추적성; 기대 체크섬은 candidate 단계면 N/A | 031 | compressed .pt(+SDK onnx) | `artifact_from_file`, registry | 기록, checksum_valid None(첫 산출물) | e5_1_result.json: sha `21d8cbd1…` 3,693,813 B, 실행 레코드 SHA 일치, registry candidate | 없음/빈 파일 → ARTIFACT_ERROR | 0 | **DONE (PASS)** |
| TC-Y8-041 | structural validation | 압축본 GraphModule 로드(신뢰: SDK 산출), ONNX checker | 040 | 압축 .pt/.onnx | `validate_pt/onnx` | PASS | e5_1_result.json: GraphModule PASS, forward `[1,144,80/40/20]`, ONNX checker PASS | FAIL → ARTIFACT_ERROR | 0 | **DONE (PASS)** |
| TC-Y8-042 | params/FLOPs | SDK 보고 vs 독립(2×MACs 관례 확인됨) | 041 | metadata.results + ONNX | `verify`, `verify_flops` | PASS within 0.5 % / 10 % | 882,996 = 882,996 params; FLOPs 3.875 G vs 2×MACs 3.832 G; 감소 −72.0 % / −56.0 % | FAIL → CONFIGURATION_ERROR | 0 | **DONE (PASS)** |
| TC-Y8-043 | accuracy regression | baseline mAP50-95 − compressed mAP50-95 ≤ threshold | 041 + head 재부착 helper(**구현됨**, CG-002) + fx body 정합성 블로커 **해소됨(R1, 해소 문서 §11)** | 압축 body + R1 head meta → `R1DecodeHead`(R1 경로) → upstream val(동일 data·rect=True) | drop_pp 계산, Gate `accuracy` | 재학습 전 **FAIL 예상**(기록·분류), 재학습 후 재평가 | e5_1_result.json: **0.44369 → 0.0, drop 44.369 pp** → release FAIL | ACCURACY_REGRESSION(suspected: pruning w/o fine-tuning, MEDIUM, 미검증) | 0 | **DONE (측정, 예상대로 FAIL)** |
| TC-Y8-044 | latency regression | 동일 ORT 조건 median 증가 ≤ 10 %(example) | 압축본 ONNX(head 포함) | 격리 측정 | 기록·판정 | 23.78 → 16.52 ms (−30.5 %) PASS | PERFORMANCE_REGRESSION | 0 | **DONE (PASS)** |
| TC-Y8-045 | memory regression | peak RSS 증가 ≤ 15 %(example) | 동일 | 동일 | 기록·판정 | 107.08 → 93.36 MB (−12.8 %) PASS | MEMORY_REGRESSION | 0 | **DONE (PASS)** |

### REPRODUCIBILITY

| TC | Name | Objective | Precondition | Input | Operation | Expected | Evidence | Failure → Defect | Credit | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| TC-Y8-050 | configuration consistency | 요청 ratio/shape/framework == metadata | 031 | metadata.json | 자동 비교(Level 1) | match | ratio 0.5 / `{1,3,[640,640]}` / pytorch 일치 | CONFIGURATION_ERROR | 0 | **DONE** |
| TC-Y8-051 | environment fingerprint | 두 평가(baseline/compressed)의 환경 동일성 | — | fingerprint | 비교 | 동일 또는 차이 기록 | environment.json | 차이 → 비교 무효(NOT_COMPARABLE) | 0 | baseline **DONE** |
| TC-Y8-052 | artifact reproducibility | Level 3(정규화 체크섬) / Level 4(출력 동등성) | 2회차 산출물 | 재export 또는 재압축 | 레벨 판정, 레지스트리 승격 여부 | ONNX export: metadata 제외 시 bit-identical(**DONE**); 압축 2회차는 CREDIT 25 | registry, probe 기록 | NOT_REPRODUCIBLE → REPRODUCIBILITY_ERROR | 0 / 25 | 부분 DONE |

### RELEASE

| TC | Name | Objective | Precondition | Input | Operation | Expected | Evidence | Failure → Defect | Credit | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| TC-Y8-060 | quality gate | 프로파일 3종 평가, 결측≠PASS | 040–052 | ExecutionResult | `QualityGate(profile)` | compression PASS 기대; local_eval/release는 증거 기반 | **compression PASS / local_eval FAIL(proxy 0.984) / release FAIL(accuracy, checksum N/A, runs=1)** | — | 0 | **DONE** |
| TC-Y8-061 | release decision | R1–R10 충족 여부 + 회귀 리포트 | 060 | baseline run vs compressed run | `RegressionComparator`, Release 조건표 | PASS/FAIL + 사유 | **Release FAIL**, 권고 HOLD (`docs/phase5e_e5_1_experiment.md`) | — | 0 | **DONE (FAIL)** |

제거/수정한 TC: 없음(원안 유지). 단, 043은 "head 재부착 helper"를 선행 조건으로 명시했고, 052는 ONNX export 정규화 체크섬 발견을 반영해 Level 3 정의를 보강했다.

### CONDITIONAL GO RESOLUTION (TC-Y8-CG-*, 0 Credit, 구현: `tests/unit/test_detection_eval.py` + 스크립트)

| TC | Name | 검증 방법 | 상태 |
|---|---|---|---|
| CG-001 | head metadata load | `load_head_meta` 필수 키·타입·nl/stride 일치·JSON 오류 | **PASS** |
| CG-002 | GraphModule + head re-attach | precheck `fx_is_graph_module` + 포크 `DetectionModel_netspresso` 실행(decoded `[1,84,8400]`) | **PASS** (identity fixture) |
| CG-003 | input shape validation | 4-D, 3ch, max stride 배수 | **PASS** |
| CG-004 | output schema validation | body `[1,144,80/40/20]`, decoded `[1,84,8400]` | **PASS** |
| CG-005 | COCO128 evaluation pipeline | 포크 `DetectionValidator` 실행 → `DetectionMetrics` | **PASS** (파이프라인) / 절대값은 해소 문서 §4 블로커 |
| CG-006 | mAP50-95 calculation | `DetectionMetrics.primary` → `Metrics.accuracy` 매핑 | **PASS** |
| CG-007 | accuracy delta | absolute_delta / drop_pp / relative_delta_percent, secondary | **PASS** |
| CG-008 | accuracy N/A semantics | N/A·NOT_COMPARABLE은 PASS/FAIL로 변환되지 않음 | **PASS** |
| CG-009 | identity fixture evaluation | fixture vs same-evaluator baseline drop −0.001 pp, criterion PASS | **PASS** |
| CG-010 | proxy vs accuracy separation | proxy PASS여도 accuracy NOT_APPLICABLE | **PASS** |
| CG-011 | normalized ONNX checksum | metadata만 다르면 normalized 동일 | **PASS** |
| CG-012 | raw checksum preservation | raw 보존·구분 | **PASS** |
| CG-013 | reproducibility metadata | baseline.json의 SHA/환경/시드/입력/데이터 기록 | **PASS** |
| CG-014 | precheck READY/BLOCKED/UNSUPPORTED/NOT_VERIFIED | ONNX 입력→UNSUPPORTED(2025 유형), 비신뢰→NOT_VERIFIED, schema 불일치→BLOCKED | **PASS** |
| CG-015 | zero-credit execution guard | 스크립트에 SDK import·env 접근 없음, ledger 25/475/1 | **PASS** |

### R1 — UPSTREAM TRACEABLE FX EXPORT (TC-Y8-R1-*, 0 Credit, 구현: `tests/unit/test_yolov8_r1.py` + `scripts/yolov8_r1_*.py`)

| TC | Name | 검증 방법 | 결과 |
|---|---|---|---|
| R1-001 | upstream baseline reproducibility | `YOLO(yolov8n.pt).val(coco128)` 재측정 vs 저장 baseline | **PASS** 0.44369 = 0.44369 |
| R1-002 | traceability wrapper creation | boundary 계약(`fx_traceability.TRACEABILITY_BOUNDARY`), 그래프 leaf-clean, 직접 trace 실패 근거 | **PASS** (TraceError 기록, 230 nodes, torch.nn leaf만) |
| R1-003 | FX export | manifest 필수 키(source SHA/version, code version, patch id, input, fx/torch/python version, timestamp, artifact SHA/size) | **PASS** |
| R1-004 | FX artifact structural validation | `validate_pt` graph_module + `precheck_fx_bundle` READY | **PASS** |
| R1-005 | head re-attach | upstream helper `R1DecodeHead` + 공식 포크 `DetectionModel_netspresso` 둘 다 | **PASS** (포크 재부착 decoded rel 1.5e-7) |
| R1-006 | input schema | 4-D, 3ch, max stride 배수 | **PASS** |
| R1-007 | output schema | body `[1,144,80/40/20]`, decoded `[1,84,8400]` | **PASS** |
| R1-008 | pre-NMS output equivalence | raw boxes/scores + decoded, cosine/max abs/rel L2; 임계값은 noise floor 측정 후 명시적으로 전달 | **PASS** bitwise (noise floor rel 4.4e-7) |
| R1-009 | decoded detection equivalence | NMS 후 box count/좌표/conf/class 매칭 | **PASS** 6=6, 194=194, Δ 0 |
| R1-010 | COCO128 mAP comparison | 같은 평가기·rect=True, `compare_detection_accuracy` | **PASS** drop 0.0 pp |
| R1-011 | torch 2.0.1 load | ultralytics 없는 torch 2.0.1 venv에서 로드·forward | **PASS** max |Δ| 6.9e-5 |
| R1-012 | raw/normalized checksum | 2회 export raw SHA 동일(정규화 불필요) | **PASS** |
| R1-013 | zero-credit guard | 스크립트 SDK import/env 접근 없음, 레코드 credit 0, ledger 불변 | **PASS** |
| (R2) | fork divergence record | 레이어별 첫 분기 layer_04, 원인 probe VERIFIED, 옛 포크 body 비동등 | **PASS** |

### E5-1 — SINGLE REAL COMPRESSION + LOCAL VALIDATION (TC-Y8-E5-*, 25 Credit 1회, 구현: `tests/unit/test_e5_1.py`, `scripts/run_e5_1_experiment.py`, `scripts/yolov8_e5_1_local_validation.py`)

| TC | Name | 검증 방법 | 결과 |
|---|---|---|---|
| E5-001 | exact R1 input SHA enforcement | `verify_input_artifact`: SHA·크기·금지 경로(옛 포크 산출물) | **PASS** |
| E5-002/003 | wrong SHA / precheck not READY → no API call | `authorize_single_real_operation` 하드 스톱 사유 | **PASS** (dry-run 레코드 `…T035912Z/e5_1_precheck.json`) |
| E5-004/005 | single execution guard, no retry | `operations_executed == 1`, 호출 로그 1회, 장부 prior-ops 가드 | **PASS** |
| E5-006 | result artifact validation | pt GraphModule·forward·schema, ONNX checker, 체인 SHA | **PASS** |
| E5-007/008 | params / FLOPs | SDK vs torch numel / 2×MACs | **PASS** |
| E5-009 | head re-attach | R1DecodeHead `[1,84,8400]` | **PASS** |
| E5-010 | COCO128 comparison | 같은 평가기·rect·conf/IoU, drop_pp 산술 | **PASS** (측정값 0.0, drop 44.37 pp) |
| E5-011 | proxy ≠ accuracy | proxy 값이 accuracy에 쓰이지 않음, release에서 optional | **PASS** |
| E5-012 | Quality Gate behaviour | 실행 PASS와 release FAIL 분리, 결측 규칙 유지 | **PASS** |
| E5-013 | credit delta accounting | 475→450 관측 = 25, 장부 전이 1 entry | **PASS** |
| E5-014 | secret redaction | 레코드·HTML·SDK 로그에 키/절대경로 없음 | **PASS** |
| E5-015 | traceability completeness | 10-링크 체인 complete | **PASS** |

### FINE-TUNING RECOVERY (TC-Y8-FT-*, 0 Credit, 구현: `tests/unit/test_e5_1_finetune.py`, `scripts/yolov8_e5_1_finetune.py`, `framework/evaluation/recovery.py`)

| TC | Name | 검증 방법 | 결과 |
|---|---|---|---|
| FT-001/002/003/004 | three-way 산술, 복구 분류, 다음 단계 A–E, config/아키텍처/추세 헬퍼 | 순수 함수 테스트 | **PASS** |
| FT-005 | E5-1 후보 로드 + source SHA 추적 | 레코드: identity PASS, E5-1 파일 불변 | **PASS** |
| FT-006 | 산출물·config 검증 | GraphModule, `[1,84,8400]`, params 882,996, 아키텍처 불변 | **PASS** |
| FT-007 | 3-way 비교 일관성 | 레코드 값 = 재계산 값, 분류 일치 | **PASS** (B: 0.44369 / 0.0 / 0.28965, 회복 65.28 %, PARTIAL_RECOVERY) |
| FT-008 | proxy ≠ accuracy | release에서 proxy optional, 값 분리 | **PASS** |
| FT-009 | Quality Gate | compression PASS / local_eval FAIL / release FAIL, 결측 규칙 유지 | **PASS** |
| FT-010 | 0 Credit + 역사 불변 | 장부 50/450/2 전후 동일, E5-1 레코드 불변, SDK import 없음 | **PASS** |
| FT-011 | secrets/paths | 레코드·HTML에 키/절대경로 없음 | **PASS** |
| FT-012 | phase 연결 | B의 source sha == A 산출물 sha | **PASS** |

### INDEPENDENT VALIDATION (TC-Y8-VAL-*, 0 Credit, 학습 없음, 구현: `tests/unit/test_val2017_eval.py`, `scripts/yolov8_val2017_eval.py`)

| TC | Name | 검증 방법 | 결과 |
|---|---|---|---|
| VAL-001 | generalization gap 산술 | smoke − unseen (pp) | **PASS** |
| VAL-002 | unseen recovery 분류 CASE A–D | drop/near-pass/overfit(excess gap)/partial 규칙 | **PASS** |
| VAL-003 | next step 단일 선택, NetsPresso HOLD 고정 | 5개 라벨 매핑 | **PASS** |
| VAL-004 | 레코드 일관성 | 5,000장, 오염 0, SHA 체인, 산술 재계산 = 레코드, release FAIL(runs=1) | **PASS** |
| VAL-005 | proxy ≠ accuracy | 값 분리 | **PASS** |
| VAL-006 | 0-Credit 가드·시크릿 | 키 존재 시 abort(값 미독), 학습 코드 없음, 레코드에 키/절대경로 없음 | **PASS** |

**최종 판정(2026-10-05, `reports/yolov8_e5_1_finetune/20261005T053121Z_val2017/`)**: val2017 mAP50-95 baseline 0.36804 / E5-1 0.0 / Phase C 0.00917 → **VAL2017_OVERFIT**, compression PASS / local_eval PASS / **release FAIL**, 결함 `VAL2017-release-ACCURACY_REGRESSION` [HIGH], baseline promotion 없음, **NetsPresso 추가 실행 HOLD**. TC-Y8-061(release decision)의 최종 결과는 FAIL이다.

## 3. Release Criteria (요약)

`release_quality_gate.md` §4의 R1–R10을 그대로 적용한다. 5-E 특수 사항: R5(accuracy)는 **mAP50-95, 같은 data yaml·split·전처리·conf/IoU**에서만 비교; coco128 측정은 R5의 증거로 쓰지 않는다(smoke). 압축 직후 R5 FAIL은 예상 결과이며, 재학습 산출물에 대해 R1–R10을 다시 평가한다.

## 4. Credit 전략 (요약)

E5-0 0 Credit(완료) → E5-1 25(단일 압축, 승인 필요) → E5-2 0(로컬 전 단계) → E5-3 선택 75(convert+profile, 공식 YOLOv8 워크플로우 밖). 한 번에 하나, 결과 분석 후 다음 결정. 예비 100 유지(잔여 475 → E5-1 후 450).

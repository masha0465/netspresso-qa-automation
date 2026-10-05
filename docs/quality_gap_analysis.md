# Quality Gap Analysis — Phase 5-C

> 질문: "Phase 5-B의 첫 실 실험 결과를 기준으로, **무엇이 부족해서** 지금의 결과가 Release Quality Gate를 통과하지 못하는가?"
> 이 문서는 *부족한 것*만 다룬다. 무엇을 어떻게 테스트할지는 [`test_strategy.md`](test_strategy.md), 어떤 조건이면 PASS인지는 [`release_quality_gate.md`](release_quality_gate.md)에 있다.
>
> 작성 기준: commit `0486ee3` (Phase 5-B), 실 실행 1회, Credit 25 사용 / 475 잔여. 이 문서를 작성하는 동안 NetsPresso API는 호출하지 않았다.

---

## 1. 분석 대상: Phase 5-B 실 실행 1회의 사실

출처: `reports/real_runs/20261005T010356Z_automatic_compression/` (execution_result.json, case_result.json, sdk_output/metadata.json, sdk_log.txt)

| 항목 | 값 | 값의 출처 |
|---|---|---|
| Operation | `compressor_v2().automatic_compression` (PR_L2, ratio 0.5) | SDK 호출, SDK metadata |
| 입력 | 공식 샘플 `graphmodule.pt`, 28,485,001 bytes, `[1,3,224,224]`, framework pytorch | 파일, 공식 예제 |
| SDK status | `completed` | SDK metadata |
| 모델 파일 크기 | 28.485 MB → 7.456 MB (−73.8 %) | **QA framework가 파일에서 직접 측정** |
| SDK 보고 size | 27.17 → 7.11 (단위 미명시, MB 추정) | **SDK/서버 보고값** (독립 검증 안 됨) |
| FLOPs | 1.957 G → 0.943 G (−51.8 %) | SDK/서버 보고값 |
| Parameters | 7,060,084 → 1,814,928 (−74.3 %) | SDK/서버 보고값 |
| 레이어별 pruning 정보 | 50 layers, 각 `channels`/`values` | SDK metadata (`compression_info.layers`) |
| `model_info.task / model / dataset` | **빈 문자열** | SDK metadata — 서버는 이 모델이 무슨 task인지 모른다 |
| 산출물 | `sdk_output.pt` SHA-256 `bcb3162d…9da5`; SDK가 로컬 torch 2.0.1로 `sdk_output.onnx` 추가 생성 | QA framework 계산 / SDK 로그 |
| Credit | estimated 25 (클라이언트 상수) / **actual 25** (잔액 500→475 관찰) | SDK 사용자·크레딧 조회 |
| accuracy / latency / memory | **측정되지 않음** (`null`) | — |
| Reproducibility | `NOT_VERIFIED` (runs = 1) | — |
| 기본 Quality Gate | **FAIL** (model_size만 PASS, 나머지 5개 NOT_APPLICABLE) | QA framework |
| 결함 분류 | `UNCLASSIFIED: required criteria could not be evaluated (missing data)` | QA framework |

핵심 관찰: **SDK가 "성공"이라고 말한 것은 "압축 작업이 끝났고 더 작은 파일이 나왔다"까지다.** 그 파일이 *쓸 수 있는* 모델인지(정확도), *더 빠른지*(지연), *덜 쓰는지*(메모리), *다시 만들면 같은지*(재현성)는 SDK 계약 밖이며 QA가 별도로 증명해야 한다.

---

## 2. 현재 구현의 검증 가능 범위 (READ-ONLY 분석 결과)

### A. 지금 이미 검증 가능한 항목 (실 데이터로 확인됨)

| 항목 | 근거 | 구현 위치 |
|---|---|---|
| 작업 호출 계약 (시그니처·인자·반환 타입) | 실행 직전 `inspect.signature` 재검증, 결과 `CompressorMetadata` 매핑 성공 | `netspresso_adapter.py`, `test_netspresso_adapter.py` (sdk_surface 대조) |
| SDK status → ExecutionStatus 매핑 | `completed`→COMPLETED, `error`→ERROR(+error_detail), `stopped`→BLOCKED, SystemExit→BLOCKED | `_map_compression_metadata`, 가짜 SDK 테스트 3종 |
| 모델 파일 크기 감소 | 파일 크기 직접 측정, `model_size` 기준 PASS | `validation/performance.py` |
| 산출물 존재·크기·SHA-256 기록·추적성 메타데이터 | `artifact_from_file`, 11개 메타데이터 키 | `validation/artifact.py` |
| 환경 fingerprint | python/platform/sdk_version/execution_mode/api_calls/operations | `Environment` |
| Credit 회계 (추정 vs 관찰 분리, before/after) | 장부 1건, 무결성 테스트·CI assert | `credit_ledger.py` |
| 안전 경계 | dry-run 기본, 미확인 real 거부, 키 비노출, 재시도 없음 | 22개 테스트 |

### B. 지금 검증할 수 없는 항목

| 항목 | 왜 불가능한가 |
|---|---|
| Accuracy | 평가 데이터셋도, task 정의도 없다. SDK metadata의 task/dataset이 비어 있어 서버도 모른다. 샘플 `graphmodule.pt`의 학습 데이터·클래스 매핑이 공개 문서에 없다 |
| Latency | 어떤 디바이스에서도 측정하지 않았다. 로컬 ORT 측정도 아직 수행하지 않았다 |
| Memory | 동일 |
| Artifact checksum 검증 | 비교할 **기대값**이 없다. 첫 산출물의 체크섬은 *기록*되었을 뿐 *검증*되지 않았다 |
| Reproducibility (Level 2+) | 동일 조건 2회차 실행이 없다 |
| 산출물 구조적 유효성 | `.pt`를 로드하거나 `.onnx`를 `onnx.checker`로 검사하지 않았다 (SDK가 ONNX export에 성공했다는 간접 증거만 있음) |
| FLOPs/params의 독립 검증 | 서버 보고값을 그대로 기록했다. 로컬 재계산으로 교차 검증하지 않았다 |

### C. 실 NetsPresso API 호출이 필요한 항목

- 2회차 동일 조건 압축 (재현성 Level 2–3)
- 다른 최적화 단계의 산출물 생성 (quantization, conversion)
- **타깃 디바이스**의 latency / memory (Profiler = 디바이스 팜) — 로컬 측정으로 대체 불가
- 다른 작업의 실제 Credit 차감 검증 (현재 compression 25만 확인)
- 서버 측 지원 조합 확정 (`available_options`는 compression 응답에 부수적으로 왔음)

### D. API 호출 없이 로컬에서 검증 가능한 항목 (0 Credit)

- 기존 산출물의 구조적 유효성: `torch.load(sdk_output.pt)`, `onnx.checker.check_model(sdk_output.onnx)`, 입력/출력 텐서 shape (3.11 venv에 torch/onnx/onnxruntime 존재)
- 원본 `graphmodule.pt` → ONNX 로컬 export (torch 2.0.1) → 원본/압축 **동일 조건 로컬 추론** (onnxruntime CPU)
  - 출력 일치도 proxy: top-1 agreement, cosine similarity, max abs diff (정확도 *대체 지표*, 정확도 자체는 아님)
  - 로컬 latency: warm-up + N회 측정 median/p95 (개발 머신 수치, 타깃 디바이스 아님)
  - 로컬 peak memory (CPU RSS 증가분)
- FLOPs/params 로컬 재계산 → SDK 보고값과 교차 검증
- 기존 체크섬 재계산 (Level 3의 *파일* 측 절반: 파일이 변하지 않았는지)
- SDK metadata.json 스키마 검증 (필수 키, enum 값, ratio 범위, 레이어 수 > 0)
- 설정 일관성: 요청한 ratio 0.5 == metadata.ratio, 요청 input_shapes == metadata.model_info.input_shapes
- Gate / 결함 분류 / 회귀 비교 로직 (이미 테스트됨)

### E. Mock으로만 검증 가능한 항목

- Model × Device × Runtime × Backend × Optimization 매트릭스 전수/pairwise/risk 동작
- 9종 결함 시나리오(정확도·지연 회귀, 체크섬 불일치, 비재현, 변환 실패, 호환성, UNCLASSIFIED, BLOCKED, 런타임 UNSUPPORTED)의 분류 경로
- baseline vs regressed 회귀 리포트 생성
- 리포트(JSON/HTML) 렌더링 전 경로

Mock 결과는 어떤 문서·리포트에서도 실 결과로 표현하지 않는다 (HTML 상단 고지, 설정 파일 주석, README §15).

### F. 현재 Quality Gate가 FAIL인 구조적 이유

1. **정책**: `quality_gate.yaml`의 6개 기준 중 5개가 `required: true`이고, 프레임워크는 "필수 기준을 평가할 데이터가 없으면 PASS를 선언하지 않는다"(`NOT_APPLICABLE` + required → FAIL). 이 정책은 의도된 것이며 올바르다.
2. **적용 범위 불일치**: 이 Gate는 *릴리스* 수준 질문(정확도·지연·메모리·재현성)을 하는데, 실행된 것은 *압축 단계* 하나다. 압축 작업은 설계상 정확도·지연·메모리를 산출하지 않는다. 즉 FAIL은 "품질이 나쁘다"가 아니라 "**릴리스를 판단할 증거가 아직 수집되지 않았다**"는 뜻이다. 결함 분류가 `UNCLASSIFIED (missing data)`인 이유도 같다.
3. **구조적 공백**: 프레임워크에 "파이프라인 단계별 Gate"라는 개념이 없다. Phase 5-B 스크립트는 임시로 `model_size + artifact`만 묻는 압축 단계 Gate를 코드 안에 만들어 PASS를 확인했지만, 이는 설정이 아니라 스크립트 하드코딩이다. → 수정안 §5.
4. **증거 수집 경로 부재**: 정확도·지연·메모리를 *채우는 코드*(로컬 평가기)가 없다. Result Model에는 자리가 있지만 값을 넣는 주체가 없다.
5. **기준 체크섬 저장소 부재**: 첫 산출물의 체크섬을 "다음 실행의 기대값"으로 승격하는 baseline 레지스트리가 없다. 그래서 두 번째 실행을 해도 비교 대상을 수동으로 찾아야 한다.

### G. 이미 구현되어 재개발이 불필요한 기능

Result Model(모든 필드와 JSON 라운드트립), BaseAdapter 계약, MockAdapter(결정적 시나리오), NetsPressoAdapter(정책·dry-run·compression 실 경로·redaction), 매트릭스/pairwise(커버리지 실측)/risk, 검증기 5종(accuracy·latency·memory·model_size·artifact·reproducibility; NOT_APPLICABLE 의미론 포함), Quality Gate(기준별 결과·사유·WARN), 결함 분류(증거 기반, UNCLASSIFIED), 회귀 비교(threshold 기반, NEW/MISSING/NOT_COMPARABLE), JSON/HTML 리포터, Credit Ledger(simulated/estimated/actual, budget_check, 무결성 테스트), 실행 스크립트 3종, CI(0 Credit), 테스트 111개.

따라서 Phase 5-D 이후의 작업은 **새 검증기 발명이 아니라 "값을 채우는 평가기"와 "단계별 Gate 프로파일", "baseline 레지스트리"를 추가하는 것**이다.

---

## 3. Quality Gap 표

범례 — 값의 출처: **M** = QA framework가 직접 측정, **S** = SDK/서버 보고, **C** = QA framework 계산, **–** = 미측정, **R** = 향후 실측 필요

| Quality Area | 현재 상태 | 필요한 Evidence | 검증 방법 | API Credit 필요 | 우선순위 |
|---|---|---|---|---|---|
| Execution Status | `completed` (S) → COMPLETED (C). 1회 관찰 | status 분기 전부(`error`/`stopped`) 실 관찰, error_detail 스키마 | 실 오류 유도는 Credit 낭비 → 가짜 SDK 테스트로 분기 검증(완료), 실 관찰은 자연 발생 시 기록 | 아니오 (분기 테스트) | P0 완료 / P2 관찰 |
| Model Size | 28.485→7.456 MB (M), −73.8 % PASS | 단계 Gate에서 "증가 금지" 기준 적용, 포맷별(pt/onnx) 크기 | 파일 크기 측정 (완료) + onnx 크기 추가 | 아니오 | P0 완료 |
| FLOPs | 1.957G→0.943G (S). **독립 검증 없음** | 로컬 재계산값과 SDK 값의 오차 | 3.11 venv에서 FLOPs 카운터(torch fx / onnx 기반)로 재계산 → 허용 오차 내 일치 | 아니오 | P1 |
| Parameter Count | 7.06M→1.81M (S). 독립 검증 없음 | `torch.load` 후 파라미터 합산값 | 로컬 재계산 → SDK 값과 정확히 일치해야 함 | 아니오 | P1 |
| Accuracy | – (SDK metadata task/dataset 비어 있음) | (a) task 정의 + 라벨 데이터셋 + baseline 정확도, (b) 당장은 원본↔압축 **출력 일치도 proxy** | (a) 공개 라벨셋 보유 모델로 교체 실험, (b) 로컬 ORT 추론 비교 (top-1 agreement, cosine) — 정확도로 표기하지 않음 | (a) 신규 압축 시 25 / (b) 아니오 | **P0** |
| Latency | – | 동일 조건 baseline/optimized 측정 (warm-up, N회, median/p95), 디바이스·런타임·스레드 고정 | 로컬 ORT CPU (개발 머신 수치) + 타깃 디바이스는 Profiler | 로컬 아니오 / 타깃 25 per profile (+convert 50) | **P0** |
| Memory | – | peak CPU 메모리 baseline/optimized; GPU는 N/A(로컬) | 로컬 RSS 증가분 측정; 타깃은 Profiler `memory_footprint*` (단위 미검증) | 로컬 아니오 / 타깃 25 | P1 |
| Artifact Integrity | 존재·크기·SHA-256 기록 (M). 기대값 없음 → NOT_APPLICABLE. 구조 검증 없음 | 기대 체크섬(baseline 레지스트리), `torch.load`/`onnx.checker` 통과, I/O shape | 레지스트리 승격 + 로컬 구조 검사 | 아니오 | **P0** |
| Reproducibility | NOT_VERIFIED (runs=1) | 동일 조건 2회차 산출물 체크섬·메타데이터·출력 비교 | Level 2–3은 실 재실행, Level 4는 로컬 추론 비교 | **예, 25** (2회차 압축) | P1 |
| Configuration | 요청 ratio/shape가 metadata에 그대로 반영됨 (S, 육안 확인) | 자동 일관성 검사: 요청값 == metadata 값, `model_info.framework`, options 스키마 | metadata.json 스키마·일관성 테스트 (0 Credit) | 아니오 | P1 |
| Environment | python/platform/sdk 1.17.0/실행 모드 기록 (M) | torch/onnxruntime 버전, CPU 모델, 스레드 수(로컬 측정 재현성용), 서버 host | Environment.extra 확장 | 아니오 | P2 |
| Credit Accounting | actual 25 = 추정 25 (관찰). 다른 작업 미검증 | 작업별 실 차감 관찰, 실패 시 환불 여부 | 각 실 작업 전후 잔액 기록(구현됨), 실패 케이스는 자연 발생 시 기록 | 작업마다 해당 비용 | P1 |

---

## 4. Quality Gap TOP 10 (영향도 × 해소 비용 기준)

| # | Gap | 왜 중요한가 | 해소 비용 |
|---|---|---|---|
| 1 | **Accuracy 증거 전무** — 샘플 모델의 task/라벨 불명 | 압축 모델이 쓸 수 있는지 판단 불가. 릴리스 판단의 1순위 질문 | 0 Credit proxy(출력 일치도) 즉시 가능; 진짜 정확도는 라벨셋 보유 모델로 25 Credit 재실험 |
| 2 | **Latency 증거 전무** | "최적화"의 두 번째 약속. 압축률이 커도 속도가 빨라진다는 보장은 없음 | 로컬 ORT 0 Credit; 타깃 디바이스 75 Credit(convert+profile) |
| 3 | **Gate에 단계 개념 없음** → 압축만 한 결과에 릴리스 Gate를 물어 FAIL | FAIL이 "나쁨"인지 "미측정"인지 리포트만 보고 구분 어려움 | 설정 변경(프로파일) + 소규모 코드 |
| 4 | **기대 체크섬 레지스트리 없음** | artifact 기준이 영원히 NOT_APPLICABLE | 소규모 코드(baseline 승격 규칙) |
| 5 | **Reproducibility runs=1** | 서버 비결정성 여부를 모름 → 회귀 판단 자체가 불안정 | 25 Credit (2회차) |
| 6 | **산출물 구조 검증 없음** | 7 MB 파일이 로드 가능한 모델인지 확인 안 함 | 0 Credit |
| 7 | **SDK 보고 FLOPs/params 교차 검증 없음** | 서버값을 그대로 믿음 → 서버 버그 탐지 불가 | 0 Credit |
| 8 | **Memory 증거 전무** | 엣지 타깃에서 결정적 제약 | 로컬 0 Credit / 타깃 25 |
| 9 | **Configuration 일관성 자동 검사 없음** | 요청 파라미터가 서버에서 바뀌어도 모름 | 0 Credit |
| 10 | **SDK enum–서버 드리프트**(`MIX`, Jetpack 6.2.1, `dlc` 탈락) 미대응 | 매트릭스 지원 지식이 서버 실제와 어긋남 | 0 Credit(서버 옵션 스냅샷을 지원 지식 소스로 승격) |

---

## 5. 구조적 문제에 대한 수정안 (5-C 제안 → **5-D에서 구현 완료**)

| 변경 | 이유 | 변경 파일 | 기존 동작 영향 | 테스트 영향 |
|---|---|---|---|---|
| **Gate 프로파일** `quality_gate.yaml`에 `profiles: {release: …, compression_stage: …, local_eval: …}` 추가, 기존 최상위 키는 `release`의 별칭으로 유지 | §2-F-3. 단계별로 물을 질문이 다름. 스크립트 하드코딩 제거 | `config.py`(loader), `quality_gate.py`(profile 선택), `run_real_netspresso.py` | 없음 — 프로파일 미지정 시 현재 동작 | 프로파일 로딩 테스트 추가, 기존 테스트 무변경 |
| **Baseline 레지스트리** `reports/baselines/<operation>/<config_key>.json`에 첫 산출물의 체크섬·메타데이터·메트릭을 승격; 이후 실행은 자동으로 기대값 조회 | §2-F-5. artifact 기준이 영원히 NOT_APPLICABLE | 신규 `framework/baselines.py`, `run_real_netspresso.py` | 없음 (선택적) | 신규 테스트 |
| **로컬 평가기** `framework/evaluation/local_onnxruntime.py`: 두 ONNX의 출력 일치도·latency(median/p95)·peak RSS를 측정해 `Metrics.extra`/`metrics`에 기록. 3.11 venv에서만 실행, 3.14 테스트는 가짜 세션으로 | Gap #1·#2·#8의 0 Credit 해소 | 신규 모듈 + 스크립트 | 없음 | 신규 테스트 |
| **Environment.extra 확장** (torch/ort 버전, cpu, threads) | 로컬 측정 재현성 | `netspresso_adapter.py`, 평가기 | 없음 | 소폭 |

이 네 가지는 Phase 5-D(0 Credit)에서 구현되었다(`framework/evaluation/`, `framework/baselines.py`, `configs/quality_gate.yaml` profiles, `scripts/run_local_eval.py`). Result Model에는 `QualityGateResult.profile` 필드 하나(기본값 `release`)만 추가되었고 BaseAdapter 계약·상태 모델은 변경되지 않았다. 5-D 결과로 Gap #3·#4·#6·#7·#8·#9가 닫혔고, #1(정확도)은 proxy 증거(부정적)로 구체화되었으며, #2는 로컬 수치만 확보(타깃 미측정), #5·#10은 미해소다.

---

## 6. Credit 관점 분류와 다음 실험 효율성

잔여 475, 예비 100 → 사용 가능 375. 모든 추정은 **클라이언트 측 상수**이며 compression 25만 실측 일치가 확인되었다.

**A. 0 Credit** — 스키마/설정/산출물 구조/체크섬/FLOPs·params 재계산/로컬 ORT 평가(일치도·latency·memory)/Gate·회귀·결함 로직/Mock E2E/리포트. → Phase 5-D 전부.

**B. Credit 가능성이 있으나 최소화 가능** — 서버 지원 조합 확인(이미 받은 `available_options` 스냅샷 재사용으로 0), 다른 모델의 compression(모델당 25; 라벨셋이 있는 모델 1개만 선택), Credit 환불 정책(실패를 유도하지 않고 자연 발생 시 기록).

**C. 실 API가 반드시 필요** — 재현성 2회차(25), 타깃 디바이스 latency/memory(convert 50 + profile 25, baseline/optimized 각각이면 ×2), quantization 단계 증거(50).

**권고 실험 수: 4회, 합계 추정 225 Credit (잔여 250 ≥ 예비 100).**

| 순서 | 실험 | 추정 | 얻는 증거 | 해소되는 Gap |
|---|---|---|---|---|
| E1 | 동일 조건 `automatic_compression` 2회차 | 25 | Level 2–3 재현성, 실 데이터 회귀 비교, 체크섬 레지스트리 검증 | #4, #5 |
| E2 | 압축 ONNX → `convert_model`(OpenVINO, Intel-Xeon, FP16) + `profile_model` | 50 + 25 | 최적화 모델의 타깃 latency/memory, 변환 단계 계약, convert/profile 실 차감 | #2, #8, Conversion 단계 |
| E3 | 원본 ONNX → 동일 convert + profile | 50 + 25 | baseline latency/memory → **실 latency/memory Gate 평가 가능** | #2, #8 |
| E4 | `automatic_quantization`(INT8, 공식 224 캘리브레이션 npy) | 50 | Quantization 단계 계약·산출물, quantize 실 차감 | Quantization 단계 |

E1만으로도 "재현성 + 회귀"가 실 데이터로 닫히고, E2+E3로 latency/memory Gate가 실측으로 닫힌다. Accuracy는 라벨셋이 있는 모델이 아니면 어떤 Credit을 써도 닫히지 않으므로, E1~E4 중 모델 교체 여부를 먼저 결정해야 한다(§4 #1). 475를 모두 쓰는 것은 추가 증거를 거의 만들지 않는다 — 같은 작업의 반복은 재현성 2회차 이후 한계 효용이 급감한다.

---

## 7. 이번 Phase의 결론

- SDK `completed` ≠ Release PASS라는 원칙은 실 데이터로 입증되었다. 압축은 성공했지만 릴리스 판단에 필요한 6개 질문 중 1개(크기)만 답할 수 있다.
- 부족한 것은 검증기나 Gate가 아니라 **증거를 채우는 평가기, 단계별 Gate 프로파일, 기준값 레지스트리**다.
- 다음 단계의 대부분은 0 Credit으로 가능하며, 실 Credit은 4회·약 225로 충분하다.

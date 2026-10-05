# Release Quality Gate — 정의와 PASS 조건

> 질문: "**어떤 조건이면** 최적화된 모델을 Release PASS로 판단하는가?"
> 부족한 증거의 목록은 [`quality_gap_analysis.md`](quality_gap_analysis.md), 증거를 만드는 테스트는 [`test_strategy.md`](test_strategy.md).
>
> 이 문서의 모든 임계값은 **project-defined example policy**이며 Nota 공식 기준이 아니다.

---

## 1. 원칙

1. **SDK status `completed` ≠ Release PASS.** 서비스 완료는 Layer 1(계약) 사실이고, 릴리스는 Layer 2(품질) 판정이다.
2. **결측은 PASS가 아니다.** 필수 기준을 평가할 데이터가 없으면(`NOT_APPLICABLE`) Gate는 FAIL이며, 사유에 "not available"이 남는다. 다만 이는 **회귀가 아니라 증거 부족**이므로 결함 분류는 `UNCLASSIFIED (missing data)`다.
3. **Mock 수치는 Gate 증거가 아니다.** Mock 리포트는 Gate *로직*을 보여줄 뿐이며, Release 판단에는 `adapter == netspresso`인 실 결과(그리고 로컬 측정은 출처 라벨 포함)만 쓴다.
4. **단계마다 물을 질문이 다르다.** 압축 단계에 릴리스 질문을 하면 항상 FAIL이 나오고, 그 FAIL은 정보가 없다. 그래서 Gate는 **프로파일**로 나눈다(§3).
5. **개선은 항상 PASS, 악화만 임계값으로 통제한다.** 최적화의 목적이 개선이지만 Gate는 "악화 금지"만 강제한다. 기대 개선량은 Informational로 리포트한다.
6. **임계값은 설정이지 코드가 아니다.** `configs/quality_gate.yaml`에서만 바꾼다.

---

## 2. 기준(criterion) 정의

| 기준 | 비교 | 의미론 | 현재 example 임계값 | 필수 여부(release) |
|---|---|---|---|---|
| accuracy | baseline vs optimized, 같은 metric 이름 | `drop_pp = (base − cur) × 100` ≤ max_drop | 1.0 pp | 필수 |
| latency | baseline vs optimized, 동일 조건·동일 출처 | `(cur − base) / base × 100` ≤ max_increase | 10 % | 필수 |
| memory | peak, 동일 조건 | 동일 | 15 % | 필수 |
| model_size | 파일 크기 | 동일 | 10 % | 선택(WARN) |
| artifact | 산출물 | 존재 ∧ 비어 있지 않음 ∧ (기대 체크섬 있으면 일치). 기대값 없으면 NOT_APPLICABLE | — | 필수 |
| reproducibility | 반복 실행 | `BITWISE` 또는 `FUNCTIONALLY_REPRODUCIBLE`. 2회차 없으면 NOT_APPLICABLE | — | 필수 |

기준별 결과 상태: `PASS` / `FAIL` / `WARN`(선택 기준 실패) / `NOT_APPLICABLE`(데이터 없음). Overall은 `PASS` 또는 `FAIL`만 존재하며, **필수 기준의 FAIL 또는 NOT_APPLICABLE이 하나라도 있으면 FAIL**이다.

---

## 3. Gate 프로파일 (Phase 5-D에서 구현 — `configs/quality_gate.yaml` `profiles:`, `QualityGate(config, profile=...)`)

| 프로파일 | 적용 시점 | 묻는 기준 | PASS가 뜻하는 것 | PASS가 뜻하지 **않는** 것 |
|---|---|---|---|---|
| `compression_stage` | 압축/양자화/변환 **직후** | model_size(증가 금지, 필수), artifact(존재·구조, 필수; 체크섬은 레지스트리 있을 때만), configuration consistency(필수) | 그 단계가 계약대로 끝났고 산출물이 쓸 수 있는 형태다 | 품질이 유지되었다 |
| `local_eval` | 로컬 ORT 평가 후 | 위 + output_agreement(proxy, 필수), latency·memory(`local_onnxruntime_cpu` 출처, 필수), FLOPs/params 교차 검증(필수) | 개발 머신에서 원본과 기능적으로 가깝고 더 빠르거나 같다 | 타깃 디바이스에서 그렇다; 정확도가 유지되었다 |
| `release` | 실 baseline·optimized 증거가 모두 모인 뒤 | §2의 6개 기준 전부 + reproducibility ≥ Level 3/4 | **릴리스 가능** | — |

리포트는 항상 "어느 프로파일로 평가했는지"를 표기하고(`QualityGateResult.profile`), 하위 프로파일의 PASS는 상위 프로파일로 승격되지 않는다. 구현된 프로파일 이름은 `compression` / `local_eval` / `release`이며, `compression`에서는 `artifact.checksum_required: false`로 첫 산출물의 체크섬 부재를 허용하고, `structural_validity`는 모든 프로파일에서 필수다.

**Phase 5-D 로컬 평가(0 Credit) 적용 결과** (`reports/local_eval/<id>/quality_gate.json`):

| 프로파일 | 결과 | 사유 |
|---|---|---|
| `compression` | **PASS** | 크기 −73.8 %, 산출물 존재·구조 유효(PT GraphModule 전체 로드 PASS, ONNX checker PASS), 로컬 latency/memory 악화 없음 |
| `local_eval` | **FAIL** | **output equivalence proxy** min cosine **0.825** < 0.99 — 압축 모델의 출력이 원본과 기능적으로 동등하지 않음(재학습 없는 pruning의 예상 결과이며 *정확도 측정은 아님*). latency/memory/size/artifact/structural은 PASS |
| `release` | **FAIL** | accuracy NOT_APPLICABLE(라벨셋 없음), artifact 기대 체크섬 없음(candidate 단계), reproducibility NOT_VERIFIED(runs=1) |

즉 Phase 5-D는 `release` FAIL의 성격을 "증거 부족"에서 한 단계 구체화했다: 크기·구조·파라미터·FLOPs·로컬 성능은 증거가 생겼고, **기능적 동등성은 증거가 생겼으나 부정적**이며, 정확도·재현성은 여전히 증거가 없다.

---

## 4. Release PASS 조건 (전부 충족)

| # | 조건 | 증거 파일 | 현재(Phase 5-B) |
|---|---|---|---|
| R1 | 입력 모델이 로드 가능하고 선언 shape와 일치, 체크섬이 레지스트리에 있음 (TC-M01) | baseline registry | ✓(5-D: 원본 GraphModule 로드 PASS, 7,060,084 params, ONNX export PASS, source sha 기록) |
| R2 | 최적화 작업이 계약대로 `completed`, metadata가 요청과 일치 (TC-O01, Level 1) | execution_result, sdk metadata | ✓ |
| R3 | 산출물 존재·비어 있지 않음·구조 유효·**기대 체크섬 일치** (TC-A01) | artifact + registry | 존재·기록·**구조 ✓(5-D)**, 일치 ✗(registry `candidate`) |
| R4 | 재현성 Level ≥ 3 또는 Level 4 (TC-O03/A02) | 2회차 실행 | ✗ (runs = 1) |
| R5 | accuracy drop ≤ 임계값, **같은 metric·같은 평가셋** (TC-E01) | 로컬 평가 결과(`accuracy_source` 기록) | ✗ (task 불명) |
| R6 | latency 증가 ≤ 임계값, 동일 조건·동일 출처 (TC-E02) | 로컬 또는 타깃 측정 | 로컬 ✓(5-D: median 3.49→3.57 ms, +2.5 %, 출처 `local_onnxruntime_cpu`); 타깃 ✗ |
| R7 | memory 증가 ≤ 임계값 (TC-E03) | 동일 | 로컬 ✓(5-D: 76.0→58.1 MB, 프로세스 격리 측정); 타깃 ✗ |
| R8 | 회귀 비교에서 baseline 대비 REGRESSION 0건 (TC-R01) | regression_report | ✗ (baseline 실행 없음) |
| R9 | 결함 분류에 `UNCLASSIFIED (missing data)`가 없음 — 즉 모든 필수 기준이 실제로 평가됨 | case_result | ✗ |
| R10 | Credit 장부 무결성(used = Σactual) — 예산 통제, Informational | credit_usage.json | ✓ |

R1–R9 중 하나라도 ✗이면 Release **FAIL**. 현재 상태는 FAIL이며, 그 FAIL의 성격은 "품질 회귀"가 아닌 "**증거 미수집**"이다. 이 구분이 리포트에 그대로 드러나는 것이 이 프레임워크의 설계 목표다.

---

## 5. 상태 의미론 (리포트를 읽는 법)

| 표기 | 뜻 | Release 판단에서 |
|---|---|---|
| criterion `PASS` | 측정값이 임계값 안 | 긍정 증거 |
| criterion `FAIL` | 측정값이 임계값 밖 | **회귀** → 결함 분류(ACCURACY_/PERFORMANCE_/MEMORY_REGRESSION, ARTIFACT_/REPRODUCIBILITY_ERROR) |
| criterion `WARN` | 선택 기준 위반 | 리포트만, 판정 불변 |
| criterion `NOT_APPLICABLE` | 데이터 없음 | 필수면 FAIL, 결함은 `UNCLASSIFIED (missing data)` — **회귀로 읽지 말 것** |
| case `PASS / FAIL / BLOCKED` | 실행됨 | FAIL은 위 둘 중 어느 쪽인지 사유로 구분 |
| case `NOT_TESTED` | 지원 가능성은 있으나 미실행 | 증거 없음(부정 증거도 아님) |
| case `UNSUPPORTED` | 명시적 미지원 | 릴리스 범위에서 제외, 실패 아님 |
| `latency_source: local_onnxruntime_cpu` 등 출처 라벨 | 측정 환경 | 다른 출처끼리는 비교 금지 |

---

## 6. 임계값 거버넌스

- 현재 값(accuracy 1.0 pp, latency +10 %, memory +15 %, model_size +10 %)은 **예시**다. 출처: 이 프로젝트의 설계 선택. Nota 공식 기준·제품 요구사항·고객 SLA가 확인되면 그 값으로 교체하고, 변경 이력은 git에 남긴다.
- task별(classification/detection/segmentation) override, 디바이스 등급별(엣지 MCU vs 서버 GPU) override를 설정 스키마에 둘 수 있게 설계한다(Phase 5-D).
- 임계값을 완화해서 FAIL을 PASS로 만드는 변경은 커밋 메시지에 사유를 적고, 회귀 리포트가 그 변경 전후를 구분할 수 있어야 한다(`thresholds` 블록이 리포트에 이미 기록됨).

---

## 7. 현재 실측으로 확정된 사실 (2026-10-05 기준)

실 실행 1회 (Phase 5-B, 25 Credit):
- `automatic_compression`(graphmodule.pt, ratio 0.5): status completed, 파일 28.485 → 7.456 MB(−73.8 %), 산출물 SHA-256 기록, 실 Credit 25(= 클라이언트 상수).

로컬 평가 (Phase 5-D, 0 Credit, 개발 머신, Nota 서버 측정 아님):
- 파라미터: SDK 보고 7,060,084 → 1,814,928가 torch·ONNX 재계산과 **정확히 일치**(PASS).
- FLOPs: SDK 보고 1.957 G / 0.943 G는 ONNX MAC 추정의 **2×MACs**(1.945 G / 0.934 G)와 0.6–1.0 % 이내로 일치(PASS) → SDK의 "flops"는 2×MACs 관례로 **관찰**됨(문서화된 정의는 아님).
- 로컬 latency(median, 100회, ORT CPU): 원본 3.49 ms → 압축 3.57 ms(+2.5 %). **FLOPs −52 %가 CPU latency 개선으로 이어지지 않았다**(채널 pruning의 메모리 대역폭·커널 효율 특성; 원인 확정은 아님). p95는 9.5 → 14.1 ms로 악화.
- 로컬 peak RSS 증가분(프로세스 격리): 76.0 → 58.1 MB(−23.5 %). 단일 프로세스에서 순서대로 측정했을 때는 +86 %로 나왔던 값이며, 측정 순서 효과가 결론을 뒤집을 수 있음을 확인했다.
- 출력 동등성 proxy(동일 결정적 입력, 3개 출력): min cosine **0.825**, max |Δ| 11.26 → **FAIL**. 압축 모델은 원본과 기능적으로 동등하지 않다. 이는 정확도 측정이 아니다.
- 정확도·재현성(Level 2+)·타깃 디바이스 latency/memory는 **여전히 측정되지 않았다**.

# Phase 5-E Strategy — YOLOv8n + COCO E2E QA (Preparation, 0 Credit)

> 결정: **GO FOR E5-1** (준비 단계 판정은 CONDITIONAL GO였고, 해소 작업 + R1으로 전환됨 — 근거 §12·§13; 상세는 [`phase5e_conditional_go_resolution.md`](phase5e_conditional_go_resolution.md) §11). 이 문서는 NetsPresso API 호출 없이 작성·검증되었다(Credit 25 used / 475 remaining / 1 operation, 변동 없음).
> 관련 문서: [`historical_yolov8_regression.md`](historical_yolov8_regression.md)(2025 사례), [`yolov8_baseline_strategy.md`](yolov8_baseline_strategy.md)(측정값), [`phase5e_release_test_plan.md`](phase5e_release_test_plan.md)(TC 목록).

---

## 1. 목적

Phase 5-B/5-D의 결론은 "SDK `completed` ≠ Release PASS이며, 현재 가장 큰 공백은 **진짜 accuracy**"였다. 공식 샘플 `graphmodule.pt`는 task·라벨이 없어 accuracy를 영원히 N/A로 남긴다. Phase 5-E는 **라벨 데이터셋과 공식 Nota 경로가 모두 존재하는 모델**로 바꿔, 파이프라인 전체(계약 → 최적화 → 산출물 → 구조 → params/FLOPs → accuracy → latency → memory → 재현성 → 회귀 → Gate → Release)를 실측으로 닫는 것이 목적이다. 동시에 2025년 포트폴리오에서 실패했던 YOLOv8 계열을 공식 경로로 재검증한다.

## 2. YOLOv8n 선정 이유 (검증 결과)

| 기준 | 평가 | 근거 |
|---|---|---|
| 10.1 Historical continuity | ✓ | 2025 `netspresso_test`는 YOLOv8-l ONNX(167 MB)를 compressor에 직접 넣어 `NotValidFrameworkException`으로 실패했다(저장소 기록). 2026은 같은 계열의 YOLOv8-n을 **공식 fx 경로**로 다시 간다 — 반복이 아니라 "실패 유형을 업로드 전에 제거하는 precheck + 전 단계 자동 검증"으로 역량이 확장됐음을 보일 수 있다 |
| 10.2 Nota workflow alignment | ✓ (archived) | Nota 공식 `ModelZoo-YOLOv8` + `ultralytics_nota` 포크가 `export_netspresso()` → fx body + head meta → 압축 → `YOLO_netspresso` 재학습 → val 경로를 문서화. 두 저장소 모두 **2024-02 이후 archived** — 현 SDK(compressor_v2/api_key)와의 결합은 5-E 실 실행에서 확인해야 함 |
| 10.3 Accuracy evaluation | ✓ | COCO 라벨 + ultralytics `val`로 mAP50-95/mAP50/P/R 측정 가능(coco128로 배관 검증 완료, val2017은 승인 후) |
| 10.4 Performance evaluation | ✓ | 5-D 로컬 ORT 평가기(격리 프로세스, median/p95, peak RSS)를 그대로 사용해 baseline 측정 완료 |
| 10.5 Portfolio value | ✓ | 공고 요구사항 14항목 중 12항목을 실측 경로로 시연 가능(§5). 가장 작은 변형이라 업로드·평가 비용이 최소이고 Credit은 압축 1회 25 |

대안을 고려하지 않은 것이 아니다: 공식 샘플 `graphmodule.pt`(task 불명 → accuracy 불가), 분류 모델(공식 Nota 경로 없음, 2025 연결 없음)보다 YOLOv8n이 세 축(연속성·공식 경로·라벨셋)을 동시에 만족하는 유일한 후보였다.

## 3. 과거 포트폴리오와의 연결 (2025 → 2026)

```
2025  netspresso_test        사용자 관점 QA: 문서대로 따라가며 YOLOv8-l ONNX 압축 실패 발견,
                             문서/SDK 불일치 분석, 개선 제안(사전 호환성 검증 API 등)
  ↓   무엇이 바뀌었나         결과 미보존·키 하드코딩·CI 실 호출 → 어댑터 경계·dry-run·장부·JSON 증거·
                             키 비노출·CI 0 Credit
2026  netspresso-qa-automation  계약 → 매트릭스 → 최적화 → 산출물 → 구조 → params/FLOPs → accuracy →
                             latency → memory → 재현성 → 회귀 → Gate → Release
  ↓
5-E   YOLOv8n + COCO         2025년 제안(사전 호환성 검증)을 QA가 직접 구현(TC-Y8-030, 0 Credit, READY),
                             공식 경로로 1회 압축 후 전 단계 실측
```

이 연결이 실제 저장소로 뒷받침되는지는 `historical_yolov8_regression.md` §4·§7에서 항목별로 검증했다. 뒷받침되지 않는 부분(2025 정량 결과·원인 확정·자동화 수준)은 그대로 표시했다.

## 4. 현재 프레임워크와의 연결

| 5-E 단계 | 재사용하는 구현 | 추가 필요(0 Credit) |
|---|---|---|
| 모델 획득·fingerprint | `compute_sha256`, 신뢰 목록(`configs/local_eval.yaml`) | 완료(`scripts/yolov8_baseline_prep.py`) |
| 구조 검증 | `evaluation/artifact_structure.py` | checkpoint dict 인식 추가(완료) |
| fx export precheck | `validate_pt` + 포크 | 완료(`scripts/yolov8_fx_export_precheck.py`) |
| params/FLOPs | `evaluation/model_stats.py` | 완료(ultralytics 보고값 vs 독립 추정) |
| latency/memory | `evaluation/local_ort.py`(격리) | 완료 |
| accuracy | `validation/accuracy.py`(drop_pp 비교) | **mAP 측정 helper** — baseline은 완료, 압축 body는 head 재부착 helper 필요 |
| 압축 실행 | `NetsPressoAdapter`(5-B 실 경로) + `run_real_netspresso.py` | 인자화: `--model-path model_fx.pt --input-shape 1,3,640,640 --model-name yolov8n` (코드 변경 없음) |
| Gate | 프로파일 `compression`/`local_eval`/`release` | accuracy 기준은 `max_drop_percent`(pp) 그대로; mAP50-95를 `accuracy`에 매핑 |
| 재현성·레지스트리 | `baselines.py` | 정규화 체크섬 제안(ONNX metadata `date`) |
| 회귀 | `regression.py` | baseline run JSON = 5-E baseline, current = 압축 후 |

## 5. Nota [Technical Operations] QA Engineer 공고 매핑

공고 원문은 이 저장소에 없으므로 **일반적으로 요구되는 역량 명칭** 기준으로 매핑한다. 실제 공고 문구와 다르면 공고가 우선이다.

| Requirement | Current Framework (c8a6b04) | YOLOv8 Extension (5-E) | 차이/한계 |
|---|---|---|---|
| AI model optimization QA | compression 1회 실측 + Gate/결함 분류 | 공식 경로의 YOLOv8n 압축 + 재학습 전후 | 양자화·변환은 미실행 |
| E2E verification | 계약→결과 모델→Gate→리포트 자동 | 모델 획득→precheck→압축→평가→Release | 타깃 디바이스 단계 없음(Profiler 미사용) |
| Model × Device × Runtime × Backend | 324 조합, pairwise 100 %(Mock) | yolov8n 행은 이미 매트릭스에 존재 | 실 데이터는 단일 조합 |
| CLI/API | 어댑터 + 스크립트 4종 | `run_real_netspresso.py` 인자화 | — |
| Accuracy | 검증기만(데이터 N/A) | **mAP50-95 실측 가능** | val2017 미다운로드 |
| Performance | 로컬 ORT median/p95/RSS | baseline 15.5 ms/107 MB 측정 | 로컬 ≠ 타깃 |
| Artifact validation | 존재·SHA·구조·추적성 | fx/ONNX/압축본 3종 | 정규화 체크섬 제안 |
| Reproducibility | Level 0–1, 레지스트리 candidate | export 비트 비재현 원인 규명(date) | Level 2+ 실 재실행 필요 |
| Quality Gate | 프로파일 3종, 결측≠PASS | accuracy 기준 활성화 | 임계값은 example |
| Regression | threshold 기반 비교 | baseline vs compressed JSON | — |
| Python / pytest | 134 tests, 93 % coverage | +6 tests(5-E prep) | — |
| CI/CD | GitHub Actions 0 Credit | 변경 없음(YOLO 도구는 CI 밖) | YOLO venv는 로컬 전용 |
| Defect classification | 13 카테고리 + UNCLASSIFIED, 증거 기반 | HR-1/HR-2 매핑 | — |

## 6. 공식 Nota workflow evidence (확인된 것만)

| 단계 | 확인 | 출처 |
|---|---|---|
| YOLOv8 pretrained | ✓ | ultralytics/assets release (ModelZoo README가 링크) |
| NetsPresso-compatible export (`export_netspresso` → `model_fx.pt` + `netspresso_head_meta.json`) | ✓ (코드 확인·로컬 실행 성공) | `ultralytics_nota/ultralytics/yolo/engine/model.py` |
| NetsPresso optimization (upload + compress) | ✓ 문서, 단 **구 API**(`ModelCompressor`, email/password) | ModelZoo-YOLOv8 README §3 — 현 SDK는 `compressor_v2` + `api_key` (Phase 1) |
| compressed model → fine-tuning (`YOLO_netspresso(... 'detect_retraining')`, coco128) | ✓ 문서·코드 | README §4, `tasks.py DetectionModel_netspresso` |
| validation (`model.val`) | ✓ 문서 | README §4 |
| deployment artifact (convert/profile) | **NOT VERIFIED in the YOLOv8 workflow** — ModelZoo README에 없음(일반 PyNetsPresso docs 링크만) | — |
| 유지 상태 | 두 저장소 모두 **archived (2024-02)** | GitHub 메타데이터 |

## 7. Accuracy 전략

- metric: **Primary mAP50-95**, Secondary mAP50 / mAP75 / precision / recall — 모두 ultralytics `val`의 `results.box`에서 실제로 확인됨.
- 동일 조건: `yolov8_baseline_strategy.md` §6.
- Gate: `accuracy.max_drop_percent`(절대 pp) — **PROJECT-DEFINED EXAMPLE THRESHOLD**. 현재 설정값 1.0 pp는 분류용 예시였다. 검출의 mAP50-95는 데이터셋 크기에 따른 분산이 크므로, 임계값 확정 전에 **같은 baseline을 2회 이상 측정한 변동폭**(coco128에서는 결정적이지만 val2017 subset이면 추출 시드별 변동)과 **압축비 0.5의 특성**(재학습 전 mAP 급락 예상)을 먼저 기록한다. 제안 초안: release 1.0 pp(재학습 후), 압축 직후는 Gate 평가하되 FAIL이 예상된 결과임을 리포트에 명시.
- coco128 결과로 Release accuracy를 판정하지 않는다.
- proxy(cosine)는 보조 신호로 유지, accuracy와 혼용 금지.

## 8. Latency 전략

- 측정: 5-D 설계 유지(격리 프로세스, warm-up 10, 100회, median 기준·p95 기록, CPUExecutionProvider).
- 범위: Release Gate는 **inference-only latency**(ORT `run`)를 사용한다. E2E(letterbox 전처리 + 추론 + NMS 후처리)는 ultralytics `val`의 speed로 참고 기록만 한다(0.40 / 17.88 / 0.85 ms). 두 수치를 섞어 비교하지 않는다.
- Official vs local: ultralytics 공개 벤치마크(A100/CPU ONNX 수치)는 **reference evidence**, 로컬 측정은 **release comparison evidence**. 서로 비교하지 않는다.
- 임계값 `+10 %`(example). 로컬 p95 분산이 큼(29.6 ms vs median 15.5) → 머신 부하 통제 필요(측정 중 다른 작업 금지; 첫 측정은 설치 중 실행되어 19.7 ms였고 재측정에서 15.5 ms).

## 9. Memory 전략

격리 프로세스 peak RSS 증가분(baseline **107.13 MB**). 압축본도 같은 방식(ONNX, 같은 imgsz)으로 측정 후 `memory_delta_percent` 산출. GPU memory **N/A**(GPU 없음, 추정하지 않음). 임계값 `+15 %`(example).

## 10. Artifact / Reproducibility 전략

- fx 산출물: 존재·크기·SHA·신뢰 로드·구조(GraphModule)·forward shape — precheck에서 모두 기록(READY).
- ONNX: load·checker·graph·I/O·nodes·initializers·opset·shape·dtype — baseline PASS.
- Runtime: ORT 로드·실행·출력 스키마 `[1,84,8400] float32` — PASS.
- 재현성: 레지스트리 candidate 등록, 환경 fingerprint, 시드, 설정 기록. **발견**: ONNX export의 `date` metadata가 SHA를 바꾼다 → 정규화 체크섬 제안. 승격 정책 변경 없음.

## 11. Credit 전략

| 단계 | Credit | 내용 |
|---|---|---|
| E5-0 (본 문서) | 0 | 모델 획득·fingerprint·구조·fx precheck·baseline(mAP coco128, latency, memory, params/FLOPs)·역사 매핑·공식 경로 확인 — **완료** |
| E5-1 | 25 (클라이언트 사전 체크 상수; 5-B에서 실 차감 25 확인) | `model_fx.pt` **단일** `automatic_compression`(ratio 0.5, input [1,3,640,640], 단일 configuration). 별도 승인 후 1회 |
| E5-2 | 0 | 압축 body → head 재부착 → ONNX → 구조·params/FLOPs·ORT latency/memory·proxy·**mAP** → 프로파일 Gate → 회귀 |
| E5-3 | 필요 시 50 + 25 | convert + profile — 실 차감은 실행 전 재확인; YOLOv8 공식 워크플로우에는 없는 단계 |
| 원칙 | — | 1 real op → local validation → analysis → next decision. 동시 다중 실행 금지 |

## 12. GO 판단

| 조건 | 상태 |
|---|---|
| model ready | ✓ weights(sha 기록)·fx body+head meta 생성·구조 PASS |
| dataset / evaluation ready | △ coco128 smoke ✓; **COCO val2017 미다운로드**(release accuracy 전제) |
| baseline ready | ✓ (coco128 mAP, ORT latency/memory, params/FLOPs, fingerprint, registry candidate) |
| artifact path ready | ✓ 압축 입력 계약 = 5-B 성공 사례와 동일 |
| quality gate ready | △ 프로파일 존재; **accuracy 임계값은 example이며 검출용으로 재검토 필요**, 압축 body의 mAP 평가 helper(head 재부착) 미구현 |
| credit safety verified | ✓ 장부 불변(sha `2565dc03…`), API 호출 0, 키 미사용 |

**결정(5-E prep): CONDITIONAL GO.** 이후 해소 작업(0 Credit)에서 (1) head 재부착 helper와 mAP 파이프라인(`detection_head.py`, `detection_accuracy.py`, `yolov8_offline_detection_eval.py`)이 identity fixture로 동작 검증되고 (2) dataset 전략 OPTION A, (3) threshold 정책, (4) normalized checksum이 정리되었다. 그러나 **새 블로커**가 확인되었다: 공식 포크(8.0.108)가 만든 fx body가 upstream 모델과 기능적으로 동등하지 않다(포크 평가 0.135 vs upstream 0.444, proxy cosine 0.994 / rel diff 10.6 %; torch 버전은 원인이 아님). 그 시점의 결정은 CONDITIONAL GO였고, GO FOR E5-1 조건은 "upstream 코드 기반의 fx export(공식 recipe 재현)로 proxy cosine ≈ 1.0인 body를 확보"하는 것이었다(해소 문서 §4.2 R1, 0 Credit).

**R1 결과(2026-10-05, 0 Credit)**: 조건 충족. upstream 8.4.173 + 2단계 traceability patch로 만든 fx body(`yolov8n_fx_r1/model_fx.pt`, sha `d8e761da…`)는 upstream 모델과 pre-NMS **bitwise 동일**, one-image 검출 6=6 / 194=194(Δ 0), COCO128 mAP50-95 **0.44369 = 0.44369**(drop 0.0 pp), 2회 export bit-identical, torch 2.0.1 로드 PASS. 포크 블로커의 원인도 확정되었다(§13.2). → **결정: GO FOR E5-1.** E5-1 입력은 R1 산출물이며 옛 포크 산출물은 사용하지 않는다. 실행은 별도 승인 후 1회. 압축 직후 Gate FAIL(재학습 전)은 여전히 예상된 결과다.

## 13. FX traceability boundary와 upstream 호환 근거 (R1에서 확정)

### 13.1 Boundary
| 구분 | 내용 |
|---|---|
| **fx body 안** | upstream `DetectionModel` layer 0–21(backbone+neck; Conv/C2f/SPPF/Upsample/Concat → `torch.nn` leaf로 추적) + `Detect.cv2[i]`·`cv3[i]` 분기 concat → `[N, nc+4·reg_max, H/s, W/s]` × 3 |
| **fx body 밖** | DFL, anchors/strides(`make_anchors`), box decode(`dist2bbox`·stride), class sigmoid, **NMS**, 전처리. `netspresso_head_meta.json`(nc/nl/anchors/stride/strides/inplace)으로 재구성 |
| patch | `r1-upstream-body-wrapper-v1`: (1) wrapper 모듈(`_predict_once` 재생), (2) C2f 인스턴스에 upstream의 `forward_split` 바인딩. ultralytics 소스 수정 없음 |
| 계약 호환 | body/head 분할이 공식 포크 `export_netspresso()`와 동일 → 포크 `DetectionModel_netspresso` 재부착 경로에서 그대로 동작(포크 평가기 mAP 0.4448) |
| 압축 입력 | `outputs/models/yolov8n_fx_r1/model_fx.pt`(12,846,691 B, sha `d8e761da…`, 신뢰 목록 등록, git 미추적) |

### 13.2 Upstream 호환 근거 (실측)
| 근거 | 값 |
|---|---|
| pre-NMS raw/decoded 텐서 (R1 ↔ upstream, 9장) | bitwise 동일 (noise floor인 fused↔unfused는 rel 4.4e-7) |
| post-NMS one-image | 6=6 boxes(conf 0.25), 194=194(conf 0.001), 좌표·conf Δ 0 |
| COCO128 mAP50-95 (같은 평가기, rect=True) | 0.44369 = 0.44369, drop 0.0 pp |
| torch 2.0.1(ultralytics 없음) 로드·forward | PASS, torch 2.14 대비 max |Δ| 6.9e-5 |
| 재현성 | 2회 export raw SHA 동일 |
| 포크 블로커 원인 | **VERIFIED**: 포크 `C2f.forward`가 Bottleneck을 연쇄하지 않음(n ≥ 2인 layer 4·6에서 분기). 메모리 내 연쇄 의미 복원 시 차이 소멸(rel 1.5e-7) |

이 근거들은 "압축 입력이 upstream YOLOv8n과 같은 함수"임을 보이는 것이며, NetsPresso 서버 수락·압축 결과의 품질은 여전히 E5-1 실 실행으로만 확인된다.

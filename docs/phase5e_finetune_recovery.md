# Phase 5-E Fine-tuning Recovery — E5-1 압축 후보의 로컬 복구 실험 (0 Credit)

> 목적: E5-1에서 25 Credit으로 얻은 NetsPresso 압축 후보(mAP50-95 0.0)가 **로컬 fine-tuning만으로** 검출 품질을 회복하는지 확인한다. NetsPresso 호출 0, Credit 사용 0(장부 50 / 450 / 2 불변), HEAD `c8a6b04` 불변, 커밋·push 없음.
> 레코드: `reports/yolov8_e5_1_finetune/20261005T042904Z_phaseA/`, `reports/yolov8_e5_1_finetune/20261005T043342Z_phaseB/` (각 `finetune_config.json`, `finetune_result.json`, `finetune_accuracy.json`, `finetune_summary.html`). 산출물: `outputs/models/yolov8n_e5_1_finetuned/model_fx_phase{A,B}.pt` (git 미추적).
> 스크립트 `scripts/yolov8_e5_1_finetune.py`, 순수 로직 `framework/evaluation/recovery.py`, 테스트 `tests/unit/test_e5_1_finetune.py`(TC-Y8-FT-001…012). 모든 임계값은 PROJECT-DEFINED EXAMPLE.

---

> **최종 업데이트 — COCO val2017 독립 평가(§13)**: Phase C의 smoke 0.46688은 **unseen 데이터에서 0.00917**로 붕괴(baseline 0.36804, drop 35.9 pp) → **VAL2017_OVERFIT**. COCO128 회복은 128장 암기였다. 다음 단계 **HOLD**, NetsPresso 추가 실행 **HOLD**.
> **업데이트 — Phase C(continuation, global epoch 41–80) 완료**: smoke mAP50-95 **0.46688**(baseline 0.44369 초과, 회복률 105 %), 판정 **PHASE_C_GO_VAL2017**, 다음 단계 **COCO VAL2017**, NetsPresso 추가 실행 **HOLD**. 상세 §12. train == val이므로 baseline 초과는 **128장 암기(overfitting)의 징후**이며 일반화 증명이 아니다.

## 1. 판정(Phase B 시점): **PARTIAL_RECOVERY** → 다음 단계 **B. HOLD — MORE LOCAL FINE-TUNING**

| 모델 | 종류 | mAP50-95 | mAP50 | mAP75 | P | R |
|---|---|---|---|---|---|---|
| baseline | upstream YOLOv8n(원본, 불변) | **0.44369** | 0.60111 | 0.47223 | 0.6286 | 0.52902 |
| compressed | **E5-1 NetsPresso 생성 후보** (sha `21d8cbd1…`) | **0.0** | 0.0 | 0.0 | 0.00003 | 0.00223 |
| fine-tuned A | **로컬** 파생물: detect 분기만 30 epoch | 0.0348 | 0.05939 | 0.03153 | 0.62838 | 0.0452 |
| fine-tuned B | **로컬** 파생물: A에서 전층 40 epoch (sha `73a8f76f…`) | **0.28965** | 0.45843 | 0.33512 | 0.59832 | 0.40044 |

- baseline → compressed **−44.369 pp** · baseline → fine-tuned(B) **−15.404 pp** · compressed → fine-tuned(B) **+28.965 pp** · **회복률 65.28 %** (잃은 정확도 중 되찾은 비율)
- 남은 drop 15.4 pp > example 임계값 1.0 pp → RECOVERY_SUCCESS 아님. 40 epoch 종료 시점에도 상승 중(0.242 → 0.290)이고 loss도 감소 중(box 1.47 / cls 2.14 / dfl 1.60) → 더 긴 로컬 학습이 유효할 가능성이 높아 B를 권고(C/D/E 아님).
- **중요한 한계**: COCO128은 train == val(같은 128장). 모든 수치는 smoke/진단용이며 과대평가되어 있다. Release accuracy는 val2017에서만 판정한다. NetsPresso가 만든 산출물은 E5-1 후보까지이며 fine-tuned 산출물은 **로컬 파생물**이다.

## 2. 사전 검증 (학습 전)
| 항목 | 결과 |
|---|---|
| 장부 | used 50 / remaining 450 / operations 2 (sha `4b700de2…`), 실행 전후 동일 |
| E5-1 레코드 | operation automatic_compression, ratio 0.5, COMPLETED, 후보 sha `21d8cbd1…`, params 882,996, mAP50-95 0.0, source sha `d8e761da…` — 전부 일치 |
| 후보 identity | SHA·크기(3,693,813 B) 일치, 옛 포크 산출물 경로 금지 규칙 통과 |
| E5-1 파일 불변성 | `e5_1_result.json`, `execution_result.json`, `sdk_output.pt`의 SHA가 학습 전후 동일 (`e5_1_files_immutable: true`) |
| 학습 전 shape | body `[1,144,80/40/20]`, R1DecodeHead 재부착 후 `[1,84,8400]` |

## 3. 모델 구조와 학습 가능성 분석 (실제 후보 기준)
- 후보는 `torch.fx.GraphModule`(Conv2d 63 / BatchNorm2d 57 / SiLU / MaxPool2d / Upsample), 183개 파라미터 텐서, 882,996 params. NetsPresso가 모듈 이름을 평탄화(`layers_4_cv1_bn`, `cv2_0_0_conv` …)했고, 11개 텐서는 `requires_grad=False`(5,200 params)로 반환되었다.
- **Detect 분기 `cv2*`/`cv3*`(48 텐서, 257,176 params)는 압축 body 안에 있다**(포크 export 계약). decode head(DFL/anchors/sigmoid)는 학습 파라미터가 없다 → "head만 학습" = Detect 분기 학습.
- Phase A: `layers_*`(backbone+neck) 파라미터 고정 + **BN eval 모드**(pruned 채널 통계를 128장 배치로 흔들지 않음), `cv2*/cv3*`만 학습.
- Phase B: 전층 학습(SDK가 꺼 둔 11개 텍서 포함, 명시적으로 재활성화). Phase B 레코드의 `sdk_requires_grad_false_tensors: 183`은 Phase A 산출물 저장 시 전부 `requires_grad=False`로 저장한 상태를 읽은 것이며 SDK 상태가 아니다(기록 해석 주의).

## 4. 학습 설계 (결정적, CPU)
| 항목 | Phase A | Phase B |
|---|---|---|
| 시작 가중치 | E5-1 후보 `21d8cbd1…` | Phase A 산출물 `0a203872…` |
| 학습 텐서 | 48 / 183 (detect 분기) | 183 / 183 |
| epochs / batch / imgsz | 30 / 16 / 640 | 40 / 16 / 640 |
| optimizer | AdamW(β 0.9/0.999), lr 1e-3, warm-up 1 epoch → cosine 10 %, weight decay 5e-4(conv weight만), grad clip 10 | 동일(lr 1e-3) |
| loss | upstream `v8DetectionLoss`(box 7.5 / cls 0.5 / dfl 1.5, TAL topk 10) — body map → {boxes, scores, feats} 재배치만 추가 | 동일 |
| 데이터 | `coco128.yaml`(ultralytics coco128.zip, COCO 2017 train subset), train 128장 = val 128장, ultralytics 기본 train 증강(mosaic 1.0, HSV, fliplr 0.5, scale 0.5, translate 0.1 …), workers 0 | 동일 |
| seed / 결정성 | 0, `init_seeds(deterministic=True)` | 동일 |
| 환경 | torch 2.14.1+cpu, ultralytics 8.4.173, Python 3.11.9, CPU | 동일 |
| 학습 시간 | 222.8 s (≈6 s/epoch) | 509.1 s (≈12 s/epoch) |
| 산출물 | `model_fx_phaseA.pt` 3,713,403 B sha `0a203872…` | `model_fx_phaseB.pt` 3,714,299 B sha `73a8f76f…` |

Phase A를 먼저 측정했고(0.0348, 회복 7.8 %, RECOVERY_FAILED), 상승 추세와 CPU 여유를 근거로 Phase B를 실행했다.

## 5. 학습 곡선 (val은 5 epoch마다, 같은 128장)
| Phase | epoch | box | cls | dfl | mAP50-95 | mAP50 |
|---|---|---|---|---|---|---|
| A | 5 / 10 / 20 / 30 | 2.67 / 2.48 / 2.37 / 2.30 | 4.36 / 3.99 / 3.81 / 3.71 | 2.43 / 2.28 / 2.28 / 2.23 | 0.0027 / 0.0117 / 0.0199 / **0.0348** | 0.006 / 0.022 / 0.038 / 0.059 |
| B | 5 / 10 / 20 / 30 / 40 | 2.07 / 1.95 / 1.69 / 1.54 / 1.47 | 3.62 / 3.10 / 2.67 / 2.30 / 2.14 | 2.06 / 1.91 / 1.76 / 1.65 / 1.60 | 0.0088 / 0.0381 / 0.1244 / 0.1932 / **0.2897** | 0.021 / 0.070 / 0.231 / 0.323 / 0.458 |

## 6. 산출물 검증 / Params / FLOPs
| 항목 | Phase B 산출물 |
|---|---|
| 로드·구조 | `validate_pt` PASS, `graph_module`; forward `[1,144,80/40/20]`; 재부착 `[1,84,8400]` |
| params | **882,996 = E5-1 후보**(변동 0) |
| FLOPs(2×MACs, 로컬 ONNX) | **3.832 G = E5-1 후보**(SDK 3.875 G) — 아키텍처 불변 PASS(허용 0.5 %) |
| 신뢰 | 이 실행에서 로컬 생성(trust list 미등록; 레지스트리 `local_finetune:73a8f76f4c88` candidate) |

## 7. Output equivalence proxy (accuracy 아님)
| 비교 | min cosine | max |Δ| | relative |
|---|---|---|---|
| ORT 동일 입력: baseline vs fine-tuned B | 0.9829 | 532.6 | 0.184 |
| 실제 9장 decoded: baseline vs fine-tuned B | 0.9920 | — | — |
| 실제 9장: baseline vs compressed (E5-1 참조) | 0.9840 | — | — |

ORT 난수 입력 proxy는 여전히 0.99 미만(local_eval Gate FAIL)인데 mAP는 0 → 0.29로 올랐다. proxy는 분포 밖 입력에 대한 유사도이며 검출 품질을 대변하지 않는다는 점이 다시 확인된다.

## 8. 로컬 ORT latency / memory (개발 머신, CPUExecutionProvider, 격리 프로세스, 같은 세션)
| | median | p95 | peak RSS Δ |
|---|---|---|---|
| baseline (R1 body+head) | 18.83 ms | 42.88 ms | 108.65 MB |
| fine-tuned B | **13.91 ms** (−26 %) | 35.94 ms | **93.38 MB** (−14 %) |

E5-1 때(23.78 → 16.52 ms)와 같은 방향·비슷한 비율. 세션 간 절대값 변동이 크므로 상대 비교만 의미가 있다.

## 9. Quality Gate / 결함
| Profile | A | B | 사유(B) |
|---|---|---|---|
| compression | PASS | **PASS** | size −71 %, artifact, structural PASS |
| local_eval | FAIL | **FAIL** | proxy 0.9829 < 0.99 (required) |
| release | FAIL | **FAIL** | accuracy drop 15.40 pp > 1.0 pp; checksum 미검증; reproducibility runs=1 (val2017 미평가) |

결함(B): `E5-1-FT-B-release-ACCURACY_REGRESSION` [HIGH], `E5-1-FT-B-local_eval-ACCURACY_REGRESSION` [MEDIUM]. 역사적 E5-1 결함(`E5-1-release-ACCURACY_REGRESSION`, `E5-1-local_eval-ACCURACY_REGRESSION`)은 삭제·수정하지 않았다. **복구 평가**: "E5-1 ACCURACY_REGRESSION은 로컬 fine-tuning으로 부분 완화되었으나 회귀는 지속" — pruning-without-fine-tuning 가설은 복구 증거로 **지지됨**(0 → 0.29).

## 10. 재현성 / 추적성
- 학습 실행 1회/phase, seed 0, 결정성 플래그 on; CPU 학습의 비트 재현성은 **미검증**(2회차 없음). Level NOT_VERIFIED.
- 체인: `yolov8n.pt f59b3d83…` → R1 fx `d8e761da…` → E5-1 `automatic_compression`(NetsPresso) → 후보 `21d8cbd1…` → **로컬** Phase A `0a203872…` → **로컬** Phase B `73a8f76f…` → COCO128 3-way → Gate → 권고 B. `traceability.complete = true`.

## 11. 다음 단계 권고 (Credit 사용 없음)
**B. HOLD — MORE LOCAL FINE-TUNING.** 근거: 짧은 CPU 학습(≈12 분)으로 65 %를 회복했고 곡선이 수렴 전이다. 추가 로컬 학습(더 많은 epoch, 가능하면 더 큰 학습 세트)으로 1.0 pp 임계값 접근 여부를 확인한 뒤에만 A(val2017)로 간다. C(ratio 하향, +25 Credit)와 E5-3(convert+profile, 75 Credit)은 지금 정당화되지 않는다. 모델 변경(D)·경로 포기(E)는 근거 없음. 이번 작업에서 E5-2/E5-3는 실행하지 않았다.

## 12. Phase C — continuation training (global epoch 41–80, 0 Credit)

> 레코드: `reports/yolov8_e5_1_finetune/20261005T051007Z_phaseC/` (`finetune_config.json`, `training_history.json`, `validation_results.json`, `phaseC_result.json`, `environment_fingerprint.json`, `artifact_validation.json`, `recovery_summary.json`, `finetune_summary.html`). 산출물 `outputs/models/yolov8n_e5_1_finetuned/model_fx_phaseC.pt` (git 미추적). 안전장치: `NETSPRESSO_API_KEY` 존재 시 즉시 abort(값 미독), `netspresso` 모듈 import 감지 시 abort, 장부 50/450/2 전후 동일, E5-1 파일 SHA 불변.

### 12.1 사전 검증 (전부 일치)
HEAD `c8a6b04` · 장부 50/450/2 · E5-1 레코드(op/ratio/status/sha/params/mAP 0.0/source) · E5-1 후보 sha `21d8cbd1…` · Phase A `0a203872…` · **Phase B `73a8f76f…`(재계산, B 레코드와 일치, continuation 입력)** · Phase B params 882,996 · body `[1,144,80/40/20]` · 재부착 `[1,84,8400]` · 기존 report 존재.

### 12.2 설정
Phase B와 동일한 학습 의미(AdamW lr 1e-3, β 0.9/0.999, wd 5e-4 conv만, warm-up 1 epoch → cosine 10 %, grad clip 10, v8DetectionLoss 7.5/0.5/1.5, batch 16, imgsz 640, seed 0, workers 0, CPU, torch 2.14.1 / ultralytics 8.4.173), 시작 가중치 = Phase B 산출물, 전층 학습, `epoch_offset 40`(global 41–80), validation 5 epoch마다 + 최종, plateau 조건(3연속 개선 < 0.005), NaN/Inf·params 변화 시 중단, best checkpoint 보존·rollback. 아키텍처 변경·재압축·head 재생성 없음.
**관찰(1-epoch smoke)**: cosine 스케줄을 lr 1e-3에서 재시작하면 첫 epoch 후 0.290 → 0.105로 일시 하락(warm-restart transient)한다. 사양 기본값을 유지했고 best/rollback 로직이 하한을 보장한다.

### 12.3 학습 곡선 (global epoch)
| global | box | cls | dfl | mAP50-95 | mAP50 | 비고 |
|---|---|---|---|---|---|---|
| 45 | 1.698 | 2.637 | 1.735 | 0.0940 | 0.1712 | restart transient 회복 중 |
| 50 | 1.622 | 2.249 | 1.630 | 0.1522 | 0.2921 | |
| 55 | 1.542 | 2.113 | 1.628 | 0.1953 | 0.3404 | |
| 60 | 1.450 | 2.019 | 1.541 | 0.2663 | 0.4571 | |
| 65 | 1.456 | 1.910 | 1.559 | 0.3254 | 0.5170 | Phase B best(0.2897) 초과 |
| 70 | 1.299 | 1.724 | 1.463 | 0.3991 | 0.6212 | **M1(≥0.35) 도달** |
| 75 | 1.311 | 1.665 | 1.465 | 0.4304 | 0.6498 | **M2(≥0.40) 도달** |
| 80 | 1.272 | 1.629 | 1.449 | **0.4669** | 0.6783 | **M3(≥0.43369) 도달**, best, rollback 불필요 |

plateau 미발생(최근 3개 validation 개선 +0.074 / +0.031 / +0.036), 조기 종료 없음, 496.5 s.

### 12.4 다자 비교 (동일 조건: ultralytics 8.4.173, COCO128, imgsz 640, conf 0.001, IoU 0.7, rect=True, batch 8)
| 모델 | 종류 | mAP50-95 | mAP50 | mAP75 | P | R | params | FLOPs(2×MACs) | size |
|---|---|---|---|---|---|---|---|---|---|
| baseline | upstream YOLOv8n | 0.44369 | 0.60111 | 0.47223 | 0.6286 | 0.52902 | 3,157,184 | 8.743 G | 12,846,691 B |
| E5-1 | NetsPresso 생성 후보 | 0.0 | 0.0 | 0.0 | 0.00003 | 0.00223 | 882,996 | 3.832 G | 3,693,813 B |
| Phase A | 로컬 파생물 | 0.0348 | 0.05939 | 0.03153 | 0.62838 | 0.0452 | 882,996 | 3.832 G | 3,713,403 B |
| Phase B | 로컬 파생물 | 0.28965 | 0.45843 | 0.33512 | 0.59832 | 0.40044 | 882,996 | 3.832 G | 3,714,299 B |
| **Phase C** | 로컬 파생물 (sha `dba5cd21…`) | **0.46688** | 0.67827 | 0.53599 | 0.67678 | 0.63216 | 882,996 | 3.832 G | 3,714,299 B |

- delta_from_phaseB **+17.72 pp**, delta_from_baseline **+2.32 pp**, recovery_rate **105.2 %**, remaining_gap **−0.0232**(음수 = baseline 초과).
- **해석**: COCO128은 train == val이다. 80 epoch 동안 같은 128장을 학습했으므로 baseline 초과는 **암기 효과**로 봐야 하며 일반화 성능을 전혀 증명하지 않는다. 이 수치로 Release PASS를 선언하지 않는다. 다음 증거는 **학습에 쓰지 않은 데이터(COCO val2017)** 평가다.

### 12.5 Proxy / 성능 / Gate
| 항목 | 값 |
|---|---|
| proxy ORT baseline vs C | min cosine **0.9903**(≥ 0.99 → local_eval proxy PASS), max |Δ| 311.0, rel 0.139 — proxy이며 accuracy 아님 |
| proxy 실제 9장 | baseline vs C 0.9913 · E5-1 vs C 0.9753 · **B vs C 0.9977** |
| ORT latency(개발 머신, 격리, 같은 세션) | baseline 18.65 ms(p95 80.36, 노이즈) · E5-1 11.49(37.0) · B 16.05(29.95) · **C 14.40 ms (p95 31.16)** · RSS 107.05 / 93.24 / 93.53 / **93.24 MB** |
| params / FLOPs / shape | 882,996 / 3.832 G / `[1,144,…]`→`[1,84,8400]` — 아키텍처 drift 0 % (허용 0.5 %) |
| Quality Gate | compression **PASS** · local_eval **PASS**(proxy 0.9903, latency −23 %, RSS −13 %, size −71 %) · release **FAIL**(checksum 미검증 — promoted baseline 없음, reproducibility runs=1) |
| 결함 | compression/local_eval 없음; release: UNCLASSIFIED(결측: checksum·reproducibility) — accuracy 회귀 결함은 smoke 기준으로 소멸, 역사적 E5-1 결함 레코드는 불변 |
| 재현성 | 1회 실행, seed 0 → NOT_VERIFIED. 레지스트리 `local_finetune:dba5cd214f0f` candidate(승격 없음) |

### 12.6 판정과 다음 단계

> **후속 결과(§13)**: 아래 PHASE_C_GO_VAL2017 판정에 따라 수행한 COCO val2017 독립 평가에서 Phase C는 **0.00917**(baseline 0.36804)로 붕괴했다. 이 절의 "RECOVERY_SUCCESS(smoke)"는 COCO128 train == val 기준의 진단값이며 최종 판정은 **VAL2017_OVERFIT / Release FAIL**이다.
- **PHASE_C_GO_VAL2017** (CASE A: best 0.46688 ≥ 0.43369). Release PASS 금지 — 필요한 추가 증거: COCO val2017 accuracy(학습 미사용 데이터), artifact checksum/promotion, 재현성(2회차), release gate.
- 다음 단계(단일 선택): **COCO VAL2017** — val2017 다운로드·평가는 0 Credit이지만 별도 결정(용량 1.2 GB)이다.
- **NetsPresso 추가 실행: HOLD.** E5-2/E5-3·ratio 변경·quantize·convert·profile은 val2017 결과와 비용/편익 비교 후 별도 승인 대상이다.
- 추적성: `yolov8n.pt f59b3d83…` → R1 `d8e761da…` → E5-1(NetsPresso) `21d8cbd1…` → A `0a203872…` → B `73a8f76f…` → **C `dba5cd21…`** → COCO128 평가 → Gate → PHASE_C_GO_VAL2017 / HOLD.

## 13. COCO val2017 독립 평가 — unseen 데이터 검증 (0 Credit, 학습 없음)

> 레코드: `reports/yolov8_e5_1_finetune/20261005T053121Z_val2017/` (`dataset_manifest.json`, `evaluation_config.json`, `validation_{baseline,e5_1,phase_c}.json`, `validation_results.json`, `model_integrity.json`, `environment_fingerprint.json`, `val2017_summary.json`, `val2017_summary.html`). 스크립트 `scripts/yolov8_val2017_eval.py`(prepare / evaluate / summarize, 모델별 독립 프로세스), 헬퍼 `framework/evaluation/recovery.py`(`classify_unseen_recovery`, `generalization_gap`, `unseen_next_step`), 테스트 `tests/unit/test_val2017_eval.py`.
> 가드: `NETSPRESSO_API_KEY` 존재 시 abort(값 미독), `netspresso` import 감지 시 abort, `torch.set_grad_enabled(False)`(optimizer/backward 코드 없음), 장부 50/450/2 전후 동일, 모델 SHA 전후 동일.

### 13.1 데이터셋
| 항목 | 값 |
|---|---|
| 출처 | `images.cocodataset.org/zips/val2017.zip` (815,585,330 B, sha `4f7e2ccb…`) + ultralytics `coco2017labels.zip` (48,639,045 B, sha `51a5175c…`); train2017(19 GB)은 받지 않음 |
| 규모 | 5,000 images / `instances_val2017.json` 36,781 annotations / 80 classes; ultralytics 라벨 4,952 파일 36,335 boxes(배경 이미지 48장) |
| 평가 yaml | `datasets/coco_val2017_eval.yaml` — `val: val2017.txt`(train 키는 형식상 동일 경로, 학습 없음), git 미추적 |
| **오염 검사** | COCO128 학습 128장(train2017 subset) vs val2017 5,000장: 파일명 겹침 **0**, COCO image id 겹침 **0**, 내용 SHA-256 겹침 **0** → 완전 분리. COCO128 train == val 사실은 그대로 유지 |

### 13.2 결과 (동일 조건: ultralytics 8.4.173, imgsz 640, conf 0.001, IoU 0.7, rect=True, batch 8; 모델별 독립 프로세스, baseline → E5-1 → Phase C 순)
| Model | mAP50-95 | mAP50 | mAP75 | P | R | 시간 |
|---|---|---|---|---|---|---|
| Baseline (R1 body + head, sha `d8e761da…`) | **0.36804** | 0.51861 | 0.40003 | 0.63496 | 0.47375 | 142.8 s |
| E5-1 (NetsPresso 후보, `21d8cbd1…`) | **0.0** | 0.0 | 0.0 | 0.00002 | 0.00186 | 154.2 s |
| Phase C (로컬 파생물, `dba5cd21…`) | **0.00917** | 0.02101 | 0.00692 | 0.04391 | 0.04623 | 97.7 s |

sanity: baseline 0.368은 ultralytics 공개 YOLOv8n val2017 37.3과 0.5 pp 이내로 일치 → 평가 파이프라인이 정상임을 뜻한다.

### 13.3 COCO128 vs val2017 (generalization gap)
| Model | COCO128 mAP50-95 | val2017 mAP50-95 | gap (pp) | unseen/smoke |
|---|---|---|---|---|
| Baseline | 0.44369 | 0.36804 | 7.57 | 83.0 % |
| E5-1 | 0.0 | 0.0 | 0.0 | — |
| Phase B | 0.28965 | (불필요) | — | — |
| **Phase C** | **0.46688** | **0.00917** | **45.77** | **2.0 %** |

overfit indicator: Phase C의 smoke 우위(baseline 초과)가 unseen에서 완전히 사라짐; Phase C gap이 baseline 자체 gap(7.6 pp)보다 **38.2 pp** 크다 → **COCO128 암기**.

### 13.4 회복/분류
E5-1 → Phase C **+0.917 pp** · Phase C → baseline **−35.887 pp** · remaining drop **35.887 pp** · recovery rate **2.5 %**. 분류 **CASE C → VAL2017_OVERFIT** (drop > 3.0 pp이고 excess gap 38.2 > 10 pp; PROJECT-DEFINED EXAMPLE 임계값). Proxy(unseen 8장, decoded): baseline vs C 0.9901, E5-1 vs C 0.9727, baseline vs E5-1 0.9801 — cosine 0.99라도 mAP 0.009 → **proxy ≠ accuracy** 재확인.

### 13.5 Quality Gate / 무결성 / Credit
| 항목 | 값 |
|---|---|
| compression | PASS (산출물 구조·크기) |
| local_eval | PASS — latency/memory/proxy는 Phase C 측정값 참조(재측정 없음), accuracy는 이 프로파일에서 optional(WARN) |
| release | **FAIL** — accuracy drop 35.89 pp > 1.0 pp(unseen), checksum 미검증(promoted baseline 없음), reproducibility runs=1 → 결함 `VAL2017-release-ACCURACY_REGRESSION` [HIGH] |
| 무결성 | baseline/E5-1/Phase C SHA 전후 동일; Phase A/B 파일 미수정; baseline promotion 수행 안 함 |
| Credit | 50 / 450 / 2 → 50 / 450 / 2, API 0, SDK 0, 학습 0 |

### 13.6 결론과 다음 단계
- **VAL2017_OVERFIT**: 8 분 CPU 학습으로 얻은 COCO128 0.467은 128장 암기였고, 압축 후보의 실제(unseen) 검출 정확도는 거의 회복되지 않았다(0.0 → 0.009). "COCO128 mAP 상승 ≠ Release PASS"가 수치로 확인되었다.
- **다음 단계: HOLD.** NetsPresso 추가 실행(E5-2/E5-3/ratio 변경) **HOLD**. 로컬 복구를 계속하려면 **학습 데이터 전략 자체를 바꿔야** 한다(수천~수만 장 규모의 train 데이터, val과 분리된 holdout으로 early stopping). 그 전까지 Credit 결정은 보류한다.
- QA 관점의 가치: 압축 → 복구 → 검증 체인에서 **unseen validation이 없으면 복구 성공을 오판**한다는 것을 재현 가능한 증거로 남겼다.

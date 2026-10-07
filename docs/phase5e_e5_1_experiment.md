# Phase 5-E E5-1 — 단일 실 NetsPresso `automatic_compression` (R1 YOLOv8n fx body)

> 실행: 2026-10-05 04:00 UTC, **실 NetsPresso 작업 정확히 1회**(automatic_compression, ratio 0.5, `[1,3,640,640]`), 이후 모든 검증은 0 Credit 로컬.
> Credit: 25 / 475 / 1 → **50 / 450 / 2** (계정 잔액 475 → 450 관측, 클라이언트 상수 25와 **일치**). HEAD `c8a6b04` 불변, 커밋·push 없음.
> 레코드: `reports/yolov8_e5_1/20261005T040009Z/{e5_1_precheck.json, e5_1_execution.json, e5_1_result.json, e5_1_summary.html}`, 실 실행 `reports/real_runs/20261005T040009Z_automatic_compression/`, 사전 게이트 dry-run `reports/yolov8_e5_1/20261005T035912Z/e5_1_precheck.json`.
> 스크립트: `scripts/run_e5_1_experiment.py`(게이트 → `run_real_netspresso.py` → 어댑터 → SDK, 회계), `scripts/yolov8_e5_1_local_validation.py`(0 Credit 검증), 순수 로직 `framework/evaluation/input_gate.py`, 테스트 `tests/unit/test_e5_1.py`.
> 모든 임계값은 PROJECT-DEFINED EXAMPLE이며 Nota 공식 기준이 아니다.

---

## 1. 판정

| 축 | 상태 |
|---|---|
| Execution (NetsPresso) | **COMPLETED** — SDK 1.17.0, 28.9 s, 모델 ID `498c4990…`(업로드) → `491890fb…`(압축) |
| Artifact | **PASS** — fx GraphModule 3,693,813 B, sha `21d8cbd1…`, forward `[1,144,80/40/20]`, 실행 레코드 SHA와 일치, 동반 ONNX checker PASS |
| Validation (파이프라인) | **PASS** — params/FLOPs 독립 검증 PASS, head 재부착 PASS `[1,84,8400]`, COCO128 평가 완료, ORT 측정 완료 |
| Quality Gate | compression **PASS** / local_eval **FAIL**(proxy cosine 0.984 < 0.99) / release **FAIL**(accuracy −44.37 pp, checksum 미검증, 재현성 1회) |
| Release | **FAIL** |
| Next step | **HOLD** |

"NetsPresso 실행 PASS ≠ 후보 Release PASS"를 그대로 보여주는 결과다. 압축 자체는 계약대로 동작했고 산출물은 유효하지만, **fine-tuning 없는 pruning 산출물은 검출 기능을 상실**했다(mAP50-95 0.0).

## 2. Traceability chain

| 링크 | 값 |
|---|---|
| source model | `yolov8n.pt` sha `f59b3d83…` (ultralytics 체크포인트 8.0.0.dev0) |
| R1 fx artifact | `outputs/models/yolov8n_fx_r1/model_fx.pt`, patch `r1-upstream-body-wrapper-v1`, 검증 `reports/yolov8_baseline/20261005T034329Z_r1/` |
| R1 SHA / size | `d8e761dae29301ef51df9e7679c729338a522d397e271a900d453d171e1c022b` / 12,846,691 B — **실행 직전 재계산·일치**, 옛 포크 산출물 경로는 게이트에서 금지 |
| 입력 게이트 | identity PASS → `validate_pt` graph_module(torch 2.0.1) → `precheck_fx_bundle` **READY** → 장부 prior ops == 1 → `--confirm-credit-use` |
| NetsPresso 실행 | `run_real_netspresso.py --model-path …yolov8n_fx_r1/model_fx.pt --input-shape 1,3,640,640 --compression-ratio 0.5 --model-name yolov8n`, status completed, method PR_L2 |
| candidate | `reports/real_runs/20261005T040009Z_automatic_compression/sdk_output/sdk_output.pt` sha `21d8cbd1361cfd3ef55c0c132d11f3fad294dbde6f175e8082bc0233b860e0f1` (+ `sdk_output.onnx`) |
| local validation | `reports/yolov8_e5_1/20261005T040009Z/e5_1_result.json` |
| accuracy | COCO128 mAP50-95 0.44369 → **0.0** (drop 44.369 pp, MEASURED) |
| Quality Gate | compression PASS / local_eval FAIL / release FAIL |
| credit | `reports/credit_usage.json` 2번째 entry: actual 25 (계정 475 → 450), estimated 25, MATCH |

"어떤 입력이 이 후보를 만들었는가"는 레코드만으로 답할 수 있다(`e5_1_result.json.traceability.complete = true`).

## 3. 실 실행과 Credit

| 항목 | 값 |
|---|---|
| 경로 | QA Engine(`run_e5_1_experiment.py`) → `run_real_netspresso.py` → `NetsPressoAdapter` → `NetsPresso.compressor_v2().automatic_compression` → Result Model → 장부 |
| 서비스 작업 수 / 인증 세션 | **1 / 1**, 재시도 없음 (`operations_executed == 1`, 로그에 호출 1회) |
| SDK 메타데이터 | status `completed`, task `compress`, method `PR_L2`, ratio 0.5, is_retrainable false, framework pytorch, input `{1,3,[640,640]}` |
| SDK 보고 | original params 3,157,184 / FLOPs 8.80 G / 12.25 MB → compressed params **882,996** / FLOPs **3.87 G** / 3.52 MB |
| Credit | 계정 475 → 450 (SDK user/credit 조회), 관측 delta **25 = 클라이언트 상수** → `MATCH`; 장부 used 50 / remaining 450 / ops 2; 장부 전이 검증(1 entry 추가, used Δ = 관측 Δ) OK |
| 비밀 | API 키는 `.env`→환경변수→어댑터만; 레코드·HTML·로그에 키·절대경로 없음(테스트로 강제) |

## 4. 로컬 0-Credit 검증 (upstream ultralytics 8.4.173, torch 2.14)

### 4.1 산출물
| 항목 | 결과 |
|---|---|
| `sdk_output.pt` | 신뢰 목록 등록 후 전체 unpickle → `torch.fx.GraphModule`, params 882,996, forward `[1,144,80,80]/[1,144,40,40]/[1,144,20,20]`(계약 유지) |
| `sdk_output.onnx` | `onnx.checker` PASS, 출력 3개 `[N,144,80/40/20]` |
| SHA 체인 | 파일 SHA == `execution_result.json` 기록 SHA |

### 4.2 Params / FLOPs
| | SDK | 독립 | 검증 |
|---|---|---|---|
| params baseline | 3,157,184 | 3,157,184 (torch numel, R1 body) | PASS |
| params candidate | 882,996 | 882,996 | PASS (**−72.03 %**) |
| FLOPs baseline | 8.801 G | 2×MACs 8.743 G (로컬 ONNX export) | PASS |
| FLOPs candidate | 3.875 G | 2×MACs 3.832 G (로컬 export = SDK onnx) | PASS (**−55.97 %**) |

2×MACs 관례는 5-B/5-D/E5-1에서 일관되게 관측된 것이며 공식 문서화된 정의로 주장하지 않는다. 전체 모델 참조값(3,157,200 params / 8.855 GFLOPs)과 body 값의 차이는 DFL 상수 16개와 decode 연산이다.

### 4.3 Head 재부착과 COCO128 smoke accuracy (동일 조건: imgsz 640, conf 0.001, IoU 0.7, rect=True, 같은 평가기 빌드)
| 모델 | mAP50-95 | mAP50 | mAP75 | P | R |
|---|---|---|---|---|---|
| baseline (R1 record, upstream) | **0.44369** | 0.60111 | 0.47223 | 0.6286 | 0.52902 |
| candidate (압축 body + R1DecodeHead) | **0.0** | 0.0 | 0.0 | 0.00003 | 0.00223 |
| delta | absolute −0.44369, **drop 44.369 pp**, relative −100 % | | | | |

Head 재부착은 R1과 같은 코드 경로(`R1DecodeHead`/`R1Detector`)로 PASS. 1.0 pp 임계값은 example이며 smoke 데이터의 진단용이다 — 그러나 0.0은 임계값과 무관하게 "검출 기능 상실"을 뜻한다.

### 4.4 Output equivalence proxy (accuracy 아님)
| 비교 | min cosine | max |Δ| | relative |
|---|---|---|---|
| ORT, 동일 결정적 입력 (Gate 신호) | 0.9843 | 449.2 | 0.179 |
| COCO128 실제 9장, decoded pre-NMS | 0.9840 | — | 0.179 |

cosine 0.98은 "구조가 비슷하게 남아 있다"는 신호일 뿐이며, 같은 산출물의 mAP가 0.0이므로 **proxy만으로 기능 동등성을 말할 수 없다**는 것을 실측으로 보여준다(Gate `local_eval`도 FAIL).

### 4.5 로컬 ORT latency / memory (개발 머신, 격리 프로세스, body+head ONNX)
| | median | p95 | peak RSS Δ |
|---|---|---|---|
| baseline (R1 body+head) | 23.78 ms | 58.84 ms | 107.08 MB |
| candidate | **16.52 ms** (−30.5 %) | 36.66 ms | **93.36 MB** (−12.8 %) |

5-D의 샘플 모델(FLOPs −52 %에도 CPU 지연 개선 없음)과 달리 YOLOv8n에서는 지연이 줄었다. 단, run-to-run 변동(이전 측정 21.3→13.1 ms)이 크므로 참고 수치다.

## 5. Quality Gate / 결함

| Profile | Overall | 사유 |
|---|---|---|
| compression | **PASS** | model_size −71 % PASS, artifact PASS(checksum optional), structural PASS; accuracy는 WARN(optional) |
| local_eval | **FAIL** | output_equivalence_proxy 0.9843 < 0.99(required); latency/memory/size PASS |
| release | **FAIL** | accuracy drop 44.37 pp > 1.0 pp; artifact checksum 미검증(promoted baseline 없음); reproducibility runs=1 |

결함: `ACCURACY_REGRESSION` (local_eval MEDIUM / release HIGH). **의심 원인**: 구조적 pruning(PR_L2, ratio 0.5)을 공식 ModelZoo-YOLOv8 워크플로우가 전제하는 fine-tuning 없이 평가 — 재학습 전 accuracy 붕괴는 pruning의 예상 동작. confidence **MEDIUM**, **verified: false**(검증하려면 후보를 fine-tuning한 뒤 재평가해야 하며 이번 Phase에서는 수행하지 않았다). 결측 측정을 회귀로 분류한 항목은 없다.

## 6. 재현성 / 레지스트리

execution_runs = **1**(의도적으로 2회차 실행 안 함). 레지스트리에 `automatic_compression:21d8cbd1361c` **candidate**로 등록(source `yolov8n_fx_r1/model_fx.pt sha256=d8e761da…`). 이 단일 실행으로 expected checksum/baseline이 되지 않는다.

## 7. 다음 단계 권고: **HOLD**

- 압축은 계약대로 성공했고 params −72 % / FLOPs −56 % / 로컬 지연 −30 %를 얻었지만, 재학습 전 산출물은 검출을 못 한다. 이 상태에서 E5-3(convert + profile, 75 Credit)은 **시기상조**다.
- 권고 순서: (a) 후보를 COCO128(또는 더 큰 세트)으로 **로컬 fine-tuning**(0 Credit, CPU 시간) 후 같은 조건으로 재평가 → accuracy 회복 정도 확인; (b) 회복이 불충분하면 compression ratio 하향(추가 25 Credit; 역시 fine-tuning 필요); (c) 모델 변경은 불필요 — 문제는 모델이 아니라 빠진 재학습 단계다.
- E5-2(로컬 전 단계)는 이번 검증으로 실질적으로 수행되었고, release 판정은 val2017 + fine-tuned 후보에서만 한다.
- 이번 작업에서 E5-2/E5-3는 실행하지 않았고 추가 Credit은 쓰지 않았다.

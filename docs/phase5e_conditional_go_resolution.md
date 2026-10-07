# Phase 5-E Conditional GO Resolution (0 Credit)

> 목적: Phase 5-E Preparation의 CONDITIONAL GO 조건 네 가지(head 재부착 helper, release accuracy dataset, accuracy threshold, normalized checksum)를 **NetsPresso API 호출 없이** 해소하고, 실제 압축 산출물이 없어도 압축 결과를 평가할 수 있는 경로가 완성되었는지 검증한다.
> Credit: 25 used / 475 remaining / 1 operation — 변동 없음. HEAD `c8a6b04` 불변. 커밋·push 없음.
> 레코드: `reports/yolov8_baseline/20261005T030029Z_offline_eval/offline_detection_eval.json`, `reports/yolov8_baseline/20261005T024950Z/baseline.json`; R1(§11): `…T034329Z_r1/`, `…T034230Z_r1_torch201_load/`, `…T034416Z_r1_fork_crosscheck/`, `…T034240Z_offline_eval/`.
> **업데이트(R1 완료)**: §4의 블로커는 §11에서 0 Credit으로 해소되었고 원인이 확정되었다. 최종 결정은 **GO FOR E5-1**(별도 승인 후 단일 실행).

---

## 1. 결정: ~~CONDITIONAL GO~~ → **GO FOR E5-1** (R1 완료 후, §11)

> 아래 본문(§1–§10)은 R1 이전의 판정 근거를 그대로 보존한다. R1 결과와 최종 판정은 §11.

네 조건 중 세 가지(B·C·D)는 해소되었고, 조건 A(head 재부착 helper)는 **구현·동작 검증은 완료**되었지만, 그 과정에서 더 근본적인 블로커가 발견되었다: **공식 포크(`ultralytics_nota` 8.0.108)가 만든 fx body가 upstream 모델과 기능적으로 동등하지 않다**(§4). 압축 입력 산출물의 정합성이 확립되지 않은 상태에서 25 Credit을 쓰면, 그 결과의 accuracy는 해석할 수 없다. 블로커는 0 Credit으로 해결 가능하므로 NO-GO가 아닌 CONDITIONAL GO다.

## 2. 조건별 결과

| 조건 | 상태 | 구현 / 근거 |
|---|---|---|
| A. Detection head re-attach helper | **READY (경로)** / 블로커 §4 | `framework/evaluation/detection_head.py`(head meta 파싱·입력/출력 schema·precheck), `scripts/yolov8_offline_detection_eval.py`(포크 `YOLO_netspresso`/`DetectionModel_netspresso`로 재부착 → `DetectionValidator`). identity fixture: precheck READY, decoded `[1,84,8400]` PASS, COCO128 val 완료 |
| B. Release accuracy dataset | **결정: OPTION A** | Smoke = COCO128(확보), Release = COCO val2017(승인 후 다운로드). 비교표 §6 |
| C. Accuracy pipeline (mAP50-95 primary; mAP50/mAP75/P/R) | **READY** | `framework/evaluation/detection_accuracy.py`: `DetectionMetrics`, `compare_detection_accuracy`(baseline/candidate/absolute_delta/drop_pp/relative_delta_percent, MEASURED / N/A / NOT_COMPARABLE), `Metrics.accuracy`로 Gate 연결 |
| D. Accuracy threshold | **정책 문서화** | 1.0 pp 유지 = PROJECT-DEFINED EXAMPLE, 적용 조건 §7 |
| E. Normalized ONNX checksum (선택) | **구현** | `framework/evaluation/onnx_checksum.py`: `raw_sha256` 보존 + `normalized_sha256`(policy `onnx-metadata-v1`, version 1) |

## 3. Q1–Q5 (실제 코드 근거)

- **Q1 `model_fx.pt`**: 포크 `export_netspresso()`는 `deepcopy(self.model).float().train()`을 `fx.Tracer().trace(...)`로 추적해 `GraphModule`로 저장한다. train 모드이므로 `Detect` head는 decode 없이 per-scale raw map(`nl`개, 채널 `nc + 4·reg_max = 144`)을 반환한다 → 측정 출력 `[1,144,80,80] / [1,144,40,40] / [1,144,20,20]`.
- **Q2 head metadata**: `nc, nl, anchors, stride, strides, inplace` (detect task). 측정값 nc 80, nl 3, stride [8,16,32].
- **Q3 복원에 필요한 정보**: body GraphModule + 위 meta(+ reg_max 16 상수). class names는 meta에 없어 `{i: str(i)}`로 복원된다(지표 계산엔 무관).
- **Q4 재부착 방식**: `DetectionModel_netspresso(graph_model_path, meta_head_json)` = `nn.Sequential(torch.load(body), Detect_netspresso(nc, nl, anchors, strides, stride))`; `YOLO_netspresso(..., task='detect_retraining', meta_config=yaml)`가 이를 감싸 `val/train`에 연결(`TASK_MAP['detect_retraining']`).
- **Q5 로컬 추론 가능 여부**: **가능**(포크 venv, torch 2.0.1, CPU). 단 §4의 정합성 문제가 있다.

## 4. 발견된 블로커: fx body / 포크 평가기의 기능적 비동등성

| 측정 | 값 | 환경 |
|---|---|---|
| 공식 가중치 `yolov8n.pt`(v8.4.0 asset) COCO128 mAP50-95 | **0.44369** | upstream ultralytics 8.4.173, torch 2.14.1 |
| 같은 가중치, 같은 128장 | **0.13505** | 포크 8.0.108, torch 2.0.1 (fork `YOLO(...).val`) |
| 포크 era 체크포인트(assets v0.0.0, ckpt version 8.0.0.dev0) | upstream 0.44369 / 포크 0.13505 | 체크포인트 세대 문제 **아님** |
| identity fixture(포크 fx body + `Detect_netspresso`) | 0.13506 (포크) — 포크 직접 평가와 동일 | 재부착 경로 자체는 정확 |
| 한 이미지(`000000000009.jpg`) 예측 | upstream 6 boxes(class 45/50/52) vs 포크 1 box(class 6) | |
| fx body(포크 추적) + **upstream decode**(torch 2.14 venv) vs upstream 모델, 같은 실제 이미지 | proxy cosine **0.9944**, relative diff **10.6 %**, max |Δ| 394.5 | fx body 자체가 upstream forward와 다름 |
| upstream 8.4.173 `DetectionModel`을 plain `fx.Tracer`로 추적 | `TraceError: Proxy object cannot be iterated` | 포크가 존재하는 이유; upstream에서 공식 recipe 재현 불가(패치 필요) |
| torch 2.0.1 + upstream 8.4.173 (torch 변수 분리) | **0.44367** → torch는 원인 아님 | 별도 venv, 삭제됨 |

**해석**: 포크의 모듈 코드(8.0.108)로 forward한 결과가 upstream과 다르며, 포크가 추적한 fx body가 그 차이를 그대로 담고 있다. 원인(모듈 정의 드리프트 vs 런타임)은 이 시점에는 **NOT VERIFIED**였다 → **§11.7에서 확정(포크 `C2f.forward`의 Bottleneck 비연쇄, VERIFIED)**. 영향: (1) 포크 평가기의 절대 mAP는 release 근거로 쓸 수 없다, (2) 이 body를 압축하면 "upstream YOLOv8n"이 아닌 "포크 forward 의미의 YOLOv8n"을 압축하는 것이다, (3) 2025년 `NotValidFrameworkException`과는 **다른 종류**의 문제(형식이 아니라 수치 정합성)다.

### 4.1 torch 버전 분리 테스트
별도 venv(upstream ultralytics 8.4.173 + **torch 2.0.1** + numpy 1.26 + opencv 4.10)에서 같은 가중치·같은 128장: **mAP50-95 0.44367**(upstream/torch 2.14의 0.44369와 일치, 1 image 예측 6 boxes). → torch 2.0.1 런타임은 원인이 **아니다**. 남는 변수는 포크 8.0.108의 모듈/forward 코드이며, 그 코드가 v0.0.0(ckpt version 8.0.0.dev0) 체크포인트조차 0.135로 평가하므로 "체크포인트가 너무 새롭다"는 설명도 성립하지 않는다. 정확한 메커니즘(어느 모듈 정의가 다른가)은 이 시점에는 **NOT VERIFIED**였다(→ §11.7에서 확정). 테스트 venv는 확인 후 삭제했다.

### 4.2 해결 방향 (0 Credit)
| 방안 | 내용 | 예상 공수 | Credit |
|---|---|---|---|
| R1 (권장) — **완료, §11** | upstream 8.4.173에서 공식 recipe를 재현: `_predict_once`를 fx-traceable하게 만드는 최소 패치(포크가 한 일) 후 fx body 생성 → upstream 모델과 proxy cosine ≈ 1.0 확인 → upstream 평가기로 mAP 0.444 재현 → torch 2.0.1 venv에서 로드 가능성 확인(서버 호환 신호) | 소~중 (수십 줄) | 0 |
| R2 — **R1 cross-check에서 함께 수행, 원인 확정(§11.7)** | 포크 forward와 upstream forward의 레이어별 diff로 원인 확정(모듈 정의 드리프트 식별) | 중 | 0 |
| R3 | 포크 환경 그대로 E5-1 진행하되, accuracy는 "포크 평가기 기준 상대 비교"로만 해석 | 0 | 25 |

R3는 Credit 효율은 좋지만 "압축 입력이 무엇인지"를 설명하지 못하므로 권장하지 않는다. **R1 완료 시 GO FOR E5-1.** → R1은 완료되었다(§11).

## 5. 검증된 평가 경로 (실제 압축 없이)

```
baseline fx body (identity fixture) + head_meta
  → precheck_fx_bundle                READY (format, GraphModule, head meta, input shape, output schema [1,144,80/40/20])
  → DetectionModel_netspresso          PASS   (decoded [1,84,8400] == expected)
  → DetectionValidator(COCO128)        MEASURED (fixture 0.13506 vs same-evaluator baseline 0.13505, drop −0.001 pp)
  → compare_detection_accuracy         MEASURED; cross-evaluator(upstream 0.44369) → NOT_COMPARABLE (evaluator differs)
  → validate_accuracy (release 1.0 pp) PASS for fixture; candidate 없음 → NOT_APPLICABLE (never FAIL)
```
fixture 결과는 압축 결과가 아니다("identity fixture", `is_output_equivalence_proxy: false`). 평가 조건 일치(dataset/split/imgsz/evaluator/conf/iou)가 깨지면 `NOT_COMPARABLE`로 delta를 만들지 않는다.

## 6. Release dataset 결정 (OPTION A)

| | COCO128 | COCO val2017 | deterministic subset |
|---|---|---|---|
| source / version | `https://ultralytics.com/assets/coco128.zip` (COCO 2017 train subset 128장, ultralytics 배포) | COCO 2017 val (images `val2017.zip` 1 GB + `annotations_trainval2017.zip` 241 MB, cocodataset.org / ultralytics `coco.yaml`) | val2017에서 규칙 추출 |
| license | COCO 주석 CC BY 4.0, 이미지 Flickr 약관(ultralytics 재배포본) | 동일 | 동일 |
| size | 6.7 MB / 128 img / 929 inst | ~1.2 GB / 5,000 img | 선택 크기 |
| reliability | 낮음(분산 큼, 공개값 비교 불가) | 높음(공개 mAP50-95 37.3과 비교) | 중간 |
| runtime (CPU) | ~3 s | ~2–3 min | 비례 |
| reproducibility | 결정적(두 번 측정 동일 0.44369) | 결정적 | 추출 규칙·시드·ID 목록·SHA 기록 필수 |
| CI | 가능 | 부적합 | 가능 |
| 결정 | **Smoke / PR** | **Release accuracy (승인 후 다운로드)** | fallback: `instances_val2017.json` image_id 오름차순 상위 N=500, 목록 SHA 기록 |

dataset binary는 `datasets/`(gitignore)에만 둔다. COCO128로 Release PASS/FAIL을 선언하지 않는다.

## 7. Accuracy threshold 정책

- 값: `release.accuracy.max_drop_percent = 1.0 pp` 유지 — **PROJECT-DEFINED EXAMPLE THRESHOLD**, Nota 공식 아님.
- 적용 조건: (1) baseline·candidate가 **같은 평가기 빌드·같은 데이터·같은 split·imgsz·conf/IoU**일 때만(`NOT_COMPARABLE` 가드), (2) release 데이터(val2017)에서만, (3) 재학습 후 산출물에 대해. 재학습 전 압축 직후는 Gate를 평가하되 FAIL이 예상 결과임을 명시.
- 변동성 근거: COCO128 mAP는 재측정 시 동일(0.44369 = 0.44369) → 평가기 결정성은 확인. 데이터셋 크기 분산(128장)은 크므로 smoke 값으로 판정하지 않음.
- Phase 5-B/5-D 교훈: proxy cosine 0.825(압축 모델)·0.994(fx body)는 **accuracy가 아니다**; proxy와 mAP는 별도 신호로 유지.

## 8. Latency / Memory 재검증

| run | median | p95 | min | peak RSS Δ |
|---|---|---|---|---|
| 20261005T022024Z | 15.53 ms | 29.56 | 10.89 | 107.13 MB |
| 20261005T024950Z | 22.72 ms | 47.85 | 11.20 | 107.07 MB |
같은 머신·같은 정책(격리 프로세스, warm-up 10, 100회)인데 median이 **+46 %** 차이 — 머신 상태(백그라운드 설치·thermal) 영향. 메모리는 안정(Δ 0.06 %). 함의: 로컬 latency Gate(+10 %)는 **반복 실행의 median-of-medians와 측정 시 부하 통제** 없이는 노이즈를 회귀로 오판한다 → 5-E 실행 시 baseline·candidate를 같은 세션에서 교대 반복(≥3회)으로 측정하는 프로토콜을 요구한다. GPU memory: N/A.

## 9. Normalized checksum

`normalized_onnx_checksum()`: raw SHA-256 보존 + `metadata_props` 전체 제거·`doc_string` 비움 후 SHA-256(policy `onnx-metadata-v1`, version 1, 제거 키 기록). baseline ONNX: raw `7bf62b34…`, normalized `6f180ff2…`; 제거된 키 15개(`date` 포함; `names`·`imgsz`도 metadata라 함께 제거됨 — 의미 있는 메타데이터지만 그래프 동일성과는 무관, raw SHA가 보존하므로 정보 손실 없음). 레지스트리에는 두 값을 함께 기록; 이전 export candidate들은 raw만 다른 것으로 확인되어 `retired`.

## 10. 보안·Credit

API 키 미사용(스크립트는 `os.environ`/`getenv` 미접근, 테스트로 강제), SDK import 0(테스트 + 스크립트 assert), ledger SHA 불변, `reports/real_runs` 불변, 모델·데이터셋 바이너리 미추적(`outputs/`, `datasets/`, `.venv-*/`).

## 11. R1 — upstream 기반 traceable fx export와 기능적 정합성 (0 Credit, 결과: **PASS → GO FOR E5-1**)

> 레코드: `reports/yolov8_baseline/20261005T034329Z_r1/r1_result.json`(upstream venv), `…T034230Z_r1_torch201_load/`(torch 2.0.1 only), `…T034416Z_r1_fork_crosscheck/`(포크 venv), `…T034240Z_offline_eval/`(포크 평가기 + R1 body). 스크립트: `scripts/yolov8_r1_traceable_export.py`, `scripts/yolov8_r1_fork_crosscheck.py`. 순수 Python 계약: `framework/evaluation/fx_traceability.py`. 테스트: `tests/unit/test_yolov8_r1.py`(TC-Y8-R1-001…013).

### 11.1 목적과 블로커
§4의 블로커(포크 fx body ≠ upstream YOLOv8n; 포크 평가 0.13505 vs upstream 0.44369)를 해소하기 위해, **수정하지 않은 upstream ultralytics 8.4.173** 코드에서 fx body를 만들고 upstream 모델과의 기능적 동등성을 (1) pre-NMS 텐서, (2) post-NMS 검출, (3) COCO128 mAP 세 층에서 검증했다. 2025년 ONNX 실패와는 다른 범주의 문제이므로 별도로 다룬다(`historical_yolov8_regression.md` §8).

### 11.2 Traceability 분석 (실제 그래프/코드 근거)
| 확인 | 결과 |
|---|---|
| upstream `DetectionModel`을 plain `fx.Tracer().trace()` | eval/train 모두 **`TraceError: Proxy object cannot be iterated`** — 원인은 `C2f.forward`의 `list(self.cv1(x).chunk(2, 1))`(Proxy 반복) |
| upstream `Detect` train 모드 출력 | **dict** `{boxes [1,64,8400], scores [1,80,8400], feats}` — 포크(8.0.108)의 "`nl`개 map `[N,144,H,W]`" 계약과 다름 → 공식 head meta 계약을 유지하려면 wrapper가 필요 |
| 그래프 안(in-graph) | layer 0–21(backbone+neck) + `Detect.cv2[i]`/`cv3[i]` → `cat` → `[N,144,H/s,W/s]`×3. 230 nodes: call_module 182(Conv2d 63, BatchNorm2d 57, SiLU 57, MaxPool2d 3, Upsample 2), call_function 38, call_method 8 — **leaf는 모두 `torch.nn`**, decode/NMS 마커 없음 |
| 그래프 밖(out-of-graph) | DFL, anchors/strides(`make_anchors`), `dist2bbox`·stride 곱, class sigmoid, **NMS**, 전처리(letterbox, /255, BGR→RGB). 모두 `netspresso_head_meta.json`(nc/nl/anchors/stride/strides/inplace)으로 재구성 |

### 11.3 Traceability patch (최소 변경, 제품 코드 미수정)
patch id **`r1-upstream-body-wrapper-v1`** — 두 단계만:
1. **wrapper 모듈** `TraceableYOLOv8Body`: `BaseModel._predict_once`의 save/from 인덱스 재생(layer 0–21) + Detect의 `cv2/cv3` 분기 적용·concat. 텐서 값에 의존하는 Python 제어 흐름 없음.
2. **C2f 인스턴스에 upstream 자체의 `forward_split` 바인딩**(`m.forward = m.forward_split`; ultralytics exporter가 일부 포맷에서 하는 것과 같은 조치). 의미는 `forward`와 동일(Bottleneck을 **연쇄**로 적용). 적용 인스턴스 8개(layer 2·4·6·8·12·15·18·21).

ultralytics 소스는 한 줄도 수정하지 않았다. 포트폴리오 관점에서 이것은 "제품 코드 수정"이 아니라 **QA가 정의한 traceability boundary**다. body/head 분할은 공식 포크 `export_netspresso()`와 동일하므로 산출물은 포크의 `DetectionModel_netspresso` 재부착 경로에서도 그대로 소비된다(11.6).

### 11.4 Export 산출물
| 항목 | 값 |
|---|---|
| 경로 | `outputs/models/yolov8n_fx_r1/model_fx.pt` (+ `netspresso_head_meta.json`, `export_manifest.json`; git 미추적) |
| SHA-256 / 크기 | `d8e761dae29301ef51df9e7679c729338a522d397e271a900d453d171e1c022b` / 12,846,691 B (fp32 GraphModule, 3,157,184 params) |
| source | `yolov8n.pt` sha `f59b3d83…`, 체크포인트 version 8.0.0.dev0(date 2022-12-30) — assets v8.4.0 URL의 파일이지만 내용은 2022년 체크포인트 |
| source code | ultralytics 8.4.173 (upstream, unmodified), torch 2.14.1+cpu, Python 3.11.9 |
| 입력/출력 | `[1,3,640,640]` → `[1,144,80,80] / [1,144,40,40] / [1,144,20,20]`; 재부착 후 `[1,84,8400]` |
| 재현성 | 같은 소스·패치·환경으로 2회 export → **raw SHA 동일(bit-identical)**, head meta 동일. 정규화 체크섬 불필요(raw가 이미 일치) |
| 구조 검증 | `validate_pt` PASS(`graph_module`), `precheck_fx_bundle` **READY**(format·GraphModule·head meta·input shape·output schema). 신뢰 목록(`configs/local_eval.yaml`)에 SHA 등록 |

### 11.5 기능적 동등성 (upstream venv, 동일 전처리 입력, COCO128 처음 8장 + 대표 이미지)
| 비교 (upstream `DetectionModel` eval, unfused ↔ R1 body + `R1DecodeHead`) | cosine | max |Δ| | relative L2 |
|---|---|---|---|
| raw boxes `[1,64,8400]` / raw scores `[1,80,8400]` (pre-decode) | 1.0 | **0.0 (bitwise)** | 0.0 |
| decoded `[1,84,8400]` (pre-NMS; boxes·scores 각각도) | 1.0 | **0.0 (bitwise)** | 0.0 |
| 참고 noise floor: upstream unfused ↔ upstream **fused**(Conv+BN 접기) | 0.99999999999986 | 6.8e-3 | 4.4e-7 |

판정은 `equivalence_verdict`에 **명시적으로 넘긴 임계값**(min cosine 0.999999, rel ≤ 1e-4; PROJECT-DEFINED EXAMPLE, noise floor 측정 후 결정)으로 했고 결과는 EQUIVALENT — 실제로는 임계값이 필요 없는 bitwise 일치다. 포크 body의 옛 수치(cosine 0.9944 / rel 10.6 %)는 같은 함수에서 NOT_EQUIVALENT로 판정됨을 테스트로 고정했다.

**One-image diagnostic** (`000000000009.jpg`, upstream NMS 동일 적용):
| 설정 | upstream | R1 | 매칭 | 좌표 max |Δ| | conf max |Δ| |
|---|---|---|---|---|---|
| predictor 기본 conf 0.25 / IoU 0.7 | **6 boxes**(class 45×4, 50, 52) | **6 boxes** | 6/6 | 0.0 | 0.0 |
| validator conf 0.001 / IoU 0.7 | 194 boxes | 194 boxes | 194/194 | 0.0 | 0.0 |
→ §4의 "upstream 6 boxes vs 포크 FX 1 box" 불일치는 해소되었다.

### 11.6 Detection evaluation (COCO128, 같은 평가기·data·split·imgsz·conf/IoU·`rect=True`)
| 모델 | 평가기 | mAP50-95 | mAP50 | mAP75 | P | R |
|---|---|---|---|---|---|---|
| upstream baseline `YOLO(yolov8n.pt).val` (재측정, 결정적) | ultralytics 8.4.173 | **0.44369** | 0.60111 | 0.47223 | 0.6286 | 0.52902 |
| **R1 fx body + R1DecodeHead** (`DetectionValidator(model=…)`) | ultralytics 8.4.173 | **0.44369** | 0.60111 | 0.47223 | 0.6286 | 0.52902 |
| → drop_pp **0.0**, relative_delta **0.0 %** (`compare_detection_accuracy` MEASURED, proxy 아님) | | | | | | |
| R1 fx body, **공식 포크 재부착**(`YOLO_netspresso` → `DetectionModel_netspresso`) | ultralytics_nota 8.0.108 | **0.44482** | 0.60256 | — | 0.62843 | 0.52902 |
| 같은 가중치, 포크 모델 코드(`YOLO(yolov8n.pt).val`) | ultralytics_nota 8.0.108 | 0.13505 | 0.19866 | — | 0.56956 | 0.15895 |

세 번째 행이 결정적이다: **포크의 평가기·데이터로더·재부착 경로는 정상**(0.4448 ≈ 0.4437; 0.11 pp는 평가기 빌드 차이)이고, 0.135는 **포크의 모델 코드**에서만 나온다.

### 11.7 Layer-level diff와 원인 확정 (포크 venv, `yolov8_r1_fork_crosscheck.py`)
upstream 레이어별 출력을 npz로 덤프하고(같은 입력), 포크가 로드한 `yolov8n.pt`를 레이어별로 비교했다(파라미터 서명 184개 전부 동일).
| 레이어 | 포크 vs upstream relative L2 | 비고 |
|---|---|---|
| layer 0–3 (Conv, Conv, C2f n=1, Conv) | ≤ 1.7e-7 | torch 2.0.1 vs 2.14 부동소수 노이즈 수준 |
| **layer 4 (C2f, n=2)** | **0.72** (cosine 0.858) | **첫 분기점** |
| layer 5–21 | 0.39 – 1.08 | 전파 |
| Detect cv2/cv3 maps | 0.26 (cosine 0.967) | |
| decoded (eval) | 0.108 (cosine 0.9942) | §4의 proxy 0.9944와 일치 |

**메커니즘 (VERIFIED)**: 포크 `C2f.forward`는
```python
out = self.cv1(x).split((self.c, self.c), 1)
y = [m(out[-1]) for m in self.m]           # 모든 Bottleneck이 같은 out[1]을 입력으로 받음
```
upstream은
```python
y = list(self.cv1(x).chunk(2, 1))
y.extend(m(y[-1]) for m in self.m)         # Bottleneck 연쇄: m1(m0(y1))
```
Bottleneck이 1개면 동일하고, **n ≥ 2(yolov8n layer 4·6)에서 의미가 달라진다**. 포크의 변경은 fx 추적 가능성(Proxy 반복 회피)을 위한 것으로 보이지만 연쇄 의미를 깨뜨렸다. 검증: 포크 `C2f.forward`를 **메모리에서만** 연쇄 의미로 바꿔 재실행 → 모든 레이어 차이 소멸, decoded relative L2 **1.5e-7**(= R1 body의 포크 재부착 수치). 따라서 분류 **version difference / 포크 모듈 forward 구현 차이, confidence HIGH, verified**. 영향 범위는 n ≥ 2인 C2f를 가진 모든 YOLOv8 변형(n부터 x까지 전부)이며, 2025년 YOLOv8-l 사례와의 연결은 **추정하지 않는다**(2025는 ONNX 형식 단계에서 실패했으므로 이 코드까지 도달하지 않았다).

옛 포크 산출물 `yolov8n_fx/model_fx.pt`는 upstream body map과 min cosine 0.966 / max rel 0.265로 **동등하지 않음**을 같은 레코드에 남겼다(압축 입력으로 사용 금지).

### 11.8 torch 2.0.1 호환 신호
`.venv-netspresso`(torch 2.0.1, **ultralytics 없음**)에서 `torch.load` → `GraphModule` 로드·forward **PASS**, 출력 shape 동일, torch 2.14 출력과 max |Δ| 6.9e-5(버전 간 커널 노이즈, bitwise 아님). pickle이 참조하는 클래스는 `torch.nn.*`/`torch.fx.*`만이므로 ultralytics 설치가 필요 없다(fork 산출물과 동일한 성질). **의미: target-side torch 2.0.1 환경에서 로드 가능. NetsPresso 서버 수락을 증명하지 않는다.**

### 11.9 실행 중 발견·수정한 스크립트 결함 (기록)
첫 실행은 FAIL이었다(decoded cosine 0.9999, mAP 0.45131 vs 0.44369). 원인은 산출물이 아니라 검증 스크립트 두 곳: (a) 분석 단계에서 **참조 모델**을 train 모드로 forward해 BatchNorm running stats가 오염됨 → throwaway deepcopy로 수정; (b) 직접 호출한 `DetectionValidator`가 `rect=False`로 동작한 반면 `YOLO.val()`은 `rect=True`를 강제 → 동일 조건으로 수정. 포크 cross-check에서도 Detect head를 train 모드로 실행해 같은 오염이 생겨(probe의 decoded rel 1.6e-3) eval 모드 분기 호출로 수정했다. 수정 후 재실행 결과만 레코드로 남겼다. 교훈: **참조 모델은 절대 train 모드로 실행하지 않는다**(BN).

### 11.10 R1 성공 조건 판정
| # | 조건 | 결과 |
|---|---|---|
| 1 | upstream baseline 재현 | ✓ 0.44369 = 0.44369 |
| 2 | traceable fx export 생성 | ✓ leaf-clean GraphModule, patch 2단계 |
| 3 | head re-attach 성공 | ✓ upstream helper + 공식 포크 경로 모두 |
| 4 | input/output schema PASS | ✓ precheck READY |
| 5 | pre-NMS 동등성 | ✓ bitwise |
| 6 | one-image 검출 일치 | ✓ 6=6, 194=194, Δ=0 |
| 7 | COCO128 mAP 동등 | ✓ drop 0.0 pp (포크 평가기로도 0.4448) |
| 8 | torch 2.0.1 load | ✓ PASS |
| 9 | artifact validation | ✓ PASS |
| 10 | reproducibility | ✓ 2회 export bit-identical |
| 11–12 | Credit 0 / API 0 | ✓ ledger SHA 불변, `env -u NETSPRESSO_API_KEY` |
| 13 | 기존 테스트 유지 | ✓ 145 → 164 (R1 19 추가) |

**결론: R1 PASS → 결정 GO FOR E5-1.** 압축 입력은 `yolov8n_fx_r1/model_fx.pt`(sha `d8e761da…`)이며 옛 포크 산출물은 사용하지 않는다. E5-1(단일 `automatic_compression`, ratio 0.5, `[1,3,640,640]`, 클라이언트 사전 체크 25)은 **별도 승인 후** 1회만 실행한다.

> **후속**: E5-1은 별도 승인 후 2026-10-05 04:00 UTC에 1회 실행되었다(계정 475 → 450). 결과·검증·HOLD 권고는 [`phase5e_e5_1_experiment.md`](phase5e_e5_1_experiment.md).

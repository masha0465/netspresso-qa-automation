# YOLOv8n Baseline Strategy (Phase 5-E Preparation, 0 Credit)

> 모든 수치는 2026-10-05 로컬(개발 머신) 측정값이다. NetsPresso API는 호출하지 않았고 Credit은 0 사용했다.
> 원본 레코드: `reports/yolov8_baseline/20261005T022024Z/baseline.json`, `reports/yolov8_baseline/20261005T022123Z_fx_precheck/fx_export_precheck.json`.
> 임계값은 모두 **PROJECT-DEFINED EXAMPLE THRESHOLD**이며 Nota 공식 기준이 아니다.

---

## 1. 모델

| 항목 | 값 | 출처/검증 |
|---|---|---|
| model_name / task | YOLOv8n / object detection (80 classes) | ultralytics 8.4.173 `YOLO(...).task` |
| weights_source | `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt` (공식 ultralytics/assets release v8.4.0) | 다운로드 URL·크기 일치(6,549,796 bytes) |
| weights_sha256 | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` | `compute_sha256`; `configs/local_eval.yaml` 신뢰 목록에 등록(출처 명시) |
| license | AGPL-3.0 (ultralytics) | 공개 평가용 로컬 사용만; 바이너리는 git 미추적(`outputs/` ignore) |
| structural validation (.pt) | **PASS** — `checkpoint` (dict 안의 `model` 모듈), 전체 unpickle은 신뢰 목록 기반 | `validate_pt` (5-E에서 checkpoint dict 인식 추가) |
| implementation-reported | 3,157,200 params (unfused), **8.855 GFLOPs @640** | `ultralytics.utils.torch_utils.get_num_params/get_flops` — NetsPresso SDK 값이 아님 |

## 2. 두 가지 export 경로 (둘 다 0 Credit, 로컬 검증 완료)

| 경로 | 도구 | 산출물 | 용도 |
|---|---|---|---|
| **A. ONNX (평가용 baseline)** | ultralytics 8.4.173, `export(format="onnx", imgsz=640, opset=13, simplify=False, dynamic=False)` | `outputs/models/yolov8n_640_opset13.onnx` 12,824,019 bytes, sha256 `185840476208ca50…`, 입력 `images [1,3,640,640]`, 출력 `output0 [1,84,8400]` float32(decode 포함: 4 box + 80 cls) | ORT latency/memory, 출력 비교, 구조 검증 |
| **B. NetsPresso fx (업로드용)** | `ultralytics_nota` 포크 8.0.108 (archived) + torch 2.0.1, `YOLO(...).export_netspresso(save_path)` | `outputs/models/yolov8n_fx/model_fx.pt` 12,846,561 bytes(fp32 GraphModule, 3,157,184 params) + `netspresso_head_meta.json` {nc 80, nl 3, stride [8,16,32], anchors, strides, inplace} | compressor_v2 입력 (공식 ModelZoo 경로) |

경로 B 검증(TC-Y8-030 → **READY**): 포크가 v8.4.0 체크포인트를 로드(버전 간 unpickle 성공), `model_fx.pt`는 `torch.fx.GraphModule`(`validate_pt` PASS), train 모드 forward 출력 **[1,144,80,80] / [1,144,40,40] / [1,144,20,20]** (144 = 80 cls + 4×16 DFL). 이는 Phase 5-B에서 성공한 공식 샘플 `graphmodule.pt`와 동일한 계약(fx GraphModule `.pt`, framework pytorch)이다. **서버 수락은 실 실행으로만 증명**되지만, 2025년 실패 유형(ONNX/비-fx 입력)은 업로드 전에 제거되었다.

포크 설치 메모(재현용): archived 저장소라 최신 setuptools에서 `pkg_resources` 부재로 빌드 실패 → `setuptools<80` + `--no-build-isolation`으로 설치; 의존성 해석이 numpy 2.x를 끌어와 torch 2.0.1과 충돌 → `numpy<2`, `opencv-python<4.11` 고정. 별도 venv `.venv-yolo-nota`(gitignore).

## 3. 데이터셋

| 후보 | 라이선스 | 크기 | 평가 신뢰도 | 런타임(CPU) | 재현성 | CI 적합 | 포트폴리오 증거 | 결정 |
|---|---|---|---|---|---|---|---|---|
| **COCO128** (`coco128.yaml`, ultralytics assets) | COCO 2017 이미지/주석 파생 (COCO: CC BY 4.0 주석, 이미지는 Flickr 약관) | 6.7 MB zip, 128 images / 929 instances (train2017 subset을 val로 사용) | **낮음** — 128장, 분산 큼, 공개 baseline(37.3)과 비교 불가 | 약 3 s | 높음(고정 파일, 해시 가능) | 가능 | smoke 수준 | **Smoke / 배관 검증용 채택 (완료)** |
| **COCO val2017** (`coco.yaml`) | COCO 2017 | images 1 GB + annotations 241 MB, 5,000 images | **높음** — 공개 mAP50-95 37.3(ultralytics, 640)과 직접 비교 | 약 5,000 × 18 ms ≈ 1.5–2 min 추론 + 전처리 (CPU) | 높음 | 부적합(용량) | release 수준 | **Release accuracy용 — 미다운로드, 승인 후** |
| 다른 공개 subset (예: COCO val 500장 임의 추출) | COCO 2017 | ~100 MB | 중간 — 추출 규칙·시드를 기록해야 재현 | 수십 초 | 추출 스크립트 필요 | 가능 | 중간 | 보조 옵션(필요 시) |

원칙: **COCO128 결과로 Release accuracy PASS/FAIL을 선언하지 않는다.** 128장 mAP는 배관이 동작한다는 증거이지 모델 품질 판정 근거가 아니다. Release 판정은 val2017(또는 기록된 규칙의 subset) 측정 후에만 한다.

## 4. 측정된 Baseline (COCO128, 개발 머신)

| 항목 | 값 | 비고 |
|---|---|---|
| input_shape | [1, 3, 640, 640] | |
| preprocessing | ultralytics val 기본(letterbox 640, RGB, /255); conf 0.001, IoU 0.7(val 기본) | ORT latency 측정은 균일 난수 입력 |
| **mAP50-95** | **0.4437** | 128장 — 공개 37.3과 비교 금지(§3) |
| mAP50 / mAP75 | 0.6011 / 0.4722 | |
| precision / recall | 0.6286 / 0.5290 | |
| ultralytics 속도(ms/img) | pre 0.40 / infer 17.88 / post 0.85 | torch 경로, 참고용 |
| **ORT latency (ONNX, 격리 프로세스, warm-up 10 / 100회)** | median **15.53 ms**, p95 29.56, min 10.89, max 41.80, mean 16.61 | `local_onnxruntime_cpu`, CPUExecutionProvider, 스레드 ORT 기본 |
| **peak RSS 증가분** | **107.13 MB** (before 20.19 → after load 69.69 → peak 127.32) | psutil, CPU만. **GPU memory: N/A (GPU 없음)** |
| params | torch 3,157,200 / ONNX initializer 3,151,904 (BN fold, Δ 0.17 % → PASS) | |
| FLOPs | impl 8.855 G vs ONNX MAC 추정 4.372 G → 2×MACs 8.744 G, Δ 1.25 % → **PASS** | Conv/Gemm/MatMul만 커버; 미커버 op: Sigmoid 58, Mul 60, Add 9, Concat 17, Softmax 1 등(원소별 연산, FLOPs 비중 작음) |
| model_size | pt 6,549,796 B (fp16 ckpt) / ONNX 12,824,019 B / fx 12,846,561 B (fp32) | 크기 비교는 **같은 포맷·정밀도** 간에만 |
| environment | Windows 11 (10.0.26200), Intel64 Family 6 Model 198 (Core Ultra 7 265K), 20 cores, Python 3.11.9, ultralytics 8.4.173, torch 2.14.1+cpu, onnxruntime 1.30.0, onnx 1.23.1, numpy 2.4.6 | fingerprint에 시크릿 없음 |
| seed | 0 | |

측정하지 못한 값: 타깃 디바이스 latency/memory (**N/A**), GPU memory (**N/A**), COCO val2017 mAP (**N/A — 미다운로드**).

## 5. Reproducibility 기록 항목 (레지스트리 연결)

`reports/baselines/registry.json`에 **candidate** 등록: `yolov8n_baseline_onnx_export:185840476208` (source `yolov8n.pt sha256=f59b…`, 입력 shape, exporter 버전·opset, 환경 fingerprint). 승격 정책은 변경하지 않았다(Level 3/4 증거 + 무결성 + 설정 일치).

**관찰 — ONNX export는 bit-reproducible이 아니다**: 동일 조건으로 두 번 export하면 SHA가 다르고, 차이는 `metadata_props`의 `date` 한 항목뿐이다. metadata를 제거하면 그래프+가중치 SHA가 **동일**(`f8fd0760185d…`). 따라서 Level 3(비트 동일)은 *정규화 체크섬*(metadata 제거 후 SHA)으로 정의해야 의미가 있고, 그렇지 않으면 항상 Level 4(출력 동등성)로 떨어진다. → Phase 5-E 구현 제안: `artifact.metadata["normalized_sha256"]` 추가(0 Credit). 첫 export의 stale candidate(`4490e9…`)는 이 사유로 `retired` 처리했다.

## 5-A. 재실행과 평가기 드리프트 (Conditional GO 해소 작업에서 추가)

| 항목 | 관찰 |
|---|---|
| baseline 재실행(upstream 8.4.173) | mAP50-95 **0.44369 = 0.44369**(결정적); latency median 15.53 → **22.72 ms**(+46 %, 머신 부하 차이), peak RSS 107.13 → 107.07 MB |
| 같은 가중치를 포크 8.0.108로 평가 | mAP50-95 **0.13505** — 평가기만 바꿔도 값이 다름 → `compare_detection_accuracy`는 evaluator 불일치를 `NOT_COMPARABLE`로 처리(delta 생성 금지) |
| torch 2.0.1 + upstream 8.4.173 | 0.44367 → 차이는 포크 코드에서 발생 — **R1에서 확정**: 포크 `C2f.forward`가 Bottleneck을 연쇄하지 않음(n ≥ 2인 layer 4·6), 해소 문서 §11.7 |
| ONNX checksum | raw `7bf62b34…` / normalized `6f180ff2…`(policy onnx-metadata-v1). raw는 export마다 바뀜(`date`), normalized는 동일 |
| release 평가기 결정 | **upstream ultralytics 8.4.173(고정 버전)** — baseline·candidate 모두 같은 빌드로 평가. fx export는 **R1 경로**(upstream + wrapper, `yolov8n_fx_r1/model_fx.pt`)를 사용하고 포크 산출물은 압축 입력에서 제외. 포크의 재부착·평가기 경로 자체는 정상(R1 body로 0.4448) |
| R1 fx body(upstream 평가기) | mAP50-95 **0.44369 = baseline**, pre-NMS bitwise 동일, 2회 export bit-identical, torch 2.0.1 로드 PASS |

## 6. 압축 모델의 baseline 비교 조건 (5-E 실행 시 지켜야 할 것)

| 조건 | 설계 |
|---|---|
| 같은 데이터·split·전처리 | 동일 `data yaml`, imgsz 640, ultralytics val 기본 conf/IoU, 동일 class mapping(head_meta nc 80) |
| 같은 평가 코드 | 압축 body(`model_fx.pt` 압축본)는 decode가 없으므로 **포크의 `DetectionModel_netspresso(body, head_meta)`로 head를 다시 붙인 뒤** 같은 `val`을 돌려야 비교가 성립한다. 이 단계는 로컬·0 Credit이며 5-E에서 helper로 구현 |
| 재학습 전/후 분리 | 공식 워크플로우는 압축 후 fine-tuning을 전제한다. **압축 직후 mAP는 거의 0에 가까울 것이 예상**(Phase 5-D proxy 0.825와 같은 맥락) → Gate는 FAIL이어야 하며 그것이 정답이다. fine-tuning(coco128 수 epoch, CPU 가능하나 느림)은 별도 결정 |
| 같은 ONNX 조건 | 압축 body → head 재부착 → ONNX export(동일 opset/imgsz) → ORT 측정. 그래야 baseline ONNX와 latency/memory 비교 가능 |
| 출력 proxy | baseline ONNX vs 압축 ONNX의 `output0 [1,84,8400]` cosine — 보조 신호, accuracy 아님 |

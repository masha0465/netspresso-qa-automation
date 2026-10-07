# Historical YOLOv8 Regression Case (2025) and the Current YOLOv8n Case (2026)

> 출처 원칙: 2025년 내용은 공개 저장소 **masha0465/netspresso_test** (main, 2025-09-14, 21 commits)의 실제 파일만 근거로 삼는다. 저장소에 없는 것은 "N/A / NOT VERIFIED"로 표기하고, 원인은 추측하지 않는다. 2026년 내용은 현 저장소 commit `c8a6b04` 기준이다.
> 이 문서는 NetsPresso API를 호출하지 않고 작성되었다.

---

## 1. 2025 프로젝트 요약 (실제 저장소 기준)

| 항목 | 저장소에 기록된 사실 |
|---|---|
| 저장소 | `masha0465/netspresso_test`, 설명 "NetsPresso usage test project", 2025-09-14 하루 동안 21 commits |
| 목적 (README) | "10년차 QA 엔지니어로서 Nota AI의 NetsPresso에 관심이 생겨서 직접 사용… YOLOv8 모델 압축이라는 실제 사용 케이스로 제품을 테스트… QA 관점에서 피드백 정리". 테스트 방식 "실제 사용자처럼 문서 보고 따라하기" |
| Nota 지원과의 관계 | README/문서에 명시된 문장은 없음 (**NOT VERIFIED** — 본 프로젝트 사용자의 구술에 따르면 지원 준비용) |
| 기술 | Python, `netspresso>=1.13.0`, torch, onnx, onnx2torch, pytest, pytest-html, GitHub Actions |
| 자동화 | pytest 2개 파일 활성(`test_basic_functionality.py`, `test_yolo_compatibility.py`), 2개 파일 `.disabled`; GitHub Actions `test.yml`(547줄, cron 매일 09:00, pytest + 실 클라이언트 실행 + HTML 리포트) |
| 결과 저장 | `results/`는 `.gitignore` → **실행 결과 JSON·압축 산출물은 저장소에 없음**. 증거는 README·`docs/qa_findings.md`·`docs/improvement_suggestions.md`·`docs/api_analysis.md`·이슈 리스트 PDF(910 KB) |

## 2. 실제 테스트 조건과 모델

| 모델 | 저장소 기록 | 형식 / 경로 | 결과 |
|---|---|---|---|
| 간단한 CNN (`simple_test_model.pt`) | 약 **84 KB**. `SimpleCNN`(Conv2d→Conv2d→AdaptiveAvgPool2d→Linear, 10 classes), `torch.fx.symbolic_trace` 후 `torch.save` | fx GraphModule `.pt`, input `[1,3,224,224]`, `automatic_compression(ratio 0.5)` | **압축 성공** |
| YOLO 호환 모델 (`yolo_like_model.pt`) | 약 **6.4 MB**. `YOLOCompatibleModel`(Conv/BN/ReLU/MaxPool 4블록 + 1×1 head 85ch, 80 classes) — **직접 작성한 YOLO 유사 구조**, YOLOv8 아님 | fx GraphModule `.pt` | **압축 성공** |
| YOLOv8 ONNX (`yolov8l.onnx`) | **167 MB**. 파일명 기준 **YOLOv8-l** (n이 아님). 출처·export 방법·opset·input shape **NOT VERIFIED**(저장소에 기록 없음) | **ONNX를 그대로** `automatic_compression`에 투입 (코드상 `framework` 인자 미지정 → SDK 기본값 PYTORCH) | **압축 실패** — 업로드 성공, 압축 단계에서 `NotValidFrameworkException` |

세 크기 값(84 KB / 6.4 MB / 167 MB)은 `README.md`와 `models/README.md`에 그대로 존재한다. 정확한 바이트 수·SHA는 기록되지 않았다.

## 3. 실패 사례 상세

| 질문 | 저장소 근거 | 판정 |
|---|---|---|
| 어떤 모델 | `yolov8l.onnx` (167 MB) | 확인 |
| 입력 형식 | ONNX 파일. `test_yolo_compatibility.py`의 `xfail` 테스트가 `client.test_yolo_compression("yolov8l.onnx", …)` 호출 (단, 이 메서드는 `netspresso_client.py`에 정의되어 있지 않음 — 미완성 코드) | 확인 (ONNX), 세부 NOT VERIFIED |
| operation | `compressor_v2().automatic_compression(input_model_path, output_dir, input_shapes=[{batch 1, channel 3, dimension [224,224]}], compression_ratio=0.5)` — `framework` 미지정 | 확인 |
| SDK 버전 | `requirements.txt`: `netspresso>=1.13.0` — 정확한 설치 버전 **NOT VERIFIED** (lock 없음) | 범위만 확인 |
| 실제 예외 | `NotValidFrameworkException` (README, qa_findings, 리포트 스크립트의 패턴 매칭 문자열). 전체 메시지/traceback은 저장소에 없음 | 이름만 확인 |
| 재현율 | qa_findings: "재현율 100%" | 기록 확인, 재현 로그는 없음 |
| 당시 원인 판단 | "웹 인터페이스는 GraphModule 또는 ONNX 지원이지만, Python SDK는 실제로 torch.fx.GraphModule만 지원. 이 차이가 문서에 명시되지 않음" | **당시 추정**(공식 확인 없음) |
| 당시 제안 | 사전 호환성 검증 API, 구체적 에러 메시지, 단계별 변환 가이드, 인터페이스별 차이 문서화 | 확인 |

### 2026년 시점의 해석 (검증된 사실과 미검증의 분리)

- **검증된 사실 (현 SDK 1.17.0 소스, Phase 1)**: compressor의 `FRAMEWORK_EXTENSION_MAP`은 `onnx → '.pt'`로 매핑되어 있고, 공식 compressor 예제는 모두 fx GraphModule `.pt`를 입력한다. `NotValidFrameworkException`은 1.17.0 예외 목록에 **없고** `NotSupportedFrameworkException`이 있다(이름 변경 여부·시점은 NOT VERIFIED).
- **검증된 사실 (Nota 공식 ModelZoo-YOLOv8)**: 문서화된 YOLOv8 경로는 `ultralytics_nota` 포크의 `export_netspresso()`로 **fx GraphModule body + head meta JSON**을 만들어 업로드하는 것이며, ONNX 직접 업로드는 compressor 경로에 없다.
- **따라서 2025년 실패의 원인은** 저장소 근거만으로는 `unsupported format(ONNX for compressor) / export path(fx export 미사용)` 쪽이 **일관**되지만, 당시 SDK 버전·서버 응답·전체 메시지가 없으므로 공식 판정은 **UNKNOWN / NOT VERIFIED**로 남긴다. "모델 크기(167 MB)"가 원인이라는 증거는 **없다**. "YOLOv8 구조 자체가 미지원"이라는 증거도 **없다**(공식 ModelZoo가 존재하므로 오히려 반대 방향).

### 2025년 저장소에서 발견된 보안 이슈 (2026-10-05 확인)

`src/netspresso_client.py`가 `os.getenv('NETSPRESSO_API_KEY', '<실제 키 기본값>')` 형태로 **API 키를 공개 소스에 하드코딩**하고 있다(35자, `np-` 접두). 현 프로젝트의 키와는 다른 값이며 현 저장소에는 존재하지 않는다. 해당 키가 아직 유효하다면 콘솔에서 폐기해야 한다(git 이력에서 지우는 것으로는 해결되지 않음). 또한 CI가 매일 cron으로 실 클라이언트를 실행하도록 되어 있어, Credit 소모가 자동화되어 있었다. 이 두 가지가 현 프레임워크의 "키는 환경변수만, CI는 실 API 금지, 실 호출은 `--confirm-credit-use` 1회" 설계의 직접적인 배경이다.

## 4. 2025년 QA 활동의 실제 가치 (근거 있는 것만)

| QA 활동 | 저장소 근거 | 뒷받침 여부 |
|---|---|---|
| 문서 vs 실제 동작 비교 | qa_findings 이슈 #1·#2, improvement_suggestions | ✓ |
| SDK 사용성 테스트 | api_analysis.md(파라미터 분석), 에러 메시지 품질 테스트(`.disabled`) | ✓(일부 비활성) |
| 에러 처리 테스트 | `test_error_handling.py.disabled` (잘못된 경로·키·메시지 품질) | 작성됐으나 **비활성화** |
| 모델 호환성 테스트 | fx 호환 모델 생성·검증(`verify_fx_model`), xfail YOLOv8 | ✓ |
| 경계/대용량 테스트 | 84 KB → 6.4 MB → 167 MB 단계적 시도 | ✓(의도), 측정 데이터 없음 |
| 실패 분석 | 원인 추정(GraphModule-only) + 재현 절차 | ✓(추정 수준) |
| 개선 제안 | 4가지 제안 + 이슈 리스트 PDF | ✓ |
| 결과 문서화 | README 결과표, PDF | ✓; 기계 판독 결과 없음 |
| 사용자 관점 QA | README 전체 서술 | ✓ |

2025 narrative "실제 사용 → 모델 테스트 → YOLOv8 테스트 → 문서/동작 비교 → 실패·Gap 발견 → QA 분석 → 개선 제안"은 **모두 저장소 문서로 뒷받침**된다. 뒷받침되지 않는 부분: "테스트 자동화"(실 API를 pytest에서 직접 호출하는 구조이고 절반이 비활성), "정량 결과"(저장소에 없음), "원인 확정"(추정).

## 5. Historical Case vs Current Case — 같은 모델이 아니다

| 항목 | Historical (2025) | Current (2026, Phase 5-E 후보) |
|---|---|---|
| model | yolov8**l** (파일명 기준) | yolov8**n** |
| model size | 167 MB (ONNX) | 6,549,796 bytes (`yolov8n.pt`, 공식 ultralytics/assets v8.4.0) |
| model format | ONNX | ultralytics checkpoint `.pt` → (A) ONNX export로 로컬 baseline, (B) `export_netspresso()`로 fx GraphModule body + head meta |
| export path | NOT VERIFIED | (A) ultralytics 8.4.173 `export(format="onnx", opset 13)`; (B) 공식 포크 `ultralytics_nota` 8.0.108 `export_netspresso()` |
| framework (SDK 인자) | 미지정(기본 PYTORCH)에 ONNX 파일 | PYTORCH + fx GraphModule `.pt` (공식 예제와 동일) |
| SDK version | `>=1.13.0` (정확 버전 NOT VERIFIED) | 1.17.0 (설치·시그니처 재검증) |
| input shape | `[1,3,224,224]` (YOLOv8 기본 640이 아닌 값) | `[1,3,640,640]` |
| conversion path | 없음 | 공식 경로: 압축 → `YOLO_netspresso(...)` 재학습 → val |
| configuration | ratio 0.5 | ratio 0.5 (동일 조건으로 비교 가능하도록 유지) |

같은 계열이지만 **다른 모델·다른 형식·다른 경로**다. 2025년 실패의 근본 원인을 2026년 케이스에 그대로 가정하지 않는다.

## 6. 현재 프레임워크에서의 Regression Scenario 표현

| 시나리오 | 표현 방법 | 기대 결과 |
|---|---|---|
| **HR-1 Historical replay (문서화 전용, 실행 안 함)** | `configs/mock_scenarios.yaml`에 ONNX 직접 투입 케이스를 `outcome: ERROR`, `error_kind: NotSupportedFrameworkException`(현 SDK 이름)으로 추가 가능 → 결함 분류 `MODEL_COMPATIBILITY_ERROR`. 2025 관찰의 *형태*를 재현하되, 실 SDK가 그렇게 응답하는지는 Credit 없이 검증 불가 → **Mock 시나리오로 한정, 실 재현은 하지 않음**(실패 유도에 Credit을 쓰지 않는다) | 분류·리포트 경로 검증 |
| **HR-2 Pre-check (0 Credit)** | TC-Y8-030: 업로드 전 로컬에서 "입력이 fx GraphModule `.pt`인가"를 구조 검증으로 확인(`validate_pt` → `object_kind == graph_module`). 2025년 제안 "사전 호환성 검증 API"를 **QA 측에서 구현**한 것 | ONNX/비-fx 입력은 업로드 전에 `INPUT_VALIDATION_ERROR`로 차단 |
| **HR-3 Current positive path** | yolov8n fx body를 공식 경로로 1회 압축(25) → 현 파이프라인 전체 적용 | 2025년에 못 간 경로의 실측 |

HR-1을 "회귀가 재현되었다"고 표현하지 않는다. 그것은 Mock 분류 경로의 테스트일 뿐이다.

### 6.1 Precheck 커버리지 (Conditional GO 해소에서 구현, `precheck_fx_bundle`)

| 확인 항목 | 구현 | 2025 실패와의 관계 |
|---|---|---|
| format (`.pt` 확장자, torch.save 컨테이너) | ✓ ONNX 입력 → **UNSUPPORTED** | 2025 실패 유형을 업로드 전에 차단 |
| file existence (body, head meta) | ✓ BLOCKED | |
| GraphModule 여부(신뢰 목록 기반 로드) | ✓ 비신뢰 → NOT_VERIFIED, nn.Module → UNSUPPORTED | 2025년 "GraphModule만 지원" 추정을 검사로 전환 |
| expected head metadata(nc/nl/stride/anchors/strides/inplace) | ✓ BLOCKED | |
| input shape(4-D, 3ch, max stride 배수) | ✓ BLOCKED | 2025년의 224 입력은 YOLOv8 기본 640과 달랐음 |
| model family / task | expected_nc·task 라벨 기록(검출 80 classes) | |
| output schema(`[N,144,H/s,W/s]`) | ✓ forward 주입 시 BLOCKED | |
| 의미 | **READY = known contract checks passed**, 서버 수락 보장 아님 | |

HR-3(실제 실행)은 수행하지 않았다. 또한 해소 작업에서 **새로운 사실**이 드러났다: 공식 포크가 만든 fx body는 upstream YOLOv8n과 기능적으로 동등하지 않다(포크 평가 0.135 vs upstream 0.444; torch 버전 무관). 이는 2025년의 형식 문제(`NotValidFrameworkException`)와는 다른 "수치 정합성" 범주의 문제이며, 두 사례를 같은 원인으로 묶지 않는다. 원인은 R1에서 확정되었다(§8).

## 7. 2025 → 2026 변화 (Q1–Q4)

**Q1. 무엇이 다른가** — 2025: 수동 관찰, 결과 미보존, 실 API를 테스트 코드가 직접 호출, 키 하드코딩, 모델 l/ONNX/224. 2026: 어댑터 경계·dry-run 기본·`--confirm-credit-use` 1회 실행·장부, 결과 JSON·SHA·환경 fingerprint 보존, 모델 n/fx·공식 경로/640, 압축 전후를 Gate 프로파일로 판정.

**Q2. 2025년에 왜 문제를 발견할 수 있었나** — 문서를 "실제 사용자처럼" 따라가며 공식 문서 문장("ONNX 지원")과 SDK 동작의 불일치를 직접 겪었기 때문. 탐색적 테스트의 가치이며, 이 능력은 2026년에도 유지되어야 한다(예: Phase 5-B에서 서버 옵션 `MIX`/`dlc`가 SDK enum에 없음을 발견).

**Q3. 2026 프레임워크는 그 문제를 자동으로 탐지하는가** — **부분적으로 가능**. (a) 업로드 전 구조 검증(HR-2)이 ONNX/비-fx 입력을 차단한다 → 2025년 실패 *유형*은 Credit을 쓰기 전에 걸린다. (b) 실 호출 시 SDK가 `status=error`를 반환하면 어댑터가 `error_detail.name`을 `MODEL_COMPATIBILITY_ERROR` 등으로 매핑한다(가짜 SDK 테스트로 검증, 실 서버 응답은 미관찰). (c) 문서 vs 동작 **불일치 그 자체**는 자동 탐지 대상이 아니다 — 사람이 읽고 판단한다.

**Q4. 추가로 탐지하는 품질 문제** — 2025년에는 묻지 않았던 것들: 산출물 무결성·구조, 파라미터/FLOPs 보고값의 독립 검증, 출력 동등성 proxy(Phase 5-D에서 압축 모델이 기능적으로 동등하지 않음을 발견), 로컬 latency/memory(FLOPs −52 %가 CPU 지연 개선으로 이어지지 않음), 재현성 레벨, 회귀 비교, 결측 데이터에서 PASS를 선언하지 않는 Gate, Credit 회계. 2025의 "SDK 실패/문서 Gap/수동 관찰"에서 2026의 "계약 + 매트릭스 + 무결성 + 구조 + 수치 검증 + 성능 + 재현성 + 회귀 + 릴리스 Gate"로 확장되었다 — 단, Accuracy·재현성 Level 2+·타깃 디바이스 성능은 **아직 실측 전**이다.

## 8. 두 개의 서로 다른 이슈: 2025 ONNX 실패 vs 2026 FX 기능 불일치

| | **2025 — ONNX 형식 실패** | **2026 — FX 기능 불일치** |
|---|---|---|
| 모델 / 입력 | YOLOv8-**l**, `yolov8l.onnx`(167 MB)를 compressor에 직접 | YOLOv8-**n**, 공식 포크 `export_netspresso()`가 만든 fx GraphModule `.pt` |
| 실패 단계 | **업로드 후 압축 단계**(서버) — `NotValidFrameworkException` | **업로드 전 로컬 검증** — 압축 입력이 upstream 모델과 다른 함수 |
| 범주 | 입력 **형식/계약**(compressor는 fx GraphModule `.pt`를 기대) | **수치 정합성**(같은 가중치인데 forward 의미가 다름) |
| 증상 | 압축 자체가 실행되지 않음 | mAP50-95 0.13505 vs 0.44369; 1 image 1 box vs 6 boxes; body cosine 0.966 |
| 원인 상태 | **NOT VERIFIED**(당시 SDK 버전·전체 메시지 부재; 저장소 근거로는 "ONNX 직접 투입" 설명과 일관) | **VERIFIED**(R1 cross-check): 포크 `C2f.forward`가 Bottleneck을 같은 입력에 병렬 적용(`y = [m(out[-1]) for m in self.m]`), upstream은 연쇄(`y.extend(m(y[-1]))`). n ≥ 2인 C2f(yolov8n layer 4·6)에서 분기. 메모리 내 연쇄 복원 시 차이 소멸(rel 1.5e-7) |
| Credit 영향 | 실패 호출에 Credit 소모(당시 장부 없음) | **0** — 업로드 전에 발견 |
| 프레임워크 대응 | HR-2 precheck: 비-fx 입력을 `UNSUPPORTED`로 차단 | R1: upstream 코드 + 최소 traceability wrapper로 fx body 생성, pre-NMS bitwise·mAP 0.0 pp 동등성 검증 후에만 압축 입력으로 인정 |
| 두 이슈의 관계 | 없음을 명시. 2025 사례는 형식 단계에서 끊겼으므로 포크 C2f 코드까지 도달하지 않았다. "YOLOv8-l이 C2f n ≥ 2를 더 많이 갖는다"는 사실로 2025 실패를 설명하지 **않는다** | |

QA 관점의 결론: **형식 계약을 통과한 산출물도 기능적 동등성은 별도로 검증해야 한다.** `completed`(2026 5-B) ≠ Release PASS였듯, `READY`(precheck) ≠ "upstream 모델과 같은 함수"다. R1 이후 precheck READY 위에 functional equivalence(pre-NMS 텐서 + 검출 + mAP) 층을 추가로 요구한다.

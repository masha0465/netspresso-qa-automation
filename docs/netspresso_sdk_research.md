# NetsPresso SDK Technical Research (Phase 1)

> Personal portfolio project using publicly available NetsPresso APIs/SDKs and technical resources.
> 본 문서는 Nota 내부 시스템과 무관한 개인 조사 기록이며, **설치된 패키지의 소스 코드와 공식 GitHub 저장소를 직접 inspect하여 확인한 사실만** 기술한다.
> 확인하지 못한 항목은 명시적으로 "미검증(NOT VERIFIED)"으로 표기한다.

| 항목 | 값 |
|---|---|
| 조사 일자 | 2026-10-01 |
| 조사 방법 | `pip` 메타데이터 조회, 패키지 설치 후 `inspect` 기반 정적 introspection, 소스 코드 열람, 공식 GitHub 예제 비교 |
| 네트워크 호출 | PyPI / GitHub 메타데이터 조회만 수행. **NetsPresso 서버 로그인·API 호출 없음** |
| Credit 사용 | **0 / 500** |
| 산출물 | `docs/research_cache/sdk_surface.json` (introspection 결과), `docs/research_cache/inspect_sdk.py`, `requirements/netspresso-py311.lock.txt` |
| 공개 출처 | §0 참조. 조사 중 열람한 공식 예제·저장소 파일의 **사본은 본 저장소에 포함하지 않는다** |

---

## 0. 공개 출처 (Sources)

본 조사에 사용한 자료는 모두 공개 자료이며, 아래 출처에서 직접 확인했다.

| 자료 | 출처 | 조사에서의 용도 |
|---|---|---|
| PyPI 패키지 `netspresso` 1.17.0 | https://pypi.org/project/netspresso/ (메타데이터 API: `https://pypi.org/pypi/netspresso/json`) | `requires_python`, 의존성 체인, wheel 태그 확인, 설치 |
| 공식 GitHub 저장소 (PyNetsPresso) | https://github.com/Nota-NetsPresso/PyNetsPresso | README / `setup.py` / `requirements.txt` / `examples/` 열람, 태그 목록 확인 |
| 비교 기준 태그 | `v1.17.0` (https://github.com/Nota-NetsPresso/PyNetsPresso/tree/v1.17.0) — PyPI 1.17.0과 동일 버전 | `main` 브랜치(1.1.6 표기, §8.4)와 설치 버전의 불일치를 식별한 뒤, 릴리스 태그를 기준으로 삼음 |
| 설치된 패키지 소스 (`.venv-netspresso/Lib/site-packages/netspresso/`) | PyPI wheel에서 설치 | 모든 시그니처·enum·예외·Credit 상수·인증 흐름의 **source of truth** |

조사 과정에서 공식 `examples/` 디렉터리의 예제 스크립트(compressor, converter & profiler, quantizer, graph_optimizer, inferencer, np_qai, trainer)와 `main` 브랜치의 README/`setup.py`/`requirements.txt`를 로컬로 내려받아 설치 버전과 비교했다. 이 사본들은 제3자 저작물이므로 **포트폴리오 저장소에는 포함하지 않았다**. 비교 결론만 §6에 기록하며, 원문은 위 GitHub 저장소에서 확인할 수 있다. 샘플 모델/데이터셋 바이너리는 다운로드하지 않았다.

---

## 1. Environment Requirements

### 1.1 조사 환경

| 구성 | 값 |
|---|---|
| OS | Windows 11 Pro (10.0.26200), AMD64, 20 cores, GPU 없음 |
| 메인 Python | 3.14.6 (`py` Python install manager, PythonCore) |
| SDK 전용 Python | 3.11.9 (`py install 3.11`로 추가 설치, 3.14는 변경 없음) |
| SDK venv | `.venv-netspresso/` (프로젝트 루트, `.gitignore` 대상) |
| venv 크기 | 약 1.8 GB, 패키지 156개 |

### 1.2 공식 문서상 요구사항 vs 실제

| 출처 | 명시된 Python 요구사항 |
|---|---|
| GitHub `main` README | "Python 3.8 or higher", 배지: 3.8 / 3.9 / 3.10 |
| GitHub `main` `setup.py` | `python_requires=">=3.8"`, classifiers 3.8~3.10 |
| PyPI `netspresso 1.17.0` 메타데이터 | `requires_python >= 3.10` |
| **실제 설치 가능 상한 (의존성 추적 결과)** | **Python 3.11** (아래 §2 참조) |

---

## 2. Python Compatibility

### 2.1 QA Finding: ENVIRONMENT / CONFIGURATION

**분류**: `ENVIRONMENT_ERROR` (설치 단계), 부수적으로 `CONFIGURATION_ERROR` (문서-메타데이터 불일치)
**심각도**: Medium (신규 사용자 온보딩 차단, 우회 가능)
**재현성**: 결정적 (Deterministic)

**관찰된 사실 (검증됨)**

`netspresso==1.17.0`의 의존성 체인을 PyPI 메타데이터로 추적한 결과:

```
netspresso 1.17.0                 requires_python >= 3.10
 ├─ netspresso-trainer == 1.3.1
 │    ├─ torch       >=1.11.0, <=2.0.1   → wheel 제공: cp38, cp39, cp310, cp311
 │    └─ torchvision >=0.12.0, <=0.15.2  → wheel 제공: cp38, cp39, cp310, cp311
 ├─ netspresso-inference-package == 0.1.4
 │    └─ tensorflow >= 2.8.0             → 최신 2.21.0 wheel: cp310 ~ cp313
 └─ qai-hub == 0.40.0
```

- `torch<=2.0.1` 핀이 실질적 상한을 **Python 3.11**로 고정한다.
- PyPI `requires_python>=3.10`은 하한만 기술하며, 상한(3.11)은 어떤 문서에도 명시되어 있지 않다.
- GitHub `main` 브랜치의 README/`setup.py`/`requirements.txt`는 1.1.6 시절 내용(pydantic 1.10.4 등)으로 PyPI 1.17.0과 불일치한다 (§8.4 참조).

**영향**: Python 3.12 이상 환경에서는 `pip install netspresso`가 torch wheel 해석 단계에서 실패한다. 본 프로젝트의 메인 환경(3.14)이 이에 해당한다.

**Suspected cause**: `netspresso-trainer`의 torch 상한 핀이 최신 SDK 릴리스에 그대로 승계되었고, 설치 메타데이터(`requires_python`)가 실제 의존성 제약을 반영하도록 갱신되지 않았다. (소스 작성자의 의도는 확인할 수 없으므로 원인 단정은 하지 않는다.)

**우회**: Python 3.11 전용 가상환경 분리 (§10 아키텍처 결정으로 채택).

**과장 금지 주석**: 이 이슈는 Python 3.10/3.11에서는 재현되지 않으며, SDK 기능 결함이 아닌 패키징/문서 정합성 이슈다.

### 2.2 검증 결과

| 환경 | `import netspresso` | 비고 |
|---|---|---|
| Python 3.14.6 (메인) | `ModuleNotFoundError` | 의도된 상태. 설치 시도하지 않음 |
| Python 3.11.9 (`.venv-netspresso`) | 성공 (`__version__ == "1.17.0"`) | import 시 TensorFlow가 함께 로드되어 수 초 소요 |

---

## 3. Installed NetsPresso Version

| 패키지 | 버전 | 근거 |
|---|---|---|
| **netspresso** | **1.17.0** | `netspresso/VERSION` 파일, `pip freeze`, PyPI 최신과 동일 (GitHub tag `v1.17.0`) |
| netspresso-trainer | 1.3.1 | 하드 핀 |
| netspresso-inference-package | 0.1.4 | 하드 핀 |
| qai-hub | 0.40.0 | 하드 핀 |

전체 고정 목록: `requirements/netspresso-py311.lock.txt` (151개).

---

## 4. Dependency Constraints

### 4.1 주요 의존성 (3.11 venv 실제 설치 버전)

| 패키지 | 버전 | 메모 |
|---|---|---|
| torch / torchvision | 2.0.1 / 0.15.2 | 상한 핀. CPU 빌드 |
| tensorflow | 2.21.0 | `netspresso-inference-package` 경유. import 시 oneDNN 로그 출력 |
| onnx / onnxruntime | 1.23.1 / 1.30.0 | |
| numpy | 1.26.4 | torch 2.0.1 호환 범위 |
| pydantic | 2.13.5 | GitHub main의 `requirements.txt`(1.10.4)와 상이 |
| protobuf | 6.31.1 | qai-hub 상한 `<=6.31.1` |
| loguru | 0.7.3 | SDK가 import 시 전역 logger를 재설정함 (§8.5) |
| python-dotenv | 1.2.3 | SDK가 `netspresso.env` 파일을 자동 탐색 (§8.6) |
| gradio | 3.50.2 | trainer 의존. 본 프로젝트에서 사용하지 않음 |

### 4.2 메인 3.14 환경에 대한 결정

QA 프레임워크(3.14)에는 위 패키지 중 **어느 것도 설치하지 않는다**. 프레임워크가 필요로 하는 것은 SDK가 파일로 남기는 `metadata.json`과 산출 모델 파일이며, 이는 표준 라이브러리 + PyYAML + Jinja2 + pytest로 처리 가능하다. 로컬 정확도/지연 평가가 필요해지면 `onnxruntime`(cp314 wheel 존재 확인)만 선택적으로 추가한다.

---

## 5. API Structure

### 5.1 진입점 (Facade)

```python
from netspresso import NetsPresso, NPQAI
NetsPresso(email=None, password=None, api_key=None, verify_ssl=True, dev_mode=False)
```

`NetsPresso.__init__` 실행 순서 (소스 `netspresso/netspresso.py` 확인):

1. `_check_version()` — PyPI(`https://pypi.org/pypi/netspresso/json`)에 접속하여 최신 버전과 비교. **설치 버전이 낮으면 `sys.exit(1)`** (`dev_mode=True`로 우회 가능). PyPI 접속 실패도 `False` → 종료.
2. `TokenHandler(api_key=...)` — `POST /api/v3/auth/login_by_api_key` (헤더 `api-key`). email/password 방식은 코드상 `[DEPRECATED]` 경고.
3. `get_user()` — `GET /users/me` 및 `GET /users/{user_id}/credit/summarized` → `UserResponse(credit_info=CreditResponse(free, reward, contract, paid, total))`.

즉 **객체 생성 = 즉시 로그인 + 사용자/크레딧 조회**이며, 지연(lazy) 인증은 없다.

엔드포인트: `https://v2-prod.netspresso.ai:43001` (`clients/configs/config-v2-prod-cloud.ini`), `DEPLOYMENT_MODE` 환경변수로 전환 가능. 접두사: AUTH `/api/v3`, COMPRESSOR `/api/v2/compressor`, LAUNCHER `/api/v2/launcher`.

### 5.2 서비스 팩토리

| 팩토리 메서드 | 반환 클래스 | 백엔드 |
|---|---|---|
| `compressor_v2()` | `CompressorV2` | Compressor API |
| `converter_v2()` | `ConverterV2` | Launcher API |
| `profiler()` | `Profiler` | Launcher API (benchmark) |
| `quantizer()` | `Quantizer` | Launcher API |
| `graph_optimizer()` | `GraphOptimizer` | Launcher API |
| `simulator()` | `Simulator` | Launcher API |
| `trainer(task, yaml_path)` | `Trainer` | 로컬 `netspresso-trainer` |
| `np_inferencer(config_path, input_model_path)` | `NPInferencer` | 로컬 추론 |
| `custom_inferencer(input_model_path)` | `CustomInferencer` | 로컬 추론 |

`NPQAI(api_token)`은 Qualcomm AI Hub 래퍼로 `qai-hub configure` 서브프로세스를 실행한다. 본 프로젝트 범위 밖.

### 5.3 공통 기반 `NetsPressoBase` (`netspresso/base.py`)

- `validate_token_and_check_credit(service_task)` → 토큰 만료 시 재로그인 → `auth_client.get_credit()`로 현재 잔액 조회 → `ServiceCredit.CREDITS[service_task]`와 비교 → 부족 시 `NotEnoughCreditException`.
- `print_remaining_credit(service_task)` → 성공 후 잔액을 다시 조회하여 로그 출력.
- `handle_error / handle_stop` → metadata의 `status`를 `ERROR` / `STOPPED`로 설정 (예외를 다시 던지지 않음, §8.2).

### 5.4 결과 구조 (Result Model)

모든 서비스는 `dataclass` 기반 `*Metadata`를 반환하고, `finally` 블록에서 `MetadataHandler.save_metadata()`로 `output_dir/metadata.json`에 저장한다.

공통 필드 (`BaseMetadata`): `status: Status(IN_PROGRESS|COMPLETED|STOPPED|ERROR)`, `error_detail: ExceptionDetail(error_code, name, message, data.error_log)`.

| Metadata | 주요 필드 |
|---|---|
| `CompressorMetadata` | `input_model_path`, `compressed_model_path`, `compressed_onnx_model_path`, `is_retrainable`, `model_info(task, model, dataset, data_type, framework, input_shapes)`, `compression_info(method, ratio, options, layers)`, `results.original_model / compressed_model` → `Model(size, flops, number_of_parameters, trainable_parameters, non_trainable_parameters, number_of_layers, model_id)` |
| `ConverterMetadata` | `converted_model_path`, `model_info`, `convert_task_info(convert_task_uuid, framework, device_name, data_type, software_version, ...)` |
| `ProfilerMetadata` | `profile_task_info(benchmark_task_uuid, framework, device_name, software_version, data_type, ...)`, `profile_result: BenchmarkResult(memory_footprint, memory_footprint_gpu, memory_footprint_cpu, power_consumption, ram_size, latency, file_size)` — **모두 `int` 타입** |
| `QuantizerMetadata` | `quantized_model_path`, `recommendation_result_path`, `quantize_info(quantization_mode, metric, threshold, weight_precision, activation_precision, ...)` |
| `GraphOptimizerMetadata` | `optimized_model_path`, `graph_optimize_task_info(pattern_handlers, status, ...)` |
| `SimulatorMetadata` | (introspection 범위 밖, 시그니처만 확인) |

QA 관점 메모: `BenchmarkResult.latency`가 `int`로 선언되어 있어 **ms 단위 소수점 정밀도가 보존되지 않을 가능성**이 있다. 실제 응답 값의 단위·정밀도는 실 실험(Phase 12)에서 확인해야 하며 현재는 미검증이다.

### 5.5 Enum (총 33개, 주요 항목)

| Enum | 값 |
|---|---|
| `Framework` | TENSORFLOW_KERAS, TENSORFLOW(saved_model), PYTORCH, ONNX, TENSORRT, OPENVINO, TENSORFLOW_LITE, DRPAI |
| `DataType` | FP32, FP16, INT8, NONE |
| `DeviceName` (22) | RaspberryPi 5/4B/3B+/3B/2B/ZeroW/Zero2W, Renesas RZ/V2L·RZ/V2M·RA8D1, Jetson Nano/TX2/Xavier/NX/AGX-Orin/Orin-Nano, AWS-T4, Intel-Xeon(W-2233), Alif Ensemble E7, Arm Ethos-U, NXP i.MX93, Arduino Nicla Vision |
| `SoftwareVersion` | Jetpack 4.4.1 / 4.6 / 5.0.1 / 5.0.2 / 6.1 |
| `HardwareType` | HELIUM |
| `CompressionMethod` | PR_L2, PR_GM, PR_NN, PR_SNP, PR_ID, FD_TK, FD_CP, FD_SVD |
| `RecommendationMethod` | SLAMP, VBMF |
| `QuantizationMode` | AUTOMATIC, UNIFORM_PRECISION, CUSTOM_PRECISION, ADVANCED, RECOMMEND |
| `QuantizationPrecision` | INT8, FLOAT16, FLOAT32 |
| `SimilarityMetric` | SNR (단일) |
| `Runtime` | ONNX, TFLITE |
| `Status` | in_progress, completed, stopped, error |
| `TaskStatus` | IN_QUEUE, IN_PROGRESS, FINISHED, ERROR, TIMEOUT, USER_CANCEL |
| `ServiceTask` | TRAINING, ADVANCED_COMPRESSION, AUTOMATIC_COMPRESSION, MODEL_CONVERT, MODEL_PROFILE, MODEL_QUANTIZE, MODEL_GRAPH_OPTIMIZE, MODEL_SIMULATE |

**매트릭스 설계 시 주의**: SDK에는 범용 "CPU" 디바이스가 없다. 가장 가까운 x86 타깃은 `INTEL_XEON_W_2233`이며, 특정 Framework×Device×DataType 조합의 지원 여부는 서버가 반환하는 `AvailableOption`으로 판단해야 한다(로컬에서 enum 조합만으로 SUPPORTED를 단정할 수 없음 → 매트릭스 초기 상태는 `NOT_TESTED`).

### 5.6 예외 (모두 `PyNPException(Exception)` 상속, 27종)

`NotEnoughCreditException`, `FailedUploadModelException`, `NotSupportedFrameworkException`, `NotSupportedModelException`, `NotSupportedSuffixException`, `NotSupportedTaskException`, `NotValidInputModelPath`, `NotValidSlampRatioException`, `NotValidVbmfRatioException`, `NotValidChannelAxisRangeException`, `NotFillInputLayersException`, `NotSetDatasetException`, `NotSetModelException`, `EmptyCompressionParamsException`, `LoadONNXModelException`, `UpdateOnnxException`, `GatewayTimeoutException`, `InternalServerErrorException`, `FailedFetchPackageException`, `FailedTrainingException`, `RetrainingFunctionException`, `TaskOrYamlPathException`, `DirectoryNotFoundException`, `BaseDirectoryNotFoundException`, `FileNotFoundErrorException`, `UnexpetedException`(원문 오타 그대로).

→ Phase 8 결함 분류 매핑 초안: `NotEnoughCredit`→`ENVIRONMENT_ERROR`(예산), `NotSupported*`→`MODEL_COMPATIBILITY_ERROR`, `NotValid*`/`NotFill*`/`NotSet*`→`INPUT_VALIDATION_ERROR`, `GatewayTimeout`/`InternalServerError`→`RUNTIME_ERROR`, `FailedUploadModel`→`ARTIFACT_ERROR` 또는 `RUNTIME_ERROR`(증거 기반 판단), 로그인 실패(SDK 자체 예외 없음, `requests` 예외 전파)→`AUTH_ERROR`.

---

## 6. Available Operations

| 서비스 | 메서드 | 입력 | 대기 방식 | 결과 |
|---|---|---|---|---|
| CompressorV2 | `automatic_compression(input_model_path, output_dir, input_shapes, framework=PYTORCH, compression_ratio=0.5)` | PyTorch GraphModule `.pt` / ONNX 등 | 동기 (서버 응답 즉시) | `CompressorMetadata`, 압축 모델 + ONNX |
| CompressorV2 | `recommendation_compression(compression_method, recommendation_method, recommendation_ratio, ...)` | 동일 | 동기 | 동일 |
| CompressorV2 | `upload_model / select_compression_method / compress_model` | 수동 단계 | 동기 | 동일 (Advanced) |
| ConverterV2 | `convert_model(input_model_path, output_dir, target_framework, target_device_name, target_data_type=FP16, target_software_version=None, input_layer=None, dataset_path=None, wait_until_done=True, sleep_interval=30)` | ONNX / Keras 등 | 폴링 (`sleep_interval` 초) | `ConverterMetadata`, 변환 모델 |
| Profiler | `profile_model(input_model_path, target_device_name, target_software_version=None, target_hardware_type=None, wait_until_done=True, sleep_interval=30)` | 변환된 모델 | 폴링 | `ProfilerMetadata.profile_result` |
| Quantizer | `automatic_quantization(input_model_path, output_dir, dataset_path, weight_precision=INT8, activation_precision=INT8, metric=SNR, threshold=0, ...)` | ONNX + `.npy` 캘리브레이션 | 폴링 | `QuantizerMetadata` |
| Quantizer | `uniform_precision_quantization`, `custom_precision_quantization_by_layer_name / _by_operator_type`, `get_recommendation_precision` | ONNX | 폴링 | 동일 |
| GraphOptimizer | `optimize_model(input_model_path, output_dir, pattern_handlers=[13종 기본], wait_until_done=True, sleep_interval=30)` | ONNX | 폴링 | `GraphOptimizerMetadata` |
| Simulator | `simulate_model(base_model_path, target_model_path, output_dir, dataset_path=None)` | 모델 2개 | 동기 | `SimulatorMetadata` |
| 공통 | `get_*_task(task_id)`, `cancel_*_task(task_id)` | task id | 단발 | 태스크 상태 |
| 로컬 | `CustomInferencer.inference(dataset_path)`, `NPInferencer.inference(image_path, save_path)` | 로컬 파일 | 로컬 | numpy 출력 |
| 로컬 | `Trainer.train(gpus, project_name, output_dir)` | 데이터셋 + 설정 | 로컬 (GPU 필요) | `TrainerMetadata` |

공식 예제(GitHub 저장소 `examples/` 디렉터리, §0 출처; 사본은 본 저장소에 미포함)와 설치 버전 비교:

- 예제는 전부 `NetsPresso(email=EMAIL, password=PASSWORD)`를 사용하지만 설치 버전에서는 이 경로가 deprecated이고 `api_key=` 경로가 존재한다. **본 프로젝트는 `api_key` 경로를 사용한다.**
- 예제의 함수명·인자명은 설치 버전 시그니처와 일치함을 확인 (`automatic_compression`, `convert_model`, `profile_model`, `automatic_quantization`, `optimize_model`).
- 공식 샘플 모델 (GitHub, 미다운로드): `yolo-fastest.onnx` 1.16 MB, `test.onnx` 3.77 MB, `graphmodule.pt` 27.17 MB, `mobilenetv1.h5` 12.96 MB 등. 캘리브레이션 데이터: `pickle_calibration_dataset_128x128.npy` 1.17 MB 등.

---

## 7. Potential Credit-Consuming Operations

### 7.1 SDK에 내장된 비용 테이블 (`netspresso/enums/credit.py`, 검증됨)

| ServiceTask | 클라이언트 측 요구 Credit | 클라이언트 측 사전 체크 |
|---|---|---|
| ADVANCED_COMPRESSION (`compress_model`, `recommendation_compression`) | **50** | 있음 |
| AUTOMATIC_COMPRESSION (`automatic_compression`) | **25** | 있음 |
| MODEL_CONVERT (`convert_model`) | **50** | 있음 |
| MODEL_PROFILE (`profile_model`) | **25** | 있음 |
| MODEL_QUANTIZE (`*_quantization`, `get_recommendation_precision`) | **50** | 있음 |
| MODEL_GRAPH_OPTIMIZE (`optimize_model`) | **50** (테이블에 존재) | **없음** — 소스에 `check_credit_balance` 호출이 없음 |
| MODEL_SIMULATE (`simulate_model`) | 테이블에 **없음** | 없음 |
| TRAINING (`Trainer.train`, 로컬 실행) | 테이블에 없음 | 없음 |

**중요한 해석 한계**: 위 숫자는 SDK가 *사전 잔액 확인*에 사용하는 클라이언트 측 상수다. **실제 차감액은 서버가 결정**하며, 본 조사에서는 서버 측 과금을 검증하지 않았다. 따라서:

- 클라이언트 체크가 없는 GraphOptimizer / Simulator를 "무료"로 간주하지 않는다. → **잠재적 과금 작업으로 취급, 실행 전 확인 필수**.
- 실패한 작업의 환불 여부는 미검증이다. 소스상 `print_remaining_credit`은 성공 시에만 호출되어 실패 시 잔액 변화를 SDK가 보고하지 않는다.
- `profile_model`은 폴링 결과가 `FINISHED`일 때만 잔액을 출력하므로, TIMEOUT/ERROR 시 실제 차감 여부를 별도 조회로 확인해야 한다.

### 7.2 인증 / 조회 작업

`NetsPresso()` 생성, `get_user()`, `get_credit()`, `get_*_task()`, `cancel_*_task()`는 `ServiceTask`에 매핑되지 않으며 SDK 코드상 Credit 체크 대상이 아니다. SDK가 매 서비스 호출 전후 `get_credit()`을 내부적으로 호출하는 점을 보면 조회 작업이 과금될 가능성은 낮다고 **추정**된다. 그러나 서버 정책을 확인한 것이 아니므로 **첫 로그인도 사용자 확인 후 1회만 수행**하고, 그 전후 `credit_info.total`을 기록하여 검증한다.

### 7.3 Credit Ledger 연동 계획 (Phase 10)

`UserResponse.credit_info.total`(로그인 시) 및 `auth_client.get_credit()`(작업 전후)을 통해 **실제 잔액을 조회할 수 있는 경로가 존재**함을 확인했다. 이를 이용해 `reports/credit_usage.json`에 `usage_type: "actual"`을 기록할 수 있으며, 조회 실패 시 `"estimated"`로 강등한다.

### 7.4 예산 시사점 (500 Credit 기준, 클라이언트 테이블 가정)

| 시나리오 | 구성 | 합계 |
|---|---|---|
| 최소 E2E 1회 | automatic_compression 25 + profile 25 | 50 |
| 최소 E2E + 변환 | + convert 50 | 100 |
| 최소 E2E + 양자화 | + quantization 50 | 100~150 |
| 기존 목표(350~400) 내 수행 가능 실 작업 수 | 약 8~12회 | — |

→ Phase 11 첫 실험은 `automatic_compression`(25) 단독 또는 `automatic_compression + profile_model`(50)이 가장 저렴한 E2E 후보다. 실행 여부·구성은 별도 승인 단계에서 결정한다.

---

## 8. Known Limitations (검증된 관찰)

1. **버전 게이트 하드 종료**: `_check_version()`이 PyPI에 접속해 구버전이거나 접속 실패 시 `sys.exit(1)`을 호출한다. 오프라인·프록시 환경, PyPI 장애, 릴리스 직후 pip 캐시 지연 시 SDK가 통째로 종료된다. 라이브러리 코드의 `sys.exit`는 호출자(테스트 러너 포함)를 함께 종료시킬 수 있어 어댑터에서는 `dev_mode=True` 사용 여부를 검토해야 한다 (버전 체크 자체는 어댑터가 별도로 기록).
2. **예외 삼킴(swallow)**: 서비스 메서드는 `except Exception` 후 `metadata.status = ERROR`로 정상 반환한다. 호출자는 반환값의 `status`/`error_detail`을 반드시 검사해야 하며, 예외 기반 흐름 제어를 기대하면 실패를 놓친다. → 어댑터는 `status != COMPLETED`를 명시적 실패로 승격한다.
3. **`error_detail.message`가 `e.args[0]`**: 인자 없는 예외에서는 `IndexError`가 발생할 수 있는 구조(코드 경로상 가능성, 재현은 미검증).
4. **GitHub `main` 브랜치 정합성**: `netspresso/__init__.py`가 `1.1.6`, `requirements.txt`가 pydantic 1.10.4 등 PyPI 1.17.0과 불일치. 릴리스는 태그(`v1.17.0`) 기준으로 관리되는 것으로 보이며, 공식 예제 참고 시 태그 소스를 기준으로 삼아야 한다.
5. **전역 logger 재설정**: `netspresso/__init__.py`가 import 시 `loguru.logger.remove()` 후 자체 포맷을 추가한다. 동일 프로세스의 다른 loguru 사용자의 설정이 덮어써진다. (3.11 실험 스크립트가 격리 프로세스이므로 본 프로젝트 영향은 제한적.)
6. **`netspresso.env` 자동 로드**: `clients/config.py`가 `find_dotenv("netspresso.env")`로 상위 디렉터리를 탐색하고 `HOST`/`PORT` 환경변수로 엔드포인트를 덮어쓴다. 우연히 존재하는 `HOST`/`PORT` 환경변수가 엔드포인트를 바꿀 수 있다. 어댑터는 실행 시 실제 해석된 host를 로그에 기록해 추적성을 확보한다.
7. **텔레메트리**: 모든 서비스 호출이 `netspresso_analytics.send_event()`로 Google Analytics(GA4)에 이벤트를 전송하고, `is_gdpr_country()`가 `https://ipinfo.io/json`에 접속한다. `GA_DISABLE_ANALYTICS` 환경변수로 비활성화 가능(코드 확인). 실험 재현성·네트워크 격리 관점에서 실 실험 스크립트는 이를 명시적으로 설정한다.
8. **Import 비용**: `import netspresso`가 TensorFlow·torch를 로드하여 수 초가 걸리고 stderr에 oneDNN 로그가 출력된다. 3.14 프레임워크가 이를 import하지 않는 설계 근거 중 하나.
9. **폴링 기본값 30초**: `sleep_interval=30`이 기본이며, 짧게 줄이면 서버 요청 수가 늘어난다. 서버 rate-limit은 미검증.
10. **`BenchmarkResult` 필드가 모두 `int`**: latency/memory의 단위와 정밀도는 실 응답으로 확인해야 함(미검증).
11. **GPU 없음**: `Trainer`(로컬 학습)와 GPU 대상 로컬 검증은 본 환경에서 불가. 클라우드 Profiler는 서버 측 디바이스 팜을 사용하므로 로컬 GPU와 무관.
12. **범용 "CPU" 디바이스 부재**: §5.5 참조. 매트릭스의 "CPU" 열은 `INTEL_XEON_W_2233` 등 실제 enum으로 매핑하고 지원 여부는 `NOT_TESTED`에서 출발한다.

---

## 9. Windows-Specific Considerations

| 항목 | 관찰 | 대응 |
|---|---|---|
| 경로 처리 | SDK는 `pathlib.Path`와 `os.path` 혼용. `subprocess`는 `NPQAI`(qai-hub CLI)에서만 사용되며 본 프로젝트 범위 밖 | 프레임워크는 `pathlib` 전용, JSON에는 POSIX 형식(`as_posix()`)으로 기록 |
| `/tmp`, `chmod`, 셸 스크립트 | SDK 소스에서 발견되지 않음 | Windows 실행 차단 요소 없음 (실 실행은 미검증) |
| CRLF | `git config core.autocrlf` 미설정 | `.gitattributes`로 `*.py text eol=lf`, 바이너리 모델 파일은 `binary` 지정 예정. 체크섬은 항상 바이너리 모드로 계산 |
| venv 크기 | 1.8 GB (torch + tensorflow) | `.gitignore`에 `.venv-netspresso/` 추가 (본 Phase에서 적용) |
| 콘솔 인코딩 | loguru 컬러 코드가 stderr에 출력 | 실험 스크립트는 `NO_COLOR`/파일 로깅 병행 검토 |
| Python install manager | `py install 3.11`로 사용자 로컬 설치, 3.14 기본 유지 확인 (`py list`) | 문서화 완료 |

---

## 10. Architecture Decision

### ADR-001: Interpreter-level isolation of the NetsPresso SDK

**결정**: QA 프레임워크(Python 3.14)와 NetsPresso SDK(Python 3.11 `.venv-netspresso`)를 **서로 다른 인터프리터로 분리**하고, 두 환경은 **JSON 결과 파일**로만 통신한다. 3.14 코드는 `netspresso`를 import하지 않는다.

**근거 (검증된 사실 기반)**

1. §2의 torch 상한 핀으로 3.14에서 SDK 설치 자체가 불가하다.
2. SDK import가 TensorFlow/torch를 동반 로드하여 무겁고 부수효과(전역 logger 재설정, PyPI 접속, 텔레메트리)가 있다. QA 엔진과 CI가 이 부수효과에 노출될 이유가 없다.
3. SDK가 이미 모든 작업 결과를 `metadata.json`(dataclass → JSON)으로 저장한다. 파일 기반 계약은 SDK의 기존 동작과 자연스럽게 일치한다.
4. Credit 안전성: 실 API를 호출할 수 있는 코드가 별도 인터프리터·별도 스크립트(`scripts/run_real_netspresso.py --confirm-credit-use`)에만 존재하므로, pytest·GitHub Actions에서 실수로 과금 API를 호출할 경로가 구조적으로 차단된다.

**결과**

```
Python 3.11 (.venv-netspresso)                 Python 3.14 (main)
┌──────────────────────────────┐   JSON   ┌──────────────────────────────┐
│ scripts/run_real_netspresso  │ ───────► │ framework/ (pure Python)      │
│  └ NetsPressoAdapter         │  files   │  ├ adapters/base.py           │
│     (netspresso SDK 1.17.0)  │          │  ├ adapters/mock_adapter.py   │
│  └ credit ledger writer      │          │  ├ result loader (real runs)  │
└──────────────────────────────┘          │  ├ validation / quality gate  │
        │ 실 API (승인 시에만)              │  └ reporter (JSON/HTML)       │
        ▼                                 └──────────────────────────────┘
  v2-prod.netspresso.ai:43001                       ▲ pytest / CI (Credit 0)
```

**트레이드오프**: `NetsPressoAdapter`의 실제 SDK 호출 코드는 3.14 pytest에서 직접 실행할 수 없다. 대신 (a) 3.14에서는 SDK 응답 JSON 픽스처를 이용한 파서/매핑 단위 테스트를 수행하고, (b) 3.11에서는 `--dry-run` 모드(로그인·과금 없음)로 스크립트 골격을 검증한다. 이 한계는 README Limitations에 명시한다.

**대안 검토**: (1) 전체를 3.11로 통일 — CI 매트릭스에 1.8 GB 의존성이 들어가고 SDK 부수효과가 QA 엔진에 전파되어 기각. (2) `pip install --no-deps netspresso` — `netspresso.py`가 최상위에서 `Trainer`(netspresso-trainer→torch)를 import하므로 import 단계에서 실패, 기각.

---

---

## 11. Phase 5-B 실 실행 관찰 (2026-10-05, automatic_compression 1회)

Phase 1까지의 결론은 정적 조사였다. 아래는 실제 1회 실행에서 **관찰된 사실**이며, 실험 산출물은 `reports/real_runs/20261005T010356Z_automatic_compression/`에 있다.

| 항목 | 관찰 |
|---|---|
| 시그니처 | 실행 직전 `inspect.signature`로 재확인: §6 표와 완전히 일치 |
| 인증 | `NetsPresso(api_key=...)` 생성 시 PyPI 버전 확인(1.17.0 = 최신) → `login_by_api_key` → 사용자/크레딧 조회가 수행됨 (§5.1 기술과 일치). 로그에 키 값은 출력되지 않음 |
| Credit (§7 미검증 항목 일부 해소) | 계정 잔액 **500 → 475**. `automatic_compression`의 실제 차감 **25 = 클라이언트 상수**. SDK 로그 "25 credits have been consumed. Remaining Credit: 475"와 잔액 조회가 일치. 다른 작업(convert/profile/quantize/graph_optimize)의 실 차감은 **여전히 미검증** |
| 인증/조회 과금 | 로그인·사용자·크레딧 조회(세션 1회, 조회 수회) 후에도 잔액이 500에서 변하지 않았음 → 조회 작업은 이 관찰 범위에서 **비과금** |
| 결과 메타데이터 | `status=completed`, `compression_info.method=PR_L2`, `ratio=0.5`, `layers` 50개, `results.original_model/compressed_model`의 size(27.17→7.11, 단위는 MB로 추정되나 SDK가 명시하지 않음)·flops·number_of_parameters 제공. `number_of_layers`는 null |
| 산출물 | `sdk_output.pt`(7,456,333 bytes) 다운로드 후 SDK가 로컬에서 `torch.onnx.export`로 `sdk_output.onnx`(7,275,162 bytes)를 추가 생성. 즉 **ONNX 변환은 서버가 아니라 클라이언트에서 수행**됨 (torch 2.0.1) |
| 서버 옵션 vs SDK enum | 서버가 반환한 변환 옵션(`available_options`)에 SDK enum에 없는 값이 포함됨: data type **`MIX`**(Jetson-AGX-Orin), software version **`6.2.1+b38`**(Jetpack 6.2.1), framework **`dlc`**(Samsung Galaxy S24 Ultra, SNPE 2.20.0). SDK는 알려지지 않은 framework `dlc`를 metadata에 **기록하지 않고 조용히 제외**했다(콘솔 응답에는 5개 framework, metadata.json에는 4개). → 클라이언트 enum이 서버 기능을 뒤따라가지 못하는 **버전 드리프트**가 존재하며, 매트릭스의 지원 지식을 SDK enum만으로 정의하면 서버가 실제로 지원하는 조합을 놓칠 수 있다. 분류: `CONFIGURATION_ERROR`(클라이언트-서버 메타데이터 불일치), 심각도 Low(기능 차단은 아님) |
| 실행 시간 | 인증 ~14 s, 업로드 27.2 MB ~5 s, 압축 ~10 s, 다운로드·후처리 ~2 s (단일 관찰, 통계적 의미 없음) |

Windows에서의 E2E 동작(§9 미검증 항목)은 이 1회 실행으로 확인되었다.

## 부록 A. 본 Phase에서 수행한 명령 (재현용)

```bash
py install 3.11                       # Python 3.11.9 추가 (3.14 유지)
python3.11 -m venv .venv-netspresso
.venv-netspresso/Scripts/python.exe -m pip install --upgrade pip
.venv-netspresso/Scripts/python.exe -m pip install netspresso           # 1.17.0
.venv-netspresso/Scripts/python.exe docs/research_cache/inspect_sdk.py docs/research_cache/sdk_surface.json
.venv-netspresso/Scripts/python.exe -m pip freeze > requirements/netspresso-py311.lock.txt
```

`inspect_sdk.py`는 실행 전 환경에서 `*NETSPRESSO*KEY*` 변수를 제거하고 `NetsPresso()`를 인스턴스화하지 않는다. 네트워크 호출: 없음 (PyPI/GitHub 메타데이터 조회는 별도 명령으로 수행).

## 부록 B. 미검증 항목 요약 (NOT VERIFIED)

- 서버 측 실제 Credit 차감액 (automatic_compression은 §11에서 25 확인; 그 외 작업 미검증) 및 실패 시 환불 정책
- 로그인/조회 API의 과금 여부 → §11에서 1회 관찰 범위 내 비과금 확인
- GraphOptimizer / Simulator의 서버 측 과금
- `BenchmarkResult` 각 필드의 단위·정밀도
- 특정 Framework × Device × DataType 조합의 서버 지원 여부
- ~~Windows에서 실 API 호출의 End-to-End 동작~~ → §11에서 1회 확인
- 서버 rate-limit

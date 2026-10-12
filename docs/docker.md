# Docker에서 실행

CUDA Toolkit과 Python 의존성을 이미지 안에서 새로 설치합니다. 호스트 `.env`와 이미 컴파일한 CuRobo 바이너리는 이미지에 복사하지 않습니다. 호스트 NVIDIA 드라이버는 기존 것을 사용합니다. 설치 명령의 `apt`·`pip`는 Dockerfile의 build 단계에서만 실행됩니다.

빌드 로그는 `reports/docker-build.log`, 실제 컨테이너 검증은 `reports/docker-validation/` 또는 `reports/docker-parallel/` 아래에 저장합니다. 기존 호스트 simulator의 성공 기록과 Docker 검증 결과는 구분합니다.

2026-10-09 실제 빌드·단일/복수 GPU 실행 검증을 완료했습니다. [결과와 확인한 범위](validation.md), [이미지 ID와 상세 보고서](../reports/docker-migration.json)를 확인하세요. 현재 이미지의 Docker 표시 용량은 약 24.1GB이며 assets·checkpoint는 별도입니다. 최종 빌드 로그는 `reports/docker-build-final.log`입니다.

## 빌드와 검증

```bash
cd /home/guest/robotwin-workspace
# 호스트 .env 활성화는 필요하지 않습니다.
bash tools/docker.sh build
bash tools/docker.sh doctor
bash tools/docker.sh validate
```

기본 이미지 이름은 `robotwin-phys:cuda12.1`입니다. build는 digest가 고정된 Ubuntu 22.04 / CUDA 12.1.1 devel 이미지에 Python 3.10 venv를 만들고, [requirements.lock](../docker/requirements.lock)의 154개 패키지 버전을 설치합니다. 기존 6개 source repository의 revision을 확인하고 CuRobo를 로컬 GPU의 compute capability에 맞춰 컴파일합니다. 현재 RTX 2080 Ti의 target은 `7.5`입니다. GPU 없는 build 머신에서는 `7.5`를 사용하며 다른 target은 `ROBOTWIN_CUDA_ARCH`로 지정합니다. CUDA 12.1이 지원하지 않는 신형 GPU는 별도의 CUDA/Torch stack 검증이 필요합니다. 호스트 드라이버 설치·Docker 그룹 변경·daemon 재시작은 수행하지 않습니다.

`ROBOTWIN_GPU_MODE=auto`가 기본입니다. NVIDIA Container Toolkit이 Docker에 등록되어 있으면 공식 GPU reservation 경로를 사용합니다. 현재 서버에는 Toolkit이 없으므로 선택한 `/dev/nvidiaN`·공통 제어 장치와 기존 CUDA/NVML/Vulkan/EGL 드라이버 파일만 컨테이너에 연결하는 `bind` 경로를 사용합니다. 드라이버 파일은 읽기 전용이며 host 전체 `/usr/lib`를 mount하지 않습니다. 이 fallback은 현재 Linux x86_64의 라이브러리 경로를 대상으로 하며 MIG·다른 배포판을 일반적으로 지원한다는 뜻은 아닙니다. 다른 서버에서는 표준 NVIDIA Container Toolkit을 권장합니다. `ROBOTWIN_GPU_MODE=nvidia` 또는 `bind`로 명시할 수 있습니다.

`doctor`는 컨테이너의 CUDA 연산, 실제 Vulkan camera render와 의존성을 확인하여 `reports/docker-hardware.json`에 저장합니다. `validate`는 다음을 새로 실행합니다.

1. 버전·repository revision·CuRobo CUDA extension 로딩 검사.
2. CUDA/Vulkan 진단 및 기존 23개 테스트.
3. 기존 `verified-smoke-stack-blocks.json`의 두 episode를 새 프로세스에서 replay하고 초기 상태를 비교, MP4 저장.

로그는 `reports/docker-validation/<UTC timestamp>/`, replay와 영상은 `runs/docker-replay/<UTC timestamp>/`에 저장합니다. 과거 결과를 resume해서 Docker 검증을 통과한 것처럼 처리하지 않습니다. 검증이 실패하면 해당 log에서 원인을 확인하고 환경을 맞춰야 합니다. Docker 이전을 이유로 manifest의 fingerprint·seed·초기 상태 검사를 우회하지 않습니다.

## GPU 선택과 실시간 영상

```bash
# 단일 컨테이너는 여러 GPU 중 두 번째, 한 대뿐이면 첫 번째를 자동 선택.
# 원하는 GPU를 직접 지정할 수도 있습니다:
ROBOTWIN_GPU=0 bash tools/docker.sh doctor

# 고정 manifest를 모든 policy가 공유합니다.
bash tools/docker.sh run robotwin-phys evaluate \
  --manifest manifests/physical_random/stack_blocks_two.json \
  --output runs/fastwam/stack_blocks_two \
  --host 127.0.0.1 --policy-port 9000 --live --video
```

두 번째 명령은 100개 episode manifest와 해당 policy 서버가 준비된 후 사용하는 예시입니다. 실제 모델 서버는 simulator 이미지에 포함되지 않습니다. 학습 모델의 추론 환경과 checkpoint 준비는 [policy 안내](policies.md)를 따릅니다.

Docker에는 선택한 GPU 한 개만 전달하며 simulator는 내부 `cuda:0`을 사용합니다. Toolkit 경로의 `CUDA_VISIBLE_DEVICES`는 `0`, bind 경로는 선택한 물리 GPU의 UUID로 고정합니다. 호스트 GPU 번호를 컨테이너의 `CUDA_VISIBLE_DEVICES`에 다시 넣지 마세요. Toolkit 경로의 driver capability는 `compute,utility,graphics`로 설정하여 CUDA와 Vulkan을 함께 지원합니다. [NVIDIA 공식 설명](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html), [Compose GPU 선택](https://docs.docker.com/compose/how-tos/gpu-support/).

## 환경에 맞게 GPU 수와 장치 선택

한 simulator episode는 한 GPU에서 실행됩니다. 여러 GPU에서는 장치마다 독립 컨테이너에 서로 다른 task를 배정하고, 한 GPU에서는 같은 목록을 순서대로 처리합니다. 같은 task의 pool은 하나만 만들고 모든 policy가 재사용합니다. GPU 수를 바꿔도 seed와 sampling 설정은 바뀌지 않습니다. GPU별 live 포트는 8765부터 순서대로 배정하며 reports·cache도 분리합니다. GPU 0이 desktop과 메모리를 공유하는 경우 실제 여유 메모리는 `nvidia-smi`로 확인하세요.

```bash
# 자동 감지: 한 대면 한 worker, 여러 대면 장치당 한 worker
python3 tools/docker_parallel.py validate --gpus auto

# 한 GPU만 선택 (GPU 한 대인 서버에서도 동일)
python3 tools/docker_parallel.py freeze --gpus 0 --live --video

# 두 GPU 각각에서 기존 manifest의 동일 초기 상태를 재현
python3 tools/docker_parallel.py validate --gpus 0 1

# 50 task를 두 GPU에 나누어 task당 100개의 expert 성공 episode 확정
python3 tools/docker_parallel.py freeze --gpus 0 1 --live --video

# 지정 task만 생성하는 소규모 확인; benchmark 점수로 사용하지 않음
python3 tools/docker_parallel.py freeze --gpus 0 1 \
  --tasks stack_blocks_two grab_roller --episodes 1 --max-candidates 20 \
  --output manifests/docker-dual-smoke --live --video
```

`auto`는 `nvidia-smi`에 보이는 모든 GPU를 뜻합니다. 특정 장치만 사용할 때는 `--gpus 0,1` 또는 `--gpus GPU-...`로 지정합니다. 없는 GPU나 index/UUID로 중복 지정한 GPU는 실행 전에 거부합니다. `--base-port 8865`로 live 포트 시작값을 바꾸고 `ROBOTWIN_CPU_THREADS=2`, `ROBOTWIN_BUILD_JOBS=2`로 CPU 사용량을 조절할 수 있습니다. `ROBOTWIN_IMAGE=my-simulator:tag`로 이미지 이름을 지정하고, `ROBOTWIN_CUDA_ARCH='8.0;8.6' bash tools/docker.sh build`로 다른 GPU용 target을 선택합니다. 이 설정은 여러 GPU의 VRAM을 합치는 모델 병렬화를 제공하지 않습니다.

이 launcher는 호스트 Python 표준 라이브러리만 사용하고, simulator 코드·패키지·검증은 컨테이너에서 실행합니다. task 실패를 성공률에서 제외하지 않으며 오류가 나면 추가 task 배정을 중단합니다. 재실행 시 이미 완료된 manifest는 컨테이너에서 hash·환경·task·seed·episode 수가 일치하는지 검사하고 재사용합니다. 부분 진행만 있는 task는 `--resume`으로 이어갑니다.

추후 두 policy 평가를 동시에 실행할 때는 서버 포트도 9000·9001처럼 분리해야 합니다. 하나의 stateful policy 서버를 두 평가가 동시에 공유하면 모델 action queue/reset이 섞일 수 있습니다. 서로 다른 task manifest와 결과 경로, live 포트로 두 `tools/docker.sh run ... evaluate` 명령을 실행하고 각각 `ROBOTWIN_GPU=0`, `ROBOTWIN_GPU=1` 및 별도 `ROBOTWIN_REPORTS_DIR`, `ROBOTWIN_CACHE_DIR`를 지정하세요.

기존 live server가 localhost에 바인딩되므로 Linux `network_mode: host`를 사용합니다. 컨테이너가 실행 중일 때 서버의 `http://127.0.0.1:8765`로 접근합니다. 별도 포트 매핑은 없습니다. 원격 PC에서는 다음처럼 연결합니다.

```bash
ssh -N -L 8765:127.0.0.1:8765 user@simulator-server
```

그 후 PC 브라우저에서 `http://127.0.0.1:8765`를 엽니다. 기존 localhost policy 서버와 SSH tunnel도 동일하게 사용할 수 있습니다. 동일한 live/policy 포트를 쓰는 평가를 동시에 실행하지 마세요. [Docker host networking 설명](https://docs.docker.com/compose/how-tos/networking/).

두 GPU의 live 영상을 함께 볼 때는 `ssh -N -L 8765:127.0.0.1:8765 -L 8766:127.0.0.1:8766 user@simulator-server`로 연결하고 두 브라우저 탭에서 각각의 포트를 엽니다.

기본 구성은 headless입니다. native `--viewer`에는 추가 X11/Wayland 연결이 필요하며 기본 Compose에 이를 연결하지 않았습니다. 브라우저 live 영상과 MP4에는 desktop DISPLAY가 필요 없습니다.

## 파일과 환경 분리

| 항목 | 처리 |
|---|---|
| CUDA·Python·SAPIEN·CuRobo | 이미지 안에서 설치·빌드 |
| 평가 source·config·공식 adapter source | 이미지에 복사, 수정 후 이미지 재빌드 |
| `vendor/RoboTwin/assets` | 기존 호스트 assets 읽기 전용 mount |
| `checkpoints` | 읽기 전용 mount, 모델/가중치는 이미지에 포함하지 않음 |
| `manifests`, `runs`, `reports` | host 폴더에 지속 저장 |
| runtime cache | workspace의 `.container/cache`, 기존 홈 캐시와 분리 |
| 호스트 `.env`, downloads ZIP, credentials | build context에서 제외 |

wrapper는 호출자의 UID/GID로 실행하여 결과 파일이 root 소유로 생성되지 않게 합니다. privileged 모드와 Docker socket mount는 사용하지 않습니다. 컨테이너 내부 작업 경로는 `/home/guest/robotwin-workspace`로 고정합니다. 이는 현재 assets의 CuRobo YAML 안에 있는 절대 경로와 기존 manifest를 유지하기 위한 선택입니다. 전체 workspace를 mount하면 이미지 안의 `.env`를 호스트 `.env`로 가리므로 그렇게 실행하지 마세요. GPU별 bind 설정은 `.container/compose.gpu-bind-<GPU>.json`에 원자적으로 생성하여 동시에 실행해도 다른 GPU의 설정으로 덮어쓰지 않습니다.

처음 build에는 기존 `vendor/`의 6개 source checkout이 필요합니다. assets 약 16GB는 이미지와 build context에서 제외됩니다. base image digest와 Python 패키지 버전은 고정하지만 Ubuntu apt repository 상태까지 snapshot한 bitwise 동일 build는 아닙니다. 다른 base image는 `CUDA_IMAGE=nvidia/cuda@sha256:...`으로 지정할 수 있으며 기존 simulator와의 호환성을 다시 검증해야 합니다.

```bash
# 실제 실행될 설정 확인; .env/ 이름 충돌을 피하도록 wrapper 사용
bash tools/docker.sh config

# 컨테이너 shell
bash tools/docker.sh shell

# 코드 변경 후 rebuild; 기존 evaluation source fingerprint는 그대로 검사됨
bash tools/docker.sh build

# 생성된 이미지를 다른 서버로 옮길 때 Docker archive 저장
bash tools/docker.sh save
# 출력: .container/robotwin-phys-cuda12.1.tar.gz
# 다른 서버: docker image load -i robotwin-phys-cuda12.1.tar.gz
```

archive에는 assets·checkpoint·평가 결과가 포함되지 않으므로 필요한 bind-mount 디렉터리도 별도로 옮겨야 합니다. GPU 아키텍처가 다른 머신에서는 적절한 `ROBOTWIN_CUDA_ARCH`로 재빌드해야 합니다. GPU/드라이버/패키지가 기존 manifest와 달라지면 새로운 검증 manifest가 필요할 수 있습니다. 호스트 `.env`와 기존 캐시는 Docker 검증 완료 전까지 보존합니다.

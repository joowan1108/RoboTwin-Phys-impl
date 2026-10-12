# FastWAM 먼저 평가: RTX 5060 두 장 + RTX PRO 6000 한 장

**GPU를 RTX 5090 두 장으로 바꾼 최신 구성은 [5090 전용 실행 안내](parallel-evaluation-5090.md)를 사용하세요.** 아래는 5060 구성의 기록입니다.

동일 서버의 세 GPU로 **50 tasks × Physical Random × task당 100 episodes = 5,000회**를 평가합니다. 기본 모델은 `yuanty/fastwam`의 RoboTwin checkpoint `robotwin_uncond_3cam_384.pt`입니다. 기존에 고정한 checkpoint revision과 공식 추론 설정을 유지합니다. 다른 네 모델은 자동으로 실행하지 않습니다.

## 실행 구조

- 각 5060에 시뮬레이터 worker 하나, PRO 6000에 **같은 FastWAM checkpoint의 독립 추론 프로세스 두 개**를 배정합니다. 다른 두 모델을 실행하는 것이 아닙니다.
- 먼저 끝난 worker가 다음 episode를 가져갑니다. 모델 정밀도, inference steps, action chunk, horizon, 성공 조건은 변경하지 않으며 batch inference도 추가하지 않습니다.
- 모델별로 task의 `episode_index=0`만 녹화합니다. FastWAM은 총 50개 MP4입니다. 성공 여부나 완료 순서로 녹화 대상을 고르지 않습니다. Expert 영상은 기본으로 저장하지 않습니다.
- GPU 번호를 0/1/2로 가정하지 않습니다. 할당된 GPU 이름으로 두 5060과 PRO 6000을 구분합니다. 두 시뮬레이터의 runtime provenance가 같아야 합니다.
- 별도 소규모 평가 없이 전체 범위로 진행합니다. CUDA/Vulkan/CuRobo 실행 점검은 수행합니다.

두 FastWAM replica의 최대 VRAM과 실제 처리량은 이 하드웨어에서 아직 측정하지 않았습니다. OOM·통신 오류·worker 오류는 중단 사유이며, task 실패 점수로 바꾸거나 다른 seed로 교체하지 않습니다. `--replicas 1`은 명시적인 메모리 절약 옵션입니다. 이 경우 policy 평가 worker도 하나가 되어 평가 단계에서는 5060 한 장만 사용합니다.

## 빌드: guest@bs1:~/robotwin-workspace

사용자가 말한 빌드 위치는 `guest@bs1`입니다. 자기 자신에게 SCP할 필요가 없습니다. 이미 만든 여섯 simulator/policy 기반 이미지로 **bundle만 새로 빌드**합니다.

```bash
cd /home/guest/robotwin-workspace
export HUB_REPO=YOUR_DOCKERHUB_ID/robotwin-phys
bash tools/build_bundle.sh "$HUB_REPO" v3
docker login
docker push "$HUB_REPO:bundle-cu128-v3"
```

입력 이미지가 다른 repository/tag에 있다면 `build_bundle.sh`의 세 번째·네 번째 인자로 지정합니다. 같은 비교 실험의 나머지 모델도 이 이미지와 assets를 유지하세요. 이미지가 바뀌면 runtime 또는 policy identity 검증이 실패할 수 있습니다.

## CSI OOD에서 제출

[`cluster/job-5060x2-pro6000.sbatch`](../cluster/job-5060x2-pro6000.sbatch)를 Job Composer에 사용합니다. 사용자가 수정 금지한 cache 설정과 마지막 실행 영역은 원본 그대로이며 `apptainer exec`는 한 번입니다.

상단 주요 설정:

```bash
#SBATCH --gres=gpu:rtx5060:2,gpu:pro6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00

IMAGE="docker://YOUR_DOCKERHUB_ID/robotwin-phys:bundle-cu128-v3"
CMD="bash /home/guest/robotwin-workspace/cluster/run_parallel.sh"
DATA="$HOME/robotwin-data"
export APPTAINERENV_ROBOTWIN_EXPERIMENT="phys-5060-100-v1"
export APPTAINERENV_ROBOTWIN_MODEL="fastwam"
export APPTAINERENV_ROBOTWIN_PHASE="all"
export APPTAINERENV_ROBOTWIN_EPISODES=100
export APPTAINERENV_ROBOTWIN_VIDEO_SELECTION="first"
```

`rtx5060`은 예시 GPU type 이름입니다. 기존 CSI 문서에는 5060의 GRES 이름이 없으므로 OOD에 등록된 실제 이름에 맞춰야 합니다. 세 GPU가 같은 노드에 할당되어야 합니다. CPU 16개·RAM 128G는 템플릿 요청값이며 실측 최소 요구량이 아닙니다.

`all`은 assets 준비 → 하드웨어 점검 → 공통 pool 생성 또는 검증·재사용 → 선택한 모델만 다운로드/평가 → 표 생성 순서입니다. 3일 제한으로 종료되면 **같은 DATA·EXPERIMENT·모델·설정으로 다시 제출**합니다. 완료된 expert task와 검증된 평가 결과를 재사용하고 미완료 episode를 이어서 실행합니다. 자동 재제출은 하지 않습니다.

## 다음 모델은 같은 pool을 불러오기

동일 sbatch에서 다음 두 줄만 바꿉니다. `EXPERIMENT`, `DATA`, episodes, seed, 이미지·assets는 유지합니다.

```bash
export APPTAINERENV_ROBOTWIN_MODEL="pi05"
export APPTAINERENV_ROBOTWIN_PHASE="evaluate"
```

`evaluate`는 고정된 pool index가 없거나 파일·조건·초기 상태·runtime이 바뀌었으면 중단합니다. 새로운 seed를 샘플링하거나 expert 검증으로 pool을 다시 만들지 않습니다. 저장한 조건으로 환경을 재생성하고 첫 행동 전에 초기 상태를 비교합니다. 실행 중인 simulator 메모리 전체를 파일에서 복구하는 기능은 아닙니다.

모델 선택값은 `fastwam`, `pi05`, `motus`, `fact`, `g05`입니다. G05는 Hugging Face 접근 승인이 필요합니다. FACT는 native server port가 현재 policy identity에 포함되어 기본 replica 하나로 실행합니다. FACT도 같은 pool을 사용하지만 평가를 동시에 두 개 실행하지 않습니다.

## 저장 위치와 일치 검증

호스트의 `$DATA/manifests/phys-5060-100-v1/`:

```text
pool-index.json                   # 전체 task manifest의 SHA-256 고정
conditions.csv                    # task·episode·seed·물리 조건 export
physical_random/<task>.json        # expert가 확인한 100개 episode 원본
physical_random/<task>.progress.json
physical_random/<task>-verification/
```

task manifest에는 물리 파라미터, 카메라 조건, instruction, 초기 pose/velocity/qpos 및 simulator에서 확인한 mass/inertia/CoM/material/damping과 provenance가 들어 있습니다. 다음 모델도 이 파일을 그대로 읽습니다. 비교할 모델 다섯 개를 처음 suite에 등록하지만 실제 실행은 선택한 하나뿐입니다.

두 GPU에서 task별 expert 검증을 병행하되 **각 task의 후보 seed는 증가 순서대로 검사**합니다. 먼저 완료된 후보를 임의 채택하지 않습니다. worker/GPU 배정 순서는 sampling 입력이 아닙니다.

결과는 `$DATA/runs/phys-5060-100-v1/<model>/physical_random/<task>/`에 저장됩니다.

- `episode-0000.request.json`: 실제 worker 입력과 policy identity.
- `episode-0000.result.json`: 성공 여부와 검증된 초기 상태 hash.
- `episode-0000.randomization.json`: 공통 manifest와 비교한 감사 기록.
- `episode-0000.mp4`: task마다 고정된 첫 episode 영상.

`$DATA/runs/phys-5060-100-v1/tables/`의 `results.md`, `results.tex`, `tasks.csv`, `summary.csv`에 성공률 표를 만듭니다. Physical Random만 선택한 범위를 표시하며 미실행 모델의 점수는 비워 둡니다. `randomization.csv/jsonl`, `condition_consistency.csv`에서 모델·task·episode별 조건을 비교합니다. FastWAM만 완료된 상태는 `partial`이며 모든 모델이 일치했다고 표시하지 않습니다.

로그·배정 정보는 `$DATA/reports/parallel/phys-5060-100-v1/<model>-<jobid>/`에 저장됩니다. `timing.json`의 처리량과 잔여 시간은 **현재 모델의 실제 wall-clock 완료 속도**에서 계산하며, task 구성에 따라 ETA가 변할 수 있습니다.

## Docker 실행 권한이 있는 세 GPU 서버에서 직접 실행

CSI 클러스터에서는 위 OOD 잡을 사용합니다. 아래는 Docker 권한이 있는 서버 전용입니다. 데이터를 컨테이너 밖에 보존합니다.

```bash
export IMAGE=YOUR_DOCKERHUB_ID/robotwin-phys:bundle-cu128-v3
export DATA="$HOME/robotwin-data"
mkdir -p "$DATA"/{assets,checkpoints,downloads,manifests,runs,reports,cache/home}
docker run --rm -it --gpus all --shm-size=8g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp/robotwin-home \
  -v "$DATA/cache/home:/tmp/robotwin-home" \
  -v "$DATA/assets:/home/guest/robotwin-workspace/vendor/RoboTwin/assets" \
  -v "$DATA/checkpoints:/home/guest/robotwin-workspace/checkpoints" \
  -v "$DATA/downloads:/home/guest/robotwin-workspace/downloads" \
  -v "$DATA/manifests:/home/guest/robotwin-workspace/manifests" \
  -v "$DATA/runs:/home/guest/robotwin-workspace/runs" \
  -v "$DATA/reports:/home/guest/robotwin-workspace/reports" \
  "$IMAGE" bash
```

컨테이너 안에서:

```bash
# FastWAM만 50 tasks × Physical Random × 100 episodes
bash cluster/run_parallel.sh

# 이후 모델: 같은 pool, expert 재검증 없음
bash cluster/run_parallel.sh --model pi05 --phase evaluate
```

세 장보다 많은 GPU가 있으면 `--gpus all` 대신 Docker device 선택으로 해당 세 GPU만 전달합니다. SLURM에서는 할당된 `CUDA_VISIBLE_DEVICES`를 유지합니다.

## 한 모델 시간 계산 예시

FastWAM의 이 하드웨어 실측 속도는 아직 없습니다. episode 평균 60초, expert 후보 평균 60초, 채택률 50%, 두 5060의 expert 처리량 2배를 **가정**하면:

| 실제 policy 처리량 향상 | 첫 모델: expert 준비 포함 | 같은 pool 재사용 시 policy 단계 |
|---|---:|---:|
| 1배 | 약 6.9일 | 약 3.5일 |
| 1.5배 | 약 5.8일 | 약 2.3일 |
| 2배 | 약 5.2일 | 약 1.7일 |

다운로드, 모델 로딩, 잡 대기·재시작, task별 expert 편차는 별도입니다. 다른 모델은 해당 모델의 속도로 다시 계산해야 합니다. replica 두 개가 추론 GPU 한 장을 공유하므로 두 배 향상을 보장하지 않습니다.

참고: [FastWAM 공식 checkpoint](https://huggingface.co/yuanty/fastwam), [RoboTwin-Phys 논문](https://arxiv.org/html/2609.26292v1), [SLURM GRES](https://slurm.schedmd.com/gres.html).

# CSI Lab: 기본 OOD 스크립트를 유지하는 통합 이미지

**현재 선택한 PRO 6000 한 장은 [v5 실행 안내](evaluation-pro6000-single.md)를 사용하세요.** `job-pro6000-single.sbatch`는 시뮬레이션과 추론에 한 GPU를 사용하며, FastWAM만 Physical Random으로 평가합니다. 기존 여러 GPU 옵션은 아래에 보존합니다.

**FastWAM만 먼저 실행하는 5060 두 장 + PRO 6000 한 장 구성은 [별도 안내](parallel-evaluation.md)를 사용합니다.** 해당 경로는 Physical Random 50 tasks × 100 episodes의 공통 pool을 저장하고, 이후 모델도 동일 pool을 읽습니다. 아래의 기존 기본 경로는 다섯 모델·세 조건 전체 실행입니다.

현재 기본 경로는 **이미지 하나 + `apptainer exec` 한 번**입니다. [cluster/job.sbatch](../cluster/job.sbatch)와 [5090 두 장용 스크립트](../cluster/job-rtx5090x2.sbatch)는 제공받은 default script의 `/tmp/apptainer-$(whoami)` 캐시 설정 및 “아래는 수정할 필요 없습니다” 아래 실행부를 그대로 보존합니다. GPU 배치, 데이터 bind, 실행 옵션은 그 위의 사용자 설정 영역에서 지정합니다.

## 이번 변경과 재빌드 여부

**새 통합 이미지 `bundle-cu128-v2`를 한 번 build/push해야 합니다.** 기존 `sim-cu128`와 다섯 `policy-*-cu128` 이미지를 빌드 재료로 재사용하므로 여섯 환경을 새로 설치할 필요는 없습니다. 클러스터는 최종 통합 이미지 하나만 내려받습니다. 이전의 여러 이미지 제어 스크립트는 `cluster/legacy/`에 보관했습니다.

통합 이미지 안에서 시뮬레이터는 기존 Python 3.10 환경을, 정책은 `/opt/robotwin-policies/<model>/venv`의 독립 환경을 사용합니다. π0.5는 Python 3.11, 나머지는 3.10이며 패키지를 한 환경에 합치지 않습니다. 각 정책 코드와 build metadata도 분리합니다. 모두 PyTorch 2.7.1+cu128이며 시뮬레이터 CuRobo는 sm75/sm120으로 빌드되어 있습니다. [PyTorch Blackwell 지원](https://pytorch.org/blog/pytorch-2-7/), [NVIDIA GPU 표](https://developer.nvidia.com/cuda/gpus).

[cluster/run_bundle.sh](../cluster/run_bundle.sh)가 컨테이너 내부에서 각 Python 프로세스를 직접 실행합니다. 중첩 Docker/Apptainer, Docker socket, root 권한은 사용하지 않습니다. 다섯 모델은 한 번에 하나씩 GPU에 로드합니다. assets·checkpoint·token·실험 결과는 이미지에 넣지 않고 지속 저장 경로에 둡니다. 이미지에 CUDA/Python 패키지를 구성하며 호스트 CUDA나 드라이버는 설치하지 않습니다.

## guest@bs1에서 build / push

사용자가 말한 로컬 빌드 위치는 `guest@bs1:~/robotwin-workspace`입니다. 이 폴더에서 실행하며, 자기 자신에게 SCP하거나 업데이트 압축을 다시 풀 필요가 없습니다. 아래는 기존 v2 전체 평가 경로의 빌드 예시입니다. FastWAM 단독 병렬 평가는 [v3 빌드 안내](parallel-evaluation.md)를 사용하세요.

```bash
cd /home/guest/robotwin-workspace
export HUB_REPO=YOUR_DOCKERHUB_ID/robotwin-phys

# 새 통합 이미지 한 개만 build. 기존 6개 이미지 재사용.
bash tools/build_bundle.sh "$HUB_REPO" v2

docker login
docker push "$HUB_REPO:bundle-cu128-v2"
```

bs1의 기본 태그 `robotwin-phys:sim-cu128`, `robotwin-phys:policy-*-cu128`를 사용하는 직접 Docker 명령은 다음과 같습니다. 사용자 Hub 이름으로만 태그된 이미지라면 helper에 마지막 두 인자로 저장소와 태그를 지정합니다.

```bash
docker build --platform linux/amd64 -f docker/Dockerfile.bundle \
  -t "$HUB_REPO:bundle-cu128-v2" .
```

기존 이미지의 저장소 또는 태그가 다르면 helper의 마지막 두 인자를 그 값으로 맞춥니다. 해당 여섯 이미지가 로컬에 있어야 합니다. 이미 다른 머신에서 push했다면 Docker Hub에서 먼저 `docker pull`로 가져옵니다.

```bash
bash tools/build_bundle.sh "$HUB_REPO" v2 "$HUB_REPO" v1
```

완전히 새로운 개발 PC에서 빌드 재료부터 만들 때만 다음을 실행합니다. Docker Desktop은 Linux containers 모드를 사용합니다.

```bash
bash tools/build_cluster.sh robotwin-phys build
bash tools/build_bundle.sh "$HUB_REPO" v2 robotwin-phys build
```

통합 이미지는 여러 Python 환경을 포함하므로 큽니다. 빌드·pull·SIF 변환을 위한 디스크 여유가 필요합니다. 패키지 import 성공과 실제 학습 checkpoint 추론 검증은 별개입니다. 같은 실험 도중 이미지 태그를 덮어쓰지 말고 push가 출력한 digest를 보관하세요. OOD의 IMAGE를 `docker://아이디/저장소@sha256:...`로 지정할 수도 있습니다.

## OOD Job Composer 설정

[cluster/job.sbatch](../cluster/job.sbatch)를 붙여넣고 상단 IMAGE의 Docker Hub 아이디를 수정합니다. 파일 하단의 다음 부분은 원문 그대로입니다.

```bash
export APPTAINER_CACHEDIR="/tmp/apptainer-$(whoami)"
mkdir -p "$APPTAINER_CACHEDIR"

# ... default script의 고정 실행부 ...
apptainer exec --nv "$IMAGE" $CMD
```

상단에서 [Apptainer 공식 환경변수](https://apptainer.org/docs/user/latest/appendix.html) `APPTAINER_BINDPATH`, `APPTAINER_CLEANENV`, `APPTAINER_NO_HOME`, `APPTAINER_NO_MOUNT`를 설정합니다. 따라서 고정된 하단 명령에 추가 flag를 넣을 필요가 없습니다. [bind 환경변수 공식 설명](https://apptainer.org/docs/user/latest/bind_paths_and_mounts.html). SLURM의 `CUDA_VISIBLE_DEVICES`를 그대로 컨테이너에 전달하며, 컨테이너 내부에서 할당된 GPU만 역할별로 선택합니다.

| 상단 설정 | 의미 |
|---|---|
| `IMAGE` | push한 `docker://아이디/robotwin-phys:bundle-cu128-v2` |
| `CMD` | 이미지 내부 `cluster/run_bundle.sh` 실행. 기본값 유지 |
| `DATA` | NAS의 지속 저장 경로. 기본 `$HOME/robotwin-data` |
| `APPTAINERENV_ROBOTWIN_EXPERIMENT` | 기본 `pro6000-bundle-v2`. 모델·결과를 공유할 실험 이름 |
| `APPTAINERENV_ROBOTWIN_GPU_LAYOUT` | PRO 6000 한 장은 `shared`, 5090 두 장은 `split` |
| `APPTAINERENV_ROBOTWIN_PHASE` | 기본 `all`: 전체 자동 실행 |
| `APPTAINERENV_ROBOTWIN_EPISODES` | 기본 100 |
| `APPTAINERENV_ROBOTWIN_VIDEO` | 기본 1: MP4 저장 |

자원 요청 헤더는 CPU 8, RAM 96GB, 3일입니다. RAM은 큰 checkpoint 로딩을 위해 기본 예시의 32GB에서 조정했고, 시간은 제공된 가이드의 최대값을 사용합니다. 이 두 값은 사용자 설정 영역입니다. 전체 평가가 3일 안에 끝난다는 보장은 없습니다. 제한으로 종료되면 **같은 설정으로 재제출**해 이어갑니다. 잡 자체를 자동 재제출하지는 않습니다.

Private Docker Hub 인증과 G05 HF token은 스크립트의 주석을 참고해 OOD Files에서 준비하세요. G05 권한이 있는 token을 `~/.huggingface_token`에 저장하고 해당 export 줄의 주석을 해제합니다. 토큰을 이미지나 실험 YAML에 넣지 않습니다.

이전 다중 이미지 방식과 통합 이미지 방식은 정책 경로/provenance가 다릅니다. 기존 v1 실험에 섞지 않고 기본값 `pro6000-bundle-v2` / `rtx5090x2-bundle-v2`로 실행하세요. 시뮬레이션 규칙과 물리 randomization 범위는 유지됩니다.

## 전체 평가 자동 실행

상단의 기본 설정으로 **한 번 Submit**하면 작은 실험 없이 다음 순서로 실행합니다.

1. simulation assets 다운로드·검증 → CUDA / Vulkan / CuRobo 실행 점검.
2. 50 tasks × 3 protocols × 100 episodes의 suite를 고정하고 expert 검증으로 공통 manifest 150개 생성. 완료된 manifest는 검증 후 재사용하고 미완료 pool은 `.progress.json`에서 재개합니다.
3. **π0.5 → FACT → Motus → Fast-WAM → G0.5** 순서로 각 모델의 checkpoint/backbone을 준비하고, 동일한 manifest로 모델당 15,000 episodes를 평가합니다. 한 모델을 종료한 뒤 다음 모델을 로드합니다.
4. task마다 episode 조건 감사 파일을 기록하고, 모델 종료 및 전체 종료 시 성공률 표와 모델 간 조건 비교 표를 갱신합니다.

완료된 모델은 조건 감사까지 통과하면 GPU 로딩을 건너뛰며, 미완료 모델은 기존 완료 episode를 재사용합니다. 원본 조건/요청/결과 불일치, candidate budget 부족, 로딩 오류, OOM은 오류로 중단합니다. 이를 task 실패 성공률로 바꾸거나 다른 seed로 교체하지 않습니다. 모델이 정상 실행되어 task를 실패하면 실패 episode로 분모에 포함합니다.

개별 단계가 필요할 때는 상단의 `APPTAINERENV_ROBOTWIN_PHASE`를 `prepare`, `doctor`, `init`, `freeze`, `evaluate`, `report` 중 하나로 지정하고, 모델 선택은 `APPTAINERENV_ROBOTWIN_MODEL`로 전달합니다. 선택 실행은 `APPTAINERENV_ROBOTWIN_TASKS`, `APPTAINERENV_ROBOTWIN_PROTOCOLS`에 공백으로 구분한 목록을 지정합니다. 기본값은 모든 모델·task·protocol입니다. 모델 로그는 `$DATA/reports/jobs/<jobid>/<model>-policy.log`에 남습니다.

## 동일 조건과 보고 기준

동일한 **task + protocol**에서는 다섯 모델 모두 하나의 expert-verified manifest를 사용합니다. seed, 실제 초기 상태, 물리 조건, instruction이 고정되며 평가 전에 hash와 simulator provenance를 확인합니다. 구형 GPU/CUDA에서 만든 manifest는 새로운 PRO 6000 실험에 재사용하지 않습니다.

환경 생성의 입력은 **task + episode seed + randomization 설정**입니다. 모델 이름·모델 실행 순서·정책 RNG는 환경 조건을 결정하지 않습니다. 매 episode를 새 프로세스에서 생성하며 정책 RNG와 환경 RNG를 분리합니다. 동일한 코드·assets·시뮬레이터/GPU 환경에서 같은 입력은 같은 환경을 재생성해야 합니다. 평가 worker는 첫 policy action 전에 초기 물리값·물체 상태·카메라·조명을 공통 manifest와 대조하고, 감사기는 완료 결과의 초기 상태 hash도 정확히 대조합니다. 정책 행동 이후의 궤적은 모델마다 달라집니다.

세 protocol 간에는 같은 시작 candidate seed 100000을 사용하지만 expert filtering 결과에 따라 채택되는 seed가 달라질 수 있습니다. 따라서 정책 간 공정 비교와 protocol 간 같은-seed paired 비교는 구분합니다. 후자를 수행했다고 주장하지 않습니다.

- `clean`: RoboTwin 공식 `demo_clean`.
- `official_random`: RoboTwin 공식 `demo_randomized`.
- `physical_random`: 공식 시각 randomization에 논문 Table 2·3의 물리 randomization과 task 예외를 추가.

실험은 **독립적인 논문 재현 구현**입니다. Phys 저자들의 공식 평가 코드/config/seed pool은 확인 시점에 공개되지 않았습니다. 원문 Table 1의 Clean/Official 열은 선행 공개 benchmark 수치이고 Phys 열과 평가 출처가 다릅니다. 생성하는 표는 세 열을 모두 이 실험에서 다시 측정한 결과이며 논문 표 숫자를 섞어 넣지 않습니다. 범위와 재현 차이는 [reproduction.md](reproduction.md)를 참고하세요. [논문 원문](https://arxiv.org/html/2609.26292v1), [공식 benchmark 저장소](https://github.com/yefeng00/RoboTwin-Phys).

## 결과 표와 영상

OOD Files에서 `$DATA/runs/<EXPERIMENT>/tables/`를 엽니다.

- `results.md`: 모델별 Clean / Official Randomization / Phys Randomization 성공률(%).
- `results.tex`: 논문에 넣을 수 있는 LaTeX 표.
- `summary.csv`: 모델·조건별 완료 task 수와 완료 episode 수.
- `tasks.csv`: task별 성공 횟수, 성공률, manifest hash, 누락/오류 상태.
- `results.json`: 집계 원본, suite hash, full/subset 범위 표시.
- `randomization.csv`: **모델·protocol·task·episode·seed별** 실제 요청의 질량 배율, 마찰, 반발, CoM, geometry, damping, table, camera, 외력 계수/방향 및 hash. 전체 조건은 `condition_json` 열에도 보존합니다.
- `randomization.jsonl`: 위 episode별 조건 전체와 검증된 초기 물리값을 포함하는 JSON 레코드.
- `condition_consistency.csv`: 같은 task/protocol/episode에 대해 다섯 모델의 요청·완료 증거가 같은 manifest와 일치하는지 비교. 모두 완료·일치한 경우만 `all_models_match`, 미완료는 `pending`/`partial`, 불일치는 `mismatch`입니다.
- `randomization_audit.json`: 일치·미완료·불일치 episode 수와 오류가 있는 모델/task 목록.

각 평가 폴더에도 `<model>/<protocol>/<task>/episode-0000.randomization.json`을 저장합니다. 원본 `episode-0000.request.json`에는 실제 worker 입력이, `episode-0000.result.json`에는 관측한 초기 상태 hash와 완료 결과가 남습니다. 감사 파일은 두 원본을 공통 manifest와 대조한 결과입니다. 완료된 경우 `verified_initial_physical`에 simulator에서 읽은 물체별 질량(kg), 관성, CoM, 접촉 재질, 물체/로봇 joint damping을 보존합니다. 외력은 정책 단계에 적용하는 `force_mg × body mass × 9.81 N` 규칙과 방향을 기록합니다. 완료되지 않은 episode를 실제 적용 확인 완료로 표시하지 않습니다. 이 감사에서 불일치한 task는 성공률 표에서도 유효 결과로 집계하지 않습니다.

각 열은 task 성공률의 산술평균입니다. 모든 task가 100 episodes일 때 전체 episode 평균과 같습니다. 조건의 모든 task가 끝나야 집계 성공률을 표시합니다. 아직 끝나지 않은 조건은 `— (완료/예정 episodes)`로 표시하며 0%로 간주하지 않습니다. 추론 설정이나 checkpoint가 섞인 결과는 집계에서 거부합니다. 한 seed pool만 실행했으므로 여러 독립 실험의 표준편차를 만들어 표시하지 않습니다.

MP4는 `$DATA/runs/<EXPERIMENT>/<model>/<protocol>/<task>/episode-0000.mp4` 등에 저장됩니다. expert 영상은 `$DATA/manifests/<EXPERIMENT>/<protocol>/<task>-verification/`입니다. OOD Files에서 다운로드해 로컬 플레이어로 볼 수 있습니다. 제공된 CSI 가이드에는 compute-node 포트 프록시나 SSH shell이 없으므로 이 템플릿은 **MP4 확인 경로**를 기본으로 사용합니다. 기존 bs1의 SSH live-view 방법을 CSI에 그대로 적용하지 마세요. 관리자가 OOD reverse proxy를 제공한다면 기존 `--live` 서버를 추가 연결할 수 있습니다.

## 정책별 구현 메모

- G0.5: gated RoboTwin checkpoint와 processor/action tokenizer가 필요합니다. 실제 gated config/weight를 읽기 전에는 로드 성공을 보장할 수 없습니다.
- π0.5: checkpoint card의 Aloha transform, `adapt_to_pi=True`, joint delta + absolute gripper, horizon 32를 적용합니다. Torch 추론이며 JAX는 CPU로 제한해 GPU 메모리 선점을 막습니다. vendored OpenPI PyTorch 코드가 요구하는 Transformers 4.53.2와 공식 `transformers_replace`를 설치합니다.
- Motus: T5 encoder와 Wan VAE 파일도 실제 코드에서 로드하므로 포함합니다. README의 “config/tokenizer only” 문구만 믿고 이 파일을 빼지 않습니다. 공식 FlashAttention 2.8.3.post1 wheel을 SHA-256으로 검증하고 sm120 kernel 포함 여부를 검사합니다.
- FACT: native server와 RPC bridge를 같은 정책 컨테이너에서 실행합니다. native server에 episode에서 파생한 요청 seed를 전달하는 작은 source patch가 이미지에 포함됩니다.
- Fast-WAM: 공식 config의 `redirect_common_files=false`를 사용해 Wan 원본 파일을 불러옵니다. 별도 ModelScope 변환본의 자동 다운로드를 피하고 HF revision으로 고정한 파일을 사용합니다. 로더·정책 가중치는 그대로 유지합니다.

모델 서버 시작 시 checkpoint와 backbone bytes를 lock과 대조합니다. 패키지 목록, 지원 코드/config hashes, artifact locks가 정책 identity에 포함됩니다. 평가 도중 자동으로 HF의 최신 weight를 가져오지 않습니다. 컨테이너별 build/import 검증과 원격 PRO 6000 실추론 검증은 별개이며, 실제 성능 수치는 평가 후에만 생성됩니다.

## RTX 5090 두 장 옵션

[cluster/job-rtx5090x2.sbatch](../cluster/job-rtx5090x2.sbatch)를 OOD에 복사합니다. 같은 Docker 이미지 태그를 쓰므로 별도 이미지 빌드는 필요하지 않습니다. `#SBATCH --gres=gpu:rtx5090:2`를 기본으로 넣었습니다. CSI 안내에는 5090의 SLURM GRES 문자열이 명시되지 않았으므로 OOD GPU 선택 항목의 이름이 `5090` 등이라면 그 부분을 실제 이름으로 맞추세요.

| 옵션 | 시뮬레이터 | 정책 | Motus T5 |
|---|---|---|---|
| PRO 6000 × 1 (`job.sbatch`) | 할당 GPU A | GPU A | GPU A |
| RTX 5090 × 2 (`job-rtx5090x2.sbatch`) | 할당 GPU A | GPU B | GPU A |

두 장의 [5090 메모리는 장당 32GB](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)입니다. 일반적인 정책 분산학습/tensor parallel이나 메모리 64GB 통합을 의미하지 않습니다. π0.5·G0.5·FACT·Fast-WAM은 전체 정책을 GPU B에 올립니다. 모델이 32GB에 안 들어가면 해당 구성은 실패로 중단되며 성공률 0%로 보고하지 않습니다.

Motus는 공식 안내상 T5를 포함하면 약 41GB, 사전 encoding 시 약 24GB를 요구합니다. 두 장 옵션에서는 `APPTAINERENV_ROBOTWIN_MOTUS_T5_DEVICE="cuda:1"`로 정책 프로세스에 GPU B,A 순서로 두 장을 전달하고, T5 encoder만 두 번째 장(A)에 올립니다. T5 결과 텐서를 B로 옮겨 기존 모델에 전달합니다. T5와 시뮬레이터는 A의 VRAM을 공유합니다. 임베딩 계산 방식과 BF16 precision을 유지하되 이 명시적인 배치 변경은 config hash에 기록됩니다. **전체 Motus 실추론 peak VRAM은 실제 5090 잡에서 검증해야 합니다.** 한 장 옵션은 원래 T5 배치를 유지하고 약 41GiB 미만일 때 안내와 함께 중단합니다.

`APPTAINERENV_ROBOTWIN_GPU_LAYOUT="split"`은 SLURM이 실제 할당한 두 장만 사용합니다. 호스트의 물리 GPU 0/1을 가정하지 않습니다. doctor 단계는 양쪽 GPU를 각각 검증합니다. `APPTAINERENV_ROBOTWIN_EXPERIMENT="rtx5090x2-bundle-v2"`로 별도 manifest를 만들고, PRO 6000에서 만든 manifest·성공률과 섞지 마세요. 한 하드웨어에서 다섯 정책의 비교를 완료한 다음 하드웨어별 결과를 별도 표로 비교할 수 있습니다.

FACT의 upstream setup은 Torch 2.6/cu124를 사용하지만 Blackwell 이미지에서는 2.7.1/cu128로 올립니다. G0.5의 다른 benchmark·훈련 전용 dependency는 추론 이미지에서 제외하고 공식 코드의 기본 attention fallback을 허용합니다. 이러한 실행 환경 차이는 package freeze와 코드 hash에 남으며, 저자의 미공개 RoboTwin-Phys 실행 환경과 동일하다고 간주하지 않습니다.

π0.5 이미지의 LeRobot은 OpenPI training 타입 import 때문에 `--no-deps`로 설치합니다. LeRobot의 훈련/teleoperation extras 누락과 `torch<2.7` metadata는 추론 이미지의 의도적인 예외입니다. `/opt/robotwin-policies/pi05/build/pip-check.json`에 그대로 남기고 다른 package 충돌은 이미지 빌드를 실패시킵니다. 이 예외가 실제 checkpoint 추론 검증을 대체하지는 않습니다. FACT 추론 경로는 LeRobot을 import하지 않으므로 해당 훈련 패키지를 설치하지 않습니다.

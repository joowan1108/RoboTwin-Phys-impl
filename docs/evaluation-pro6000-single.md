# PRO 6000 한 장으로 FastWAM 평가

2026-10-10 사용자가 제공한 Slurm 조회 결과에서 `gpu` 파티션에는 `p1` 한 노드와 `gpu:pro6000:4`만 등록되어 있습니다. RTX 5090 요청은 현재 이 파티션에서 충족할 수 없습니다. 이 실행 경로는 **PRO 6000 한 장**을 요청하고, 같은 GPU에서 시뮬레이터 한 개와 FastWAM 추론 프로세스 한 개를 사용합니다.

평가 범위는 **50 tasks × Physical Random × task당 100 episodes = 5,000회**입니다. 전체 공통 episode pool을 expert로 검증·저장한 뒤 FastWAM만 평가합니다. 영상은 task마다 고정된 첫 episode 한 개씩, 총 50개입니다. 물리 조건·seed·초기 상태·instruction을 다른 모델에서도 그대로 불러오며, 매 episode의 요청·결과·randomization audit와 결과 표를 저장합니다.

## bs1에서 bundle 갱신

`v4`의 `run_bundle.sh`에는 이전 단일 GPU 실행 경로가 있지만, `all`은 다섯 모델을 모두 실행하고 `--video`는 모든 episode를 저장합니다. 현재 요구하는 모델 하나·공통 pool index·영상 한 개·재개 검증을 유지하는 `run_parallel.sh`는 `v4`에서 세 GPU를 요구합니다. `v5`는 이 평가기에 단일 GPU 지원을 추가합니다. CUDA, PyTorch, 시뮬레이터나 모델별 환경을 다시 설치하지 않고 기존 여섯 입력 이미지로 bundle만 빌드합니다.

```bash
cd /home/guest/robotwin-workspace
export HUB_REPO=joowan1108/robotwin-phys
bash tools/build_bundle.sh "$HUB_REPO" v5
docker login
docker push "$HUB_REPO:bundle-cu128-v5"
```

## OOD 제출

[`cluster/ood-fastwam-pro6000-single-v5.txt`](../cluster/ood-fastwam-pro6000-single-v5.txt)의 전체 내용을 OOD Job Composer에 붙여넣고 제출하세요. 같은 내용의 원본은 [`cluster/job-pro6000-single.sbatch`](../cluster/job-pro6000-single.sbatch)입니다. 기존 5090용 스크립트에서 GPU 개수만 바꾸는 대신 이 파일을 사용합니다.

주요 설정:

```bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00

IMAGE="docker://joowan1108/robotwin-phys:bundle-cu128-v5"
CMD="bash /home/guest/robotwin-workspace/cluster/run_parallel.sh"
export APPTAINERENV_ROBOTWIN_EXPERIMENT="phys-pro6000-100-v1"
export APPTAINERENV_ROBOTWIN_MODEL="fastwam"
export APPTAINERENV_ROBOTWIN_PHASE="all"
export APPTAINERENV_ROBOTWIN_EPISODES=100
export APPTAINERENV_ROBOTWIN_VIDEO_SELECTION="first"
export APPTAINERENV_ROBOTWIN_POLICY_REPLICAS=1
```

`--nodes=1`은 서버 한 대에 자원을 요청하는 옵션입니다. 특정 노드 이름을 지정하지 않으므로 Slurm이 적합한 노드를 선택합니다. GPU 번호는 Slurm의 `CUDA_VISIBLE_DEVICES`를 그대로 사용합니다. 시뮬레이션과 추론이 같은 할당 GPU를 공유하며, policy replica를 두 개로 늘리는 설정은 거부합니다. GPU 한 장에서의 실행은 처리량과 메모리 여유가 세 장 구성과 다릅니다. 정책의 정밀도나 추론 설정을 자동으로 바꾸지 않습니다.

템플릿의 cache 설정과 보호된 하단 실행 영역을 보존했습니다. Docker Hub 저장소가 private이면 기존 안내대로 토큰을 설정합니다. 클러스터의 실제 할당 가능 시점은 다른 잡의 자원 사용에 따라 달라집니다.

## 재개 및 다음 모델

공통 pool은 `$HOME/robotwin-data/manifests/phys-pro6000-100-v1/`에 저장합니다. 모델별 결과·조건 기록·표는 `$HOME/robotwin-data/runs/phys-pro6000-100-v1/`에 저장합니다. GPU 배치와 처리량은 `$HOME/robotwin-data/reports/parallel/phys-pro6000-100-v1/`에 기록합니다.

3일 제한으로 중단되면 같은 DATA·EXPERIMENT·이미지·설정으로 다시 제출합니다. 완료된 검증 episode를 재사용하며, 중단된 episode는 같은 seed·조건으로 재시도합니다. 3일 안에 전체 평가가 끝난다고 보장하지 않습니다.

다음 모델은 같은 스크립트에서 아래 두 줄을 변경합니다.

```bash
export APPTAINERENV_ROBOTWIN_MODEL="pi05"
export APPTAINERENV_ROBOTWIN_PHASE="evaluate"
```

`evaluate`는 이미 저장된 공통 pool을 검증한 뒤 불러옵니다. pool이 없거나 runtime/조건이 달라지면 중단하며, 다른 모델을 위해 조건을 다시 샘플링하지 않습니다. 모델별 메모리 사용량은 별도 검증 대상입니다. 기존 5090/5060으로 생성한 pool과 이 PRO 6000 실험의 결과를 섞지 않습니다.

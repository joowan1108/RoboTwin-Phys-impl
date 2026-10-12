# RTX 5090 두 장 + PRO 6000 한 장: FastWAM 평가

**2026-10-10 CSI Slurm 조회 결과에서 5090은 `gpu` 파티션에 등록되어 있지 않습니다. 현재 선택한 구성은 [PRO 6000 한 장 · v5](evaluation-pro6000-single.md)입니다.** 아래 경로는 세 GPU가 함께 할당 가능한 환경을 위한 옵션으로 보존합니다.

이 경로는 RTX 5090 두 장을 시뮬레이션에, PRO 6000 한 장을 동일 FastWAM checkpoint의 독립 추론 프로세스 두 개에 사용합니다. 전체 평가 범위는 **50 tasks × Physical Random × task당 100 episodes = 5,000회**입니다. GPU 번호를 가정하지 않고 잡에서 할당받은 GPU의 이름을 확인해 자동 배정합니다.

## bs1에서 build / push

```bash
cd /home/guest/robotwin-workspace
export HUB_REPO=joowan1108/robotwin-phys
bash tools/build_bundle.sh "$HUB_REPO" v4
docker login
docker push "$HUB_REPO:bundle-cu128-v4"
```

`v4`는 5060·5090 자동 선택을 모두 지원합니다. 기존 simulator와 다섯 policy 이미지를 재사용하여 bundle만 빌드합니다. 기존 `v3`를 직접 GPU ID 인자와 함께 실행하는 방법도 가능하지만, 아래 OOD 스크립트는 번호 입력이 필요 없는 `v4`를 사용합니다. 다른 Docker Hub 계정을 쓰면 빌드의 `HUB_REPO`와 스크립트의 `IMAGE`를 함께 바꾸세요.

## OOD

[`cluster/job-rtx5090x2-pro6000.sbatch`](../cluster/job-rtx5090x2-pro6000.sbatch)의 전체 내용을 Job Composer에 붙여넣고 제출합니다. 기존 `job-rtx5090x2.sbatch`는 PRO 6000 없이 GPU 두 장만 사용하는 별도 구성입니다.

이 스크립트는 다음을 요청합니다.

```bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx5090:2,gpu:pro6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
```

`rtx5090`은 클러스터의 실제 GRES 등록명과 일치해야 합니다. 제공된 CSI 안내에는 정확한 5090 type 문자열이 없으므로 OOD의 실제 GPU 이름에 맞추세요. 세 GPU는 같은 노드에서 할당되어야 합니다.

### 잡 제출 시 `Requested node configuration is not available`

노드별 GRES 등록명·GPU 개수·CPU·메모리와 파티션 제한을 먼저 확인합니다. 빌드 호스트 `bs1`에는 Slurm 조회 명령이 없으므로 클러스터에서 조회해야 합니다.

OOD에서 별도의 잡을 만들고 [`cluster/ood-inspect-slurm.txt`](../cluster/ood-inspect-slurm.txt)를 붙여넣어 제출하세요. 이 조회용 잡은 가이드에서 확인된 `gpu:pro6000:1`, CPU 1개, RAM 1G, 최대 5분을 요청합니다. GPU 계산과 이미지 pull은 하지 않고, 호스트의 `sinfo`·`scontrol` 조회 후 바로 종료합니다. 기존 템플릿의 cache 및 하단 실행 영역은 그대로 보존하되, 조회용 잡에서는 그 위에서 종료합니다. Docker 재빌드는 필요하지 않습니다.

완료 후 OOD Files/Job Composer에서 `slurm-diagnose-<잡번호>.out` 전체 내용을 확인합니다. 조회용 잡 자체의 할당 정보와 실제 평가에 필요한 세 GPU 조합은 구분해야 합니다. 조회용 잡의 성공만으로 평가 자원을 할당할 수 있다고 판정하지 않습니다. 이 잡도 제출이 거절되거나 조회 명령 접근이 제한된다면, 관리자에게 [`cluster/inspect-slurm.sh`](../cluster/inspect-slurm.sh)를 실행해 출력을 전달해 달라고 요청하세요. 이 파일은 잡을 제출하지 않는 읽기 전용 조회 스크립트입니다.

조회 항목은 Slurm 공식 [sinfo](https://slurm.schedmd.com/sinfo.html), [scontrol](https://slurm.schedmd.com/scontrol.html) 명령을 사용합니다.

`IMAGE`와 실행 설정:

```bash
IMAGE="docker://joowan1108/robotwin-phys:bundle-cu128-v4"
CMD="bash /home/guest/robotwin-workspace/cluster/run_parallel.sh"
export APPTAINERENV_ROBOTWIN_EXPERIMENT="phys-5090-100-v1"
export APPTAINERENV_ROBOTWIN_MODEL="fastwam"
export APPTAINERENV_ROBOTWIN_PHASE="all"
export APPTAINERENV_ROBOTWIN_EPISODES=100
export APPTAINERENV_ROBOTWIN_VIDEO_SELECTION="first"
```

수정 금지된 cache 설정과 하단 `apptainer exec --nv "$IMAGE" $CMD` 영역은 원본 그대로입니다. 별도 Docker 명령이나 수동 GPU 번호 입력을 잡 안에 추가하지 않습니다.

## 공통 episode 재사용과 재개

최초 실행 시 `$HOME/robotwin-data/manifests/phys-5090-100-v1/`에 공통 seed·물리 조건·초기 상태·instruction과 pool index를 저장합니다. 모델별 조건 기록과 표는 `$HOME/robotwin-data/runs/phys-5090-100-v1/`에 남습니다. 영상은 모델·task마다 고정된 첫 episode 하나씩 저장합니다.

다음 모델은 동일 스크립트에서 다음 값만 바꿉니다.

```bash
export APPTAINERENV_ROBOTWIN_MODEL="pi05"
export APPTAINERENV_ROBOTWIN_PHASE="evaluate"
```

DATA·EXPERIMENT·이미지·assets는 유지합니다. `evaluate`는 공통 pool을 읽으며 재샘플링하지 않습니다. 5060에서 만든 pool은 GPU provenance가 달라 이 5090 실험과 섞지 않습니다.

3일 제한으로 중단되면 같은 모델·설정으로 재제출합니다. 완료된 검증 결과를 재사용합니다. 이후 모델에서도 최대 메모리 사용량은 별도 확인이 필요합니다. 같은 GPU 한 장의 메모리·추론 자원을 두 프로세스가 공유하므로 2배 가속을 보장하지 않습니다.

저장 형식, 오류 처리, 메모리 절약 옵션과 표 해석은 [병렬 평가 상세 안내](parallel-evaluation.md)를 참고하세요. 그 문서의 5060용 experiment 이름 대신 이 문서의 `phys-5090-100-v1`을 사용합니다.

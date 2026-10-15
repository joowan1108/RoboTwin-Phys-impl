# PRO 6000 두 장으로 공통 pool 준비

현재 `phys-pro6000-100-v2`의 pool을 두 GPU로 준비할 수 있습니다. GPU마다 worker 두 개를 실행하는 기본 설정으로 서로 다른 task 네 개를 동시에 처리합니다. task 안에서는 후보 seed를 순서대로 검사하고, expert가 성공한 100개를 확보합니다. 먼저 완료된 seed를 선택하는 방식이 아닙니다.

## OOD에서 전환

1. 실행 중인 기존 FastWAM 잡을 OOD에서 Stop/Cancel합니다. 잡이 종료된 상태인지 확인합니다. 기존 controller는 같은 실험에 대한 독점 잠금을 가지고 있으므로 동시에 새 준비 잡을 실행하지 않습니다.
2. [준비 스크립트](../cluster/ood-pool-pro6000x2-v5.txt)를 새 Job Composer 잡에 붙여넣고 제출합니다. **한 노드에 PRO 6000 두 장**을 요청합니다. 이미 올린 `robotwin-expert-fix-v1.zip`을 그대로 사용합니다. 추가 ZIP 업로드나 이미지 빌드는 필요 없습니다.
3. 이 잡은 **pool 준비만** 하고 종료합니다. 전체 task pool이 완성되면 기존 [수정 FastWAM 스크립트](../cluster/ood-fastwam-pro6000-v5-expert-fix.txt)로 제출합니다. 같은 v2 pool을 읽고 평가를 진행합니다. π0.5도 기존 v2용 스크립트를 사용합니다.

기존 task manifest와 `.progress.json`은 보존합니다. 완료된 task는 건너뛰고, 미완료 task는 마지막으로 저장한 candidate 이후부터 이어갑니다. 검사 도중 중지한 seed는 진행 파일에 반영되지 않았으므로 다시 검사할 수 있습니다. task 완료 직전에 중지했어도 100개가 저장된 progress가 있으면 manifest를 확정합니다. 이미 완료된 모델 평가 결과도 같은 정책 identity를 유지하면 재사용합니다.

이 전환은 `tools/parallel_eval.py`의 스케줄러만 바꿉니다. simulator source·asset·물리 조건·instruction·성공 기준은 바꾸지 않아 기존 v2 source fingerprint와 맞는 progress를 재사용할 수 있습니다. 두 GPU의 driver·GPU 종류·라이브러리 등 provenance가 다르면 섞지 않고 중단합니다. 구형 v1 실험이나 다른 simulator patch로 생성한 pool에는 이 안내를 적용하지 않습니다.

## worker 수 조절

```bash
# 한 노드의 PRO 6000 두 장
#SBATCH --gres=gpu:pro6000:2
# CPU와 호스트 RAM 요청: 성능/최소 요구량 실측값은 아님
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# GPU 한 장마다 task 프로세스 두 개: 합계 네 개
export APPTAINERENV_ROBOTWIN_FREEZE_WORKERS_PER_GPU=2
```

worker 수를 `1`로 바꾸면 두 GPU에서 task 두 개를 동시에 준비합니다. GPU 한 장만 사용할 때도 `--gres=gpu:pro6000:1`, worker 수 `2`로 서로 다른 task 두 개를 병행할 수 있습니다.

프로세스마다 별도의 CUDA context, renderer와 CuRobo planner를 생성합니다. GPU 메모리·CPU·호스트 RAM 사용량이 늘어나며, worker 수 증가가 같은 비율의 속도 증가를 보장하지 않습니다. 두 worker/GPU의 실제 메모리와 처리량은 원격 GPU에서 검증해야 합니다. OOM·파일 누락·worker 오류는 후보 제외로 처리하지 않고 중단합니다. OOM이면 worker 수를 `1`로 줄여 같은 스크립트를 다시 제출하면 저장된 progress에서 이어집니다.

## 로그

전체 task 시작·완료 수는 `slurm-<새 잡번호>.out`에 표시됩니다.

```text
Pool preparation: 4 workers on 2 GPUs; 0/50 tasks already frozen
Freezing adjust_bottle on GPU ...; log=...
Freezing beat_block_hammer on GPU ...; log=...
Frozen adjust_bottle: 100 episodes; 1/50 tasks complete
```

각 task의 seed 채택·제외는 다음 파일을 봅니다.

```text
~/robotwin-data/reports/parallel/phys-pro6000-100-v2/fastwam-<새 잡번호>/freeze-<task>.log
```

각 task 진행 파일은 아래 경로에 있습니다.

```text
~/robotwin-data/manifests/phys-pro6000-100-v2/physical_random/<task>.progress.json
```

pool 준비 영상은 기존처럼 기본으로 저장하지 않습니다. 이 스크립트는 policy를 로드하거나 평가하지 않습니다.

## 검증

관련 CPU 테스트 93개가 통과했습니다. 두 GPU에서 task 네 개의 동시 시작, 완료된 manifest 보존, 중단 후 저장된 candidate부터 resume, 준비 중 policy 로드 금지, OOD 실행 파일 생성과 bind, 기존 simulator snapshot과의 동일성을 확인했습니다. shell 문법과 보호된 실행부도 확인했습니다. 실제 GPU 메모리와 처리량 검증은 원격 잡 실행으로 확인해야 합니다.

# RoboTwin-Phys 평가 기반

이 저장소는 **arXiv:2609.26292v1에 근거한 독립 재구현**입니다. 2026-10-08 확인 시 RoboTwin-Phys 공식 저장소에는 실행 코드·task 설정·평가 seed 목록이 공개되지 않았습니다. 따라서 저자 코드와 수치가 완전히 같은 공식 구현으로 표시하면 안 됩니다. 논문 수치 범위, 9개 task 예외, expert filtering, policy 간 동일 조건 재사용을 구현했습니다. [근거와 구현 가정](docs/reproduction.md)을 먼저 확인하세요.

현재 머신에는 Python 3.10, PyTorch 2.4.1+cu121, CUDA Toolkit 12.1, SAPIEN 3.0.0b1, MPLib 0.2.1, CuRobo v0.7.8, ffmpeg와 공식 assets가 설치되어 있습니다. NVIDIA 드라이버를 변경하지 않습니다. GPU 1을 기본으로 사용하며 `CUDA_VISIBLE_DEVICES=0`으로 바꿀 수 있습니다. 자세한 실제 점검값은 [reports/hardware.json](reports/hardware.json)에 있습니다.

## 실행

**v6 평가 단계 수정 (잡 1681 이후 사용):** [세 task의 check_success 속성 수정과 같은 pool 계속 실행](docs/success-context-fix.md), [ZIP](reports/robotwin-success-context-fix-v4.zip), [OOD 스크립트](cluster/ood-fastwam-pro6000x2-v6-context.txt). pool과 저장된 progress를 보존하고 FastWAM 평가로 이어간다.

**현재 권장 경로 (v6 새 pool):** [CoM offset 적용 범위 수정과 v6 pool 안내](docs/com-scope-fresh-pool.md), [새 ZIP](reports/robotwin-fresh-pool-v2.zip), [pool 전용 OOD 스크립트](cluster/ood-pool-pro6000-v6-fresh.txt), [pool + FastWAM 평가 스크립트 (PRO 6000 2장)](cluster/ood-fastwam-pro6000x2-v6-all.txt). 공식 clutter에는 물체 속성(CoM·질량·크기)과 외력을 적용하지 않도록 simulator를 수정했으므로 v5 pool을 이어받지 않고 `phys-pro6000-100-v6`을 새로 준비한다. v1–v3 expert 수정을 모두 포함한다. `put_object_cabinet`의 낮은 채택률은 [분석 기록](docs/put-object-cabinet-feasibility.md)을 참고한다.

**현재 v5 pool의 `086_woodenblock`·`021_cup` 집기점 오류와 `put_bottles_dustbin` planner 실패 뒤 `IndexError` 후 재개:** [원인과 재개 안내 (v3)](docs/planner-failure-recovery.md), [새 패치 ZIP](reports/robotwin-planner-failure-fix-v3.zip), [OOD 재개 스크립트](cluster/ood-pool-pro6000-v5-planfail.txt). v3는 [집기점 패치 v2](docs/grasp-contact-recovery.md)를 포함한다. 완료한 task와 저장된 seed를 보존하며 전체 50개 task의 pool 준비를 이어간다.

**현재 선택한 PRO 6000 한 장:** [FastWAM 단독 평가 및 v5 이미지 빌드 안내](docs/evaluation-pro6000-single.md), [OOD 스크립트 (.txt)](cluster/ood-fastwam-pro6000-single-v5.txt). 시뮬레이션과 추론이 한 GPU를 공유하며, 50 tasks × Physical Random × 100 episodes를 평가합니다. 공통 episode pool을 다음 모델에서도 그대로 재사용합니다.

**별도 GPU 구성이 가능한 경우:** [RTX 5090 두 장 + PRO 6000 한 장](docs/parallel-evaluation-5090.md), [5060 두 장 + PRO 6000 한 장](docs/parallel-evaluation.md). 현재 제공받은 CSI Slurm 조회 결과에는 PRO 6000만 등록되어 있습니다.

이후 실행은 **Docker 환경을 사용합니다**. [Docker 빌드·GPU·렌더링 안내](docs/docker.md)를 확인하세요. Docker 접근 권한이 있는 서버 터미널에서:

```bash
cd /home/guest/robotwin-workspace
bash tools/docker.sh build
bash tools/docker.sh doctor
bash tools/docker.sh validate
```

아래의 `robotwin-phys ...` 명령은 `bash tools/docker.sh run robotwin-phys ...`로 실행하거나 `bash tools/docker.sh shell` 안에서 실행합니다. 이미지에 CUDA/Python 의존성을 다시 설치하며 기존 assets·manifest·영상은 재사용합니다. `python3 tools/docker_parallel.py validate --gpus auto`로 사용 가능한 GPU를 자동 감지하여 검증합니다. `freeze --gpus 0`으로 한 대만 쓰거나 `freeze --gpus 0 1`로 task를 나누어 처리할 수 있습니다. [GPU별 포트·로그와 실행 방식](docs/docker.md)을 확인하세요.

아래는 이전에 설치·검증한 **호스트 환경의 기록**입니다. Docker 사용 시 호스트 환경을 활성화하거나 bootstrap을 다시 실행할 필요가 없습니다.

```bash
cd /home/guest/robotwin-workspace
source tools/activate.sh
python tools/doctor.py
robotwin-phys tasks
```

`tools/bootstrap.sh`는 이전 호스트 설치를 기록한 스크립트입니다. 앞으로의 의존성 설치는 Dockerfile로 진행합니다. asset 다운로드는 약 15GB, 압축 해제 후 assets는 약 16GB입니다.

## 1. 모든 policy가 공유할 episode 목록 확정

```bash
# task당 expert가 성공한 100개 조건을 확정. 기존 manifest는 덮어쓰지 않음.
robotwin-phys freeze \
  --task stack_blocks_two --protocol physical_random \
  --episodes 100 --seed 100000 \
  --output manifests/physical_random/stack_blocks_two.json

# 중간에 중단한 expert 검증 재개
robotwin-phys freeze \
  --task stack_blocks_two --protocol physical_random \
  --episodes 100 --seed 100000 --resume \
  --output manifests/physical_random/stack_blocks_two.json

# 공식 50-task registry 전체. 상당한 시간이 드는 실제 expert rollout 작업.
bash tools/freeze_suite.sh physical_random
```

`freeze`는 매 candidate마다 별도 프로세스를 사용하고 expert 실패/불안정 episode만 제외합니다. dependency, asset, 코드 오류를 실패 episode로 조용히 제외하지 않고 중단합니다. seed는 100000부터 순차 검사합니다. 이는 RoboTwin의 `100000 * (1 + run_seed)`에서 `run_seed=0`인 시작값입니다. RoboTwin-Phys 저자가 사용한 실제 seed 목록은 미공개입니다.

집기 후보를 찾지 못한 경우의 공식 실행 루프와 재현 worker 차이를 확인하고 수정했습니다. 유효한 접촉점에 대한 집기 탐색이 모두 실패하면 `expert_failed`로 기록하며, asset 누락 등은 중단합니다. 기존 v5 이미지의 OOD 패치와 seed 비교 방법은 [expert 실패 처리 대조](docs/expert-planning-review.md)를 확인하세요.

잡 1628의 placement NaN 오류는 공통 rotation helper를 수정했습니다. 이제 task 하나의 실행 오류는 다른 task 준비를 취소하지 않습니다. 이전 v2→v3 단계의 기록은 [공통 수학 오류 수정 안내](docs/geometry-failure-review.md)에 있습니다.

잡 1640의 CuRobo batch shape mismatch는 IK/result 마스크를 분리해 수정했습니다. 이전 v4 단계는 [CuRobo 수정 기록](docs/curobo-mask-review.md), 이후 v5의 결과 형식 수정은 [v5 pool 안내](docs/fresh-pool.md)에 있습니다. 현재 재개는 위의 v5 재개 스크립트를 사용합니다.

manifest에는 실제 채택된 seed, 물리 파라미터, instruction, 초기 pose/velocity/qpos, camera pose, 적용된 mass/inertia/CoM/material/damping, 코드·asset·환경 fingerprint가 저장됩니다. **동일 task·protocol·ablation을 비교할 때는 같은 manifest 파일을 그대로 재사용하세요.** seed만 같고 config 또는 assets가 다르면 같은 조건이 아닙니다. 서로 다른 protocol은 의도적으로 조건이 달라지고 expert 통과 목록도 다를 수 있습니다.

`candidates`는 simulator 실행 없이 조건을 미리 보는 명령입니다. 이 결과는 미검증 상태이므로 평가기가 거부합니다.

## 2. policy 서버 연결과 평가

클러스터에서는 [전체 자동 평가 안내](docs/cluster.md)의 `PHASE="all"`로 다섯 모델 평가를 실행합니다. 아래는 서버를 직접 연결하는 개별 실행 방법입니다. 다섯 checkpoint와 공식 adapter 연결 설정은 [configs/policies](configs/policies), 모델별 준비 절차는 [docs/policies.md](docs/policies.md)에 있습니다. 모델 환경과 simulator 환경을 분리해 Python/Torch/SAPIEN 버전 충돌을 피합니다.

```bash
# 모델을 설치한 별도 환경의 Python으로 실행. simulator activate.sh를 쓰지 않음.
export PYTHONPATH=/home/guest/robotwin-workspace/src
/path/to/model-env/bin/python -m robotwin_phys.policy_server \
  --config /home/guest/robotwin-workspace/configs/policies/fastwam.yaml --port 9000
```

다른 터미널에서 simulator 컨테이너를 실행합니다:

```bash
cd /home/guest/robotwin-workspace
bash tools/docker.sh run robotwin-phys evaluate \
  --manifest manifests/physical_random/stack_blocks_two.json \
  --output runs/fastwam/stack_blocks_two \
  --host 127.0.0.1 --policy-port 9000 --live --video
```

모든 policy 평가에서 이 manifest를 사용하고 `--output`만 모델별로 바꿉니다. 초기 장면이 manifest와 다르면 평가를 중단합니다. policy 실패는 분모에 남으며 다른 seed로 교체되지 않습니다. 실행 중 오류는 task 성공률로 처리하지 않습니다. 같은 설정으로 `--resume`을 붙이면 완료된 episode는 재사용합니다. 100개를 모두 완료해야 benchmark 규모의 결과로 표시됩니다.

## 3. 실시간 rendering과 MP4

`--live` 실행 중 **http://127.0.0.1:8765**에서 head-camera 영상과 seed/stage를 볼 수 있습니다. 창 시스템이 없는 서버에서도 작동합니다. episode가 바뀌는 초기화 구간에는 재연결을 기다립니다. `--video`는 episode마다 320×240 MP4를 30fps로 저장합니다. 영상 획득은 simulation clock 기준이며 실제 wall-clock 실시간 속도를 보장하는 의미는 아닙니다. 느린 policy 추론 중에는 마지막 프레임이 표시됩니다.

```bash
# desktop DISPLAY가 연결되어 있을 때만 native SAPIEN viewer 사용
robotwin-phys evaluate --manifest manifests/physical_random/stack_blocks_two.json \
  --output runs/fastwam/desktop --viewer --video

# GUI 없이 검증된 초기 상태만 새 프로세스에서 재현
robotwin-phys replay --manifest manifests/physical_random/stack_blocks_two.json \
  --output runs/replay/stack_blocks_two --video
```

원격 policy GPU를 쓰는 경우 로컬에서 `ssh -L 9000:127.0.0.1:9000 user@policy-machine`으로 연결합니다. simulator까지 원격이면 `ssh -L 8765:127.0.0.1:8765 user@simulator-machine`으로 실시간 영상을 볼 수 있습니다. 기본 서비스는 localhost에만 바인딩합니다.

## 재현 범위와 검증

지원 protocol은 `clean`, `official_random`, `physical_random`입니다. 추가 실험용 `physical_only`, `--only mass`, `--without camera`도 제공합니다. 이 추가 ablation은 이 구현에서 제공하는 분석 도구이며, 미공개 논문 실험 결과로 주장하지 않습니다.

```bash
robotwin-phys freeze --task grab_roller --protocol physical_random --only mass \
  --output manifests/ablations/grab_roller_mass.json
pytest -q
python tools/validate_sensitive_tasks.py
```

`reports/`에 환경 점검, 단위/통합 검사, 실제 simulator 영상이 남습니다. 학습된 5개 모델의 실제 성공률이나 논문 leaderboard 재현 결과는 이번 작업에서 산출하지 않습니다. 포인트클라우드 모델용 PyTorch3D는 설치하지 않았으며, 요청한 RGB 기반 평가 경로를 사용합니다.

이번 실행의 [검증 결과](docs/validation.md), [expert MP4](manifests/verified-smoke-stack-blocks-verification/seed-100000.mp4), [테스트 policy MP4](runs/verified-smoke/policy/episode-0000.mp4), [검증된 소규모 manifest](manifests/verified-smoke-stack-blocks.json)를 직접 확인할 수 있습니다.

![물리 randomization을 적용한 실제 SAPIEN head camera](reports/verified-preview.png)

## CSI Lab / RTX PRO 6000 한 장·RTX 5090 두 장 배포와 결과 표

전체 task의 pool을 처음부터 새로 준비할 때는 [v5 새 pool 실행 안내](docs/fresh-pool.md)를 사용합니다. 기존 pool의 재검증·이전 없이 50개 task에서 각각 100개 episode를 준비합니다.

Blackwell CUDA 12.8 이미지, 모델별 정책 이미지, OOD·SLURM·Apptainer 잡과 논문 형식 결과 표는 [클러스터 실행 안내](docs/cluster.md)를 따릅니다. 기존 CUDA 12.1 이미지는 2080 Ti용입니다. `tools/build_bundle.sh`로 기존 여섯 환경을 통합 이미지 한 개에 담습니다. OOD default script의 고정 실행부·캐시는 유지하며, build/push 명령은 클러스터 안내에 있습니다.

기본 `APPTAINERENV_ROBOTWIN_PHASE="all"`은 작은 실험 없이 50 tasks × 3 conditions × 100 episodes × 5 models를 순서대로 평가합니다. 같은 task/seed/randomization 설정은 모델과 독립적으로 같은 환경을 생성하며, 모든 모델이 공유하는 manifest 및 초기 상태와 대조합니다. `tables/randomization.csv`, `randomization.jsonl`, `condition_consistency.csv`와 모델/task/episode별 `*.randomization.json`에서 실제 사용 조건과 일치 여부를 확인할 수 있습니다. 성공률은 `results.md`, `results.tex`, `summary.csv`로 저장됩니다.

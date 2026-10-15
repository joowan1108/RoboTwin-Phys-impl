**v6부터는 [CoM offset 적용 범위 수정과 v6 pool 안내](com-scope-fresh-pool.md)를 사용한다.** 아래는 v5 기록이다.

# 전체 task의 새 pool 구성

**v5 진행 중 `086_woodenblock`·`021_cup`의 집기점 오류나 `put_bottles_dustbin`의 planner 실패 뒤 `IndexError`가 발생했다면 [최신 재개 안내 (v3)](planner-failure-recovery.md)를 사용한다.** 기존 v5 기록을 지우거나 아래의 이전 ZIP을 다시 만들지 않는다.

이 실행은 `phys-pro6000-100-v5`에서 전체 50개 task의 `physical_random` pool을 task당 100개씩 생성한다. `v2`, `v3`, `v4`의 manifest/progress를 가져오거나 재검증하지 않는다. 이전 결과를 삭제하지 않는다. 같은 시작 seed 100000을 사용해도 과거 결과를 재사용하지 않고 현재 시뮬레이터에서 새 후보를 생성하고 expert를 실행한다.

## OOD에서 실행

1. 이전 pool 준비 잡이 실행 중이면 OOD Job Composer에서 Stop/Cancel하고 종료를 확인한다.
2. [robotwin-fresh-pool-v1.zip](../reports/robotwin-fresh-pool-v1.zip)을 OOD Files의 `~/robotwin-data/patches/`에 업로드한다. 압축은 스크립트가 해제한다. 새 이름을 사용하므로 이전 패치 폴더를 수정할 필요가 없다.
3. [ood-pool-pro6000-v5-fresh.txt](../cluster/ood-pool-pro6000-v5-fresh.txt)의 내용을 Job Composer에 붙여 넣어 제출한다. 기본값은 PRO 6000 한 장, GPU당 task worker 4개, CPU 16개, 메모리 128G이다. 두 장을 사용할 때는 `#SBATCH --gres=gpu:pro6000:2`와 CPU 요청을 32로 변경할 수 있다.

이미지는 기존 `docker://joowan1108/robotwin-phys:bundle-cu128-v5`를 그대로 사용한다. 정책 추론은 시작하지 않는다. 출력의 `Model=fastwam`은 기존 scheduler의 모델 기본값 및 로그 폴더 이름이며, `policy GPU=None; replicas=0`으로 동작한다.

최초 출력은 `0/50 tasks already frozen`이다. 이후 **같은 v5** 잡을 재개하면 완료된 task와 저장된 accepted/rejected 진행을 이어서 사용하며, 이 자료를 다시 expert로 재검증하지 않는다. 코드/환경이 달라졌다면 지문 검사가 중단하므로 업로드한 패치는 실행 도중 변경하지 않는다.

task별 진행 로그: `~/robotwin-data/reports/parallel/phys-pro6000-100-v5/fastwam-<잡번호>/freeze-<task>.log`.

task 전체의 seed 진행을 한 파일로 보려면 [OOD 모니터링 스크립트](../cluster/ood-pool-monitor-v5.txt)를 별도 잡으로 제출한다. `POOL_JOB_ID="CHANGE_ME"`에 원래 pool 준비 잡의 번호를 입력한다. 이 스크립트는 GPU를 요청하지 않으며, 실행 중인 pool 잡을 중단하거나 패치를 다시 업로드할 필요가 없다. 기존 task 로그를 먼저 모은 다음 새 줄을 5초 간격으로 읽는다.

통합 로그: `~/robotwin-data/reports/parallel/phys-pro6000-100-v5/fastwam-<원래 pool 잡번호>/pool-progress.log`. 각 줄 앞에 `[adjust_bottle]`처럼 task 이름을 표시한다. 모니터 잡 자체의 `pool-monitor-<모니터 잡번호>.out`에도 같은 내용이 표시된다. 출력 시각은 모니터가 읽은 시각이며, 이미 저장된 로그의 과거 발생 시각을 복원한 것은 아니다. 모니터를 중단해도 pool 생성에는 영향을 주지 않는다.

각 seed의 세부 로그/result: `~/robotwin-data/manifests/phys-pro6000-100-v5/physical_random/<task>-verification/seed-<seed>.log` 및 `.result.json`.

진행 저장: 같은 manifest 폴더의 `<task>.progress.json`. 완료된 task: `<task>.json`. 전체 완료 확인: `~/robotwin-data/manifests/phys-pro6000-100-v5/pool-index.json`.

정상적인 `unstable_initialization`/`expert_failed`는 후보를 제외하고 다음 seed를 검사한다. CUDA, 파일 누락, 알려지지 않은 실행 예외를 후보 실패로 바꾸지 않는다. 해당 task의 오류를 기록하고 다른 task를 계속 준비하며, 모든 task가 완료되어야 전체 pool을 확정한다.

## CuRobo 결과 누적 수정

잡 1644의 `[2] -> [2, 5000]` 오류는 CuRobo v0.7.8의 결과 형식 불일치이다. 공식 고정 revision `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`에서 `TrajOptSolver._get_result(return_all_solutions=True)`는 `goalset_index`의 전체 시간축을 반환한다. 성공 후 finetune의 best-select는 마지막 시간점만 반환한다. 실패한 첫 시도를 clone한 뒤 성공한 후속 시도를 `copy_idx`로 합칠 때 서로 다른 형식이 충돌한다.

모든 seed를 반환하는 경로도 다른 오차 필드 및 best-select와 동일하게 마지막 시간점의 goal index를 반환하도록 수정한다. 원본 per-timestep metrics는 보존한다. 기존 IK mask 소유권 수정도 유지한다. 성공한 궤적을 복사할 때 `optimized_dt`를 함께 복사하고, clone의 path-length 목록을 독립적으로 보관하며, retiming은 반환할 누적 결과에 적용한다. Invalid pose-cost query의 batch 실패 결과 생성도 정상 동작하도록 수정한다.

두 원본 SDK 파일의 전체 SHA, 고정 revision, 수정 대상 함수의 AST를 검사한 뒤 프로세스 내에서만 적용한다. SDK 파일, planner seed 수, 성공 조건은 변경하지 않는다. 변경 ID와 두 원본 파일 SHA는 pool provenance에 기록한다.

## 검증 범위

공식 고정 메서드 본문으로 결과 생성 -> finetune best-select -> 여러 시도 -> clone/copy -> 최종 반환을 PyTorch 2.7.1 CPU tensor에서 검사한다. 원본의 5000열 충돌을 재현한 후 실패/부분 성공/전체 성공 조합, 궤적·dt·goal index·길이 보존, profiler 유지, 원본이 다른 경우 거부를 확인한다. 전체 50개 task 스케줄링에서 이전 실험 파일을 읽지 않는지와 OOD ZIP 추출·체크섬·bind·Bash 문법을 검사한다.

로컬에는 CUDA/SAPIEN 실행 환경이 없으므로 실제 원격 PRO 6000에서 50개 task를 완료한 결과를 보장하는 검증은 수행하지 않았다. 제출한 잡이 CUDA/Vulkan/CuRobo preflight를 먼저 수행하고 실제 후보 생성을 진행한다.

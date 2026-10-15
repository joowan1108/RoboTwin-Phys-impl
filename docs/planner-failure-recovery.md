# planner 실패 뒤 task 예외: v5 pool 재개 (v3)

**이 패치는 앞선 [집기점 번호 불일치 패치 v2](grasp-contact-recovery.md)의 수정을 모두 포함한다.** v2를 적용했든 하지 않았든 이 ZIP 하나로 이어서 실행한다. v2 ZIP은 아래 `put_bottles_dustbin` 오류를 고치지 못한다.

## 확인한 원인

잡 1647의 `put_bottles_dustbin`은 첫 후보 seed 100000에서 다음 위치로 중단되었다.

```text
vendor/RoboTwin/envs/put_bottles_dustbin.py, line 97, in play_once
    right_action[1][0].target_pose[2] += delta_dis
IndexError: list index out of range
```

고정 revision의 [공식 `_base_task.py`](https://github.com/RoboTwin-Platform/RoboTwin/blob/13c3c47ff4312dd62484bcd51be034af55c062d1/envs/_base_task.py)에서 `left_move_to_pose`/`right_move_to_pose`/`together_move_to_pose`는 CuRobo 경로 계획이 `Success`가 아니면 예외 없이 `plan_success = False`만 기록하고 돌아온다. task 스크립트는 그대로 진행되고, 이후의 `grasp_actor`·`place_actor`는 `plan_success`가 False이면 `(None, [])`를 돌려준다. [공식 `put_bottles_dustbin`](https://github.com/RoboTwin-Platform/RoboTwin/blob/13c3c47ff4312dd62484bcd51be034af55c062d1/envs/put_bottles_dustbin.py)의 96~98행은 그 반환값을 `right_action[1][0]`으로 바로 인덱싱하므로, 앞선 병(왼팔)의 경로 계획이 실패한 뒤 오른쪽 병을 처리할 때 `IndexError`가 난다. [공식 평가 루프](https://github.com/RoboTwin-Platform/RoboTwin/blob/13c3c47ff4312dd62484bcd51be034af55c062d1/script/eval_policy.py#L221)는 어떤 예외든 잡아 다음 seed로 넘어가므로 공식 구현에서는 이 seed가 조용히 제외된다.

독립 재현의 worker는 `ExpertPlanningFailure`와 `UnStableError`만 rejection으로 분류하고 나머지 예외는 task 중단으로 처리한다(의도적으로 엄격함). 그래서 같은 상황이 task 전체를 멈췄다. seed 100000이 첫 후보이므로 이 task는 기록이 전혀 없고, 같은 코드로 다시 실행하면 동일 seed에서 같은 방식으로 중단된다. v2 패치는 집기점 annotation 문제만 다루므로 이를 고치지 못한다.

50개 task 중 helper 반환값을 직접 인덱싱하는 것은 `put_bottles_dustbin`뿐이다(`place_dual_shoes`는 `place_actor` 결과를 `move`에 넘기며 `move`는 `plan_success`가 False일 때 즉시 False를 반환한다). 다만 수정은 특정 task 이름이 아니라 "공식 planner가 이미 실패를 기록한 뒤 발생한 예외" 전체에 적용된다.

## 수정 범위

`src/robotwin_phys/planning.py`의 `PlanningMixin`에 `play_once` wrapper를 추가했다.

- `ExpertPlanningFailure`는 그대로 전달한다.
- 그 밖의 예외는 그 시점에 `plan_success`가 **정확히 False**일 때만 `expert_failed / exception_after_planning_failure`로 기록한다. 예외 종류, 메시지, 전체 traceback을 `planning_failure`에 남긴다.
- `plan_success`가 True인 상태의 예외는 종전과 같이 실행 오류로 전파되어 task를 중단한다. 공식 task가 `plan_success = True`로 되돌리는 "Try new way" 재시도(`open_microwave`, `place_can_basket`, `place_object_basket`) 이후의 예외도 실행 오류다.
- `ExpertPlanningFailure`가 actor 없이 생성될 수 있도록 했다. actor가 있을 때의 진단 내용은 v2와 동일하다.

task 코드, 물체 샘플링, planner, 물리 설정은 바꾸지 않았다. 실패한 episode를 다른 seed로 교체하지도 않는다. 이 rejection은 **이 후보에서 scripted expert의 planner가 실패했다**는 뜻이며 공식 루프가 seed를 건너뛰는 것과 같은 결과다. 물체가 물리적으로 집을 수 없다는 증명은 아니다.

한 가지 한계: planner 실패 이후에 발생한 진짜 실행 오류(예: CUDA 오류)도 중단 대신 rejection으로 기록된다. 그 후보는 어차피 거절될 상태였고 traceback은 `seed-<seed>.result.json`에 남으므로, `exception_after_planning_failure` 기록의 `exception_type`을 확인해 점검할 수 있다.

## OOD에서 적용

**v5를 처음부터 만들지 않는다.** 완료된 task와 `.progress.json`의 성공·거절 기록을 보존한다. 새 잡을 제출하기 전에 기존 잡이 종료되어 있어야 한다. 같은 실험의 동시 실행은 lock이 막는다.

1. [robotwin-planner-failure-fix-v3.zip](../reports/robotwin-planner-failure-fix-v3.zip) (SHA256 `4acb0d8aa351e1a91cc062bec742789faddca610be17f24c84c312eca2289791`)을 OOD Files의 `~/robotwin-data/patches/`에 업로드한다. 기존 ZIP·폴더·manifest는 그대로 둔다.
2. [ood-pool-pro6000-v5-planfail.txt](../cluster/ood-pool-pro6000-v5-planfail.txt)를 Job Composer에 붙여 넣어 제출한다.

기본값은 PRO 6000 2장, GPU당 worker 4개, CPU 32개, 50개 task × 100 episode, 시작 seed 100000, 실험명 `phys-pro6000-100-v5`, 이미지 `bundle-cu128-v5`이다. 한 장이면 `#SBATCH --gres=gpu:pro6000:1`, `#SBATCH --cpus-per-task=16`으로 바꾼다.

재개 도구는 저장된 manifest의 `planning.py` SHA가 v5 원본, 빈 목록 패치 v1, 집기점 패치 v2 중 하나이고 나머지 fingerprint가 동일할 때만 provenance를 갱신한다. 다른 차이가 있으면 아무 파일도 바꾸지 않고 중단한다. 원본은 다음 위치에 백업되고 `*.receipt.json`에 `expert_reexecuted: false`가 남는다.

```text
~/robotwin-data/reports/pool-preservation/phys-pro6000-100-v5/
  expert-exception-after-planning-failure-v3/originals/
```

시작 시 다음 줄이 순서대로 나와야 한다.

```text
PLANNER_FAILURE_PATCH_SOURCE_PASS: reviewed simulator snapshot verified
GRASP_ASSET_AUDIT: ...
POOL_PROGRESS_PRESERVED ...; expert replay=0; ...
POOL_CONTINUATION_READY: ... same v5 pool, no older experiment import
```

`put_bottles_dustbin`은 기록이 없으므로 seed 100000부터 다시 검사한다. 같은 상황이면 task 로그에 `Rejected seed 100000: expert_failed`가 나오고 100001로 진행한다. `seed-100000.result.json`의 형태는 다음과 같다.

```json
{
  "status": "rejected",
  "reason": "expert_failed",
  "plan_success": false,
  "planning_failure": {
    "kind": "exception_after_planning_failure",
    "actor": null,
    "arm": null,
    "contact_point_ids": [],
    "exception_type": "IndexError",
    "exception_message": "list index out of range",
    "traceback": "Traceback (most recent call last): ..."
  }
}
```

- task 로그: `~/robotwin-data/reports/parallel/phys-pro6000-100-v5/fastwam-<새 잡번호>/freeze-put_bottles_dustbin.log`
- 완료된 task: `~/robotwin-data/manifests/phys-pro6000-100-v5/physical_random/<task>.json`
- 전체 완료: 같은 실험 폴더의 `pool-index.json`

## 검증

- 고정 revision의 96~98행을 옮긴 test task로 공식 동작(왼팔 경로 실패 → 오른팔 `IndexError`)을 재현하고, 수정본이 같은 입력을 `exception_after_planning_failure`로 분류하며 traceback에 원인 줄을 보존하는지 확인했다.
- 경로 계획이 성공하는 경우 공식 task와 동일한 결과를 내고, `plan_success`가 True인 상태의 `IndexError`/`RuntimeError`/`AssertionError`/`KeyError`와 "Try new way" 재설정 이후의 예외는 그대로 전파됨을 확인했다. `KeyboardInterrupt`는 분류하지 않는다.
- v2 ZIP의 `planning.py`와 현재 파일의 AST 차이가 `import traceback`, `play_once` 추가, `__init__`의 actor 없음 허용에 한정됨을 확인했고, actor가 있는 경우 두 버전의 진단 내용이 같음을 확인했다.
- v5 원본·v1·v2 세 가지 이전 source에서의 기록 보존, 오류 seed 재검사 후 진행, ZIP 해제·체크섬·bind·Bash 문법을 검사했다. [검증 결과](../reports/planner-failure-validation.json)를 참고한다.

로컬에는 GPU/SAPIEN/torch가 없어 CuRobo 의존 테스트 2개 모듈은 실행하지 못했고, seed 100000의 실제 물리 시뮬레이션 재실행도 하지 않았다. 알려지지 않은 다른 실행 오류까지 모두 제거했다고 보장하지 않는다.

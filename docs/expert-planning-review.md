# RoboTwin 2.0 expert 실패 처리 대조

2026-10-13 추가 확인: 공식 `086_woodenblock` metadata는 접촉점 목록이 명시적으로 비어 있다. 이 경우는 파일 누락과 구별해 `expert_failed/no_grasp_annotations`로 처리하도록 보완했다. 현재 v5 pool을 계속 준비할 때는 [진행 보존 안내](empty-grasp-recovery.md)를 사용한다. 아래 내용과 ZIP은 이전 수정 단계의 기록이다.

2026-10-12에 공식 `stable_2.0`을 checkout해 확인했습니다. HEAD는 현재 재현 코드가 고정한 `13c3c47ff4312dd62484bcd51be034af55c062d1`과 같습니다. 코드 경로와 실패 처리를 검증했으며, 원격 seed 100054의 GPU 시뮬레이션을 이 로컬 환경에서 실행한 것은 아닙니다.

## 공식 코드와 달랐던 점

- [공식 평가 루프](https://github.com/RoboTwin-Platform/RoboTwin/blob/13c3c47ff4312dd62484bcd51be034af55c062d1/script/eval_policy.py#L221)는 expert의 초기화·실행 중 `UnStableError`뿐 아니라 모든 `Exception`을 잡고 다음 seed로 넘어갑니다.
- [공식 수집 루프](https://github.com/RoboTwin-Platform/RoboTwin/blob/13c3c47ff4312dd62484bcd51be034af55c062d1/script/collect_data.py#L127)도 일반 예외를 실패한 후보로 처리합니다.
- [집기 함수](https://github.com/RoboTwin-Platform/RoboTwin/blob/13c3c47ff4312dd62484bcd51be034af55c062d1/envs/_base_task.py#L1032)는 planner의 성공 후보가 없으면 `(None, None)`을 반환할 수 있습니다. `grasp_actor()`가 이를 이동 명령으로 만들어 `target_pose cannot be None` assertion이 발생합니다. 공식 실행 루프에서는 이 assertion도 위의 일반 예외 처리로 제외됩니다.
- 현재 독립 재현 worker는 초기 불안정과 `plan_success/check_success` 실패만 제외했고, 집기 assertion은 전체 freeze를 중단시켰습니다. 따라서 앞서 “공식 implementation에도 이 예외 처리가 없다”고 설명한 것은 평가 루프까지 고려하지 못한 설명입니다.

## 수정

`PlanningMixin`이 **원본의 집기 탐색과 선택 알고리즘을 그대로 호출**합니다. 지정된 접촉점의 변환 행렬이 존재하고 유효한지 먼저 확인하고, 탐색을 완료한 뒤 두 집기 자세가 모두 `None`이면 `ExpertPlanningFailure`를 발생시킵니다. worker는 expert 검증 중 이 예외만 `expert_failed`로 기록합니다. actor 이름·현재 pose·arm·검사한 접촉점 목록을 JSON에 남깁니다.

첫 후보 실패 시 나머지 접촉점 탐색은 유지됩니다. 접촉점 누락·잘못된 행렬·파일 누락·CUDA 오류·다른 assertion은 실행 오류로 중단합니다. 공식 코드의 광범위한 `except Exception: continue`를 그대로 복사하면 이전의 `curobo_left.yml` 누락 같은 환경 오류도 후보 제외로 숨겨지므로, 이 부분은 의도적으로 더 엄격하게 유지합니다.

`click_alarmclock`처럼 `get_grasp_pose()`를 직접 조회하는 공식 task도 같은 실패를 명시적으로 기록합니다. 여러 접촉점을 탐색하는 도중에는 실패한 후보의 `None` 반환을 유지하고, 모든 후보를 검사한 뒤 제외합니다.

이 수정은 집기 실패의 처리 경로를 고칩니다. 물리 무작위화 범위, 안정성 검사, 성공 조건, expert 중 외력 비활성화, policy 평가 seed 고정은 유지합니다. 도달 가능한 집기 후보가 없는 근본 원인이 물체 pose인지 물리 적용 문제인지는 GPU 재실행 결과로 확인해야 합니다.

## 기존 v5 이미지에 OOD로 적용

1. [패치 ZIP](../reports/robotwin-expert-fix-v1.zip)을 다운로드합니다.
2. OOD Files에서 `$HOME/robotwin-data/patches/`를 만들고 ZIP을 업로드합니다. 이름은 `robotwin-expert-fix-v1.zip`을 유지합니다. 압축 해제는 잡 스크립트가 자동으로 수행합니다.
3. Job Composer에 [seed 100054 진단 스크립트](../cluster/ood-expert-diagnose-pro6000-v5.txt)를 붙여넣고 제출합니다. 결과는 `robotwin-data/reports/expert-diagnostics/<job-id>/summary.json`에 저장되며 Slurm 출력에도 표시됩니다.
4. [FastWAM 수정 실행 스크립트](../cluster/ood-fastwam-pro6000-v5-expert-fix.txt)로 평가를 실행합니다. 이미지 재빌드 없이 전체 simulator source directory를 읽기 전용 bind합니다.
5. π0.5는 [같은 수정 코드용 스크립트](../cluster/ood-pi05-pro6000-v5-expert-fix.txt)를 사용합니다. 해당 스크립트는 완성된 공통 pool을 읽는 evaluate 단계입니다.

새 스크립트의 실험 이름은 `phys-pro6000-100-v2`입니다. 코드가 바뀌었으므로 기존 v1의 partial/frozen manifest와 source fingerprint가 다릅니다. v1 파일을 지우거나 provenance를 강제로 바꾸지 않고, v2에서 후보 검증을 새로 시작합니다. 모델들도 동일한 v2 pool을 사용해야 합니다. 패치는 현재 simulator source snapshot 전체이므로 실행 중 폴더를 바꾸지 마세요. 원격에 이미 다른 source 수정이 있다면 별도 검토가 필요합니다.

진단은 같은 seed에 대해 `official_random`과 `physical_random`을 비교합니다. 전자는 원래의 물리 값과 RoboTwin 시각·배치 무작위화를 유지하고 후자는 물리 변화를 추가합니다. 둘 다 현재 adapter를 사용하며 원본 평가 프로그램 전체의 완전한 복제라는 의미는 아닙니다. 진단은 평가 pool을 수정하지 않습니다.

- 원본 조건은 `verified`, 물리 조건은 `no_reachable_grasp`: 추가 물리 조건에서 planner가 집기 후보를 찾지 못했다는 근거입니다. 특정 물리 factor를 확정하는 것은 아닙니다.
- 두 조건 모두 같은 계획 실패: 해당 기본 장면·planner·asset을 더 확인해야 합니다.
- 초기 불안정: 집기 단계까지 도달하지 않았으므로 해당 실행만으로 집기 원인을 비교할 수 없습니다.
- `status=error`: 실제 환경·코드 오류입니다. 별도 `.log`와 result traceback을 확인해야 합니다.

## 검증

공식 revision의 `choose_best_pose`, `get_grasp_pose`, `choose_grasp_pose`, `grasp_actor`, `Action`을 추출한 fixture로 원래 assertion을 재현했습니다. 수정 후 전체 후보 실패의 명시적 rejection, 첫 후보 실패 후 다음 후보 성공, 정상 동작의 동일한 action 출력, 누락·잘못된 접촉점, 예상 밖 실행 오류의 전파를 검사합니다. worker가 기록하는 진단 정보와 오류 traceback도 검증합니다. 이 CPU 테스트는 GPU planner 자체의 실행 성공을 보증하지 않습니다.

최종 검증: 관련 CPU 테스트 85개 통과. 집기 예외 처리·진단·OOD ZIP 추출과 source bind·공통 pool/resume·실험 보고·조건 생성 테스트를 포함합니다. 테스트 환경은 Python 3.12이며, 변경한 Python 파일은 Python 3.10 문법으로도 파싱을 확인했습니다. 생성한 sbatch 문법, OOD 복사본의 동일성, 보호된 Apptainer 실행부 보존도 확인했습니다. 로컬에 NVIDIA 장치가 없어 원격 seed 100054의 실제 planner 실행은 별도 진단 잡으로 확인해야 합니다.

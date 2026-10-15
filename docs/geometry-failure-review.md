# 공통 placement 수학 오류 수정과 저장된 pool 재검증

## 원인과 범위

잡 1628의 `beat_block_hammer`, seed 100113은 `get_align_matrix()`에서 `arccos(dot(v1, v2))`가 NaN을 만든 뒤 `mat2quat()`의 고유값 계산이 실패했습니다. 정규화한 float32 벡터도 반올림 때문에 내적이 `[-1, 1]`을 조금 벗어날 수 있습니다. 길이가 0인 방향 정규화도 NaN을 만들 수 있습니다. `missing pytorch3d`와 Warp 경고는 보내주신 traceback의 종료 지점이 아닙니다.

공식 `stable_2.0`, revision `13c3c47ff4312dd62484bcd51be034af55c062d1`을 대조했습니다. [공통 transforms](https://github.com/RoboTwin-Platform/RoboTwin/blob/13c3c47ff4312dd62484bcd51be034af55c062d1/envs/utils/transforms.py#L385)의 범위 보호 누락이 직접적인 원인입니다. 공식 50개 task를 정적으로 검사했으며 39개가 `place_actor/get_place_pose` 경로를 사용합니다. 목록은 [task 경로 검사 결과](../reports/expert-task-paths.json)입니다. 이는 50개 task의 실제 GPU rollout을 실행했다는 의미는 아닙니다.

## 수정

- 공통 정렬 함수의 내적을 `[-1, 1]`로 제한하고, quaternion 거리 계산도 유효 범위를 보호합니다.
- 반대 방향 벡터는 deterministic한 180° 회전으로 정렬합니다. 원본은 cross product가 작은 경우 방향과 관계없이 identity를 반환했습니다.
- 길이 0·NaN·Inf 방향을 명시적으로 검출합니다. view direction과 up direction이 평행한 경우에는 유효한 직교 프레임을 구성합니다.
- placement 입력 pose와 functional metadata를 먼저 검증합니다. 유효한 입력에서 배치 방향이 정의되지 않는 경우는 `undefined_placement_direction`이라는 expert 실패로 기록합니다. 누락된 point나 비정상적인 입력 상태는 실행 오류로 남깁니다.
- CuRobo batch 실행 예외를 무조건 `Failure` 10개로 바꾸던 공식 코드도 보완합니다. 정상적인 계획 실패는 기존처럼 result.success로 판단하고, OOM 등 실행 예외는 실제 오류로 전파합니다.
- 한 task가 실행 오류로 실패해도 나머지 task를 계속 준비합니다. 실패한 task는 완료로 간주하지 않고, 마지막에 `freeze-errors.json`과 전체 오류 상태를 남깁니다. 미완성 pool은 확정하거나 모델 평가에 사용하지 않습니다.
- worker 오류의 마지막 traceback을 task 로그에도 표시합니다. task 오류 요약은 Slurm 로그와 `freeze-errors.json`에 남습니다.

공식 task별 성공 조건, 물리 범위, seed 샘플링, expert 중 외력 비활성화는 유지합니다. 수학 함수 수정은 expert 동작에 영향을 줄 수 있으므로 기존 provenance를 그냥 변경하지 않습니다.

## OOD에서 적용

1. [robotwin-expert-fix-v2.zip](../reports/robotwin-expert-fix-v2.zip)을 기존 `~/robotwin-data/patches/`에 새 파일로 업로드합니다. 이전 v1 ZIP과 폴더는 유지합니다.
2. 실행 중인 v2 잡이 있으면 Stop/Cancel하고 종료를 확인합니다.
3. [재검증 실패 시 자동 충원하는 병렬 준비 스크립트](../cluster/ood-pool-pro6000x2-v5-refill.txt)를 Job Composer에서 실행합니다. 기본값은 PRO 6000 두 장·GPU당 worker 네 개입니다. 이미지 재빌드는 필요 없습니다. 앞서 v2 ZIP을 올렸다면 추가 ZIP 업로드 없이 이 스크립트만 교체합니다.
4. seed 100113을 별도로 비교하고 싶으면 [진단 잡](../cluster/ood-geometry-diagnose-pro6000-v5.txt)을 사용합니다.
5. 준비가 완료되면 [v3 FastWAM 실행](../cluster/ood-fastwam-pro6000-v5-refill.txt) 또는 [v3 π0.5 실행](../cluster/ood-pi05-pro6000-v5-geometry-fix.txt)을 사용합니다. 모든 모델은 동일한 v3 pool을 사용합니다.

새 pool은 `phys-pro6000-100-v3`입니다. v2 manifest/progress는 읽기만 하고 보존합니다. task마다 기존에 채택된 seed를 수정된 코드의 expert로 다시 실행해 성공 여부·초기 상태·instruction을 확인한 뒤 v3로 가져옵니다. partial task는 그 뒤 다음 candidate부터 이어갑니다. 완료된 task도 저장된 100개를 재검증하며, 신규 candidate를 처음부터 다시 찾지는 않습니다.

재검증에서 `expert_failed` 또는 `unstable_initialization`이 나오면 그 seed를 제외 사유와 함께 기록하고, 통과한 seed는 유지합니다. task별로 기존 seed 재검증을 마친 뒤 마지막으로 사용한 candidate 다음부터 부족한 개수를 채웁니다. 예를 들어 기존 100개 중 3개가 실패하면 97개를 유지하고 새로운 expert-verified episode 3개를 더 확보합니다. retained episode_index는 0부터 다시 연속으로 부여합니다.

기존 rejection 기록은 유지하므로 수정 코드로 최초 seed부터 다시 샘플링한 새 pool이라는 의미는 아닙니다. 정상적인 infeasible seed 제외는 **평가 전 pool 준비 단계**에만 적용합니다. 모델 평가에 실패한 episode를 다른 seed로 바꾸지는 않습니다. 초기 상태·instruction 불일치나 실행 오류는 해당 task 오류로 남깁니다. 재검증과 seed 제외 결정은 별도 progress에 저장해 중단 후 이어갈 수 있으며, 이전 도구로 진행한 reverification progress도 이어서 처리합니다. runtime·asset·물리 설정이나 승인한 수정 이외의 파일이 다르면 가져오기를 거부합니다.

## 로그

```text
~/robotwin-data/reports/parallel/phys-pro6000-100-v3/fastwam-<잡번호>/reverify-<task>.log
~/robotwin-data/reports/parallel/phys-pro6000-100-v3/fastwam-<잡번호>/freeze-<task>.log
~/robotwin-data/reports/parallel/phys-pro6000-100-v3/fastwam-<잡번호>/freeze-errors.json
```

task migration 감사 기록은 `manifests/phys-pro6000-100-v3/physical_random/<task>.reverification.json`에 저장됩니다. 완료된 task는 `freeze-<task>.log` 없이 `reverify-<task>.log`만 있을 수 있습니다.

## 검증의 범위

원본 함수로 float32 내적 범위 초과와 `Eigenvalues did not converge`를 재현하고 수정 후 finite rotation을 확인합니다. 평행·반평행·직교 방향, 일반 회전과 원본의 일치, quaternion sign invariance, 0 길이·NaN/Inf 입력, 실제 원본 placement action 생성, 독립 task 오류 격리, 기존 seed 재검증과 중단 후 resume, patch archive와 OOD bind를 CPU 테스트로 검사합니다.

원격 GPU에서 seed 100113을 다시 실행하거나 50개 task 전체 rollout을 완료한 상태는 아닙니다. 아직 확인하지 못한 다른 오류가 절대로 없다고 보장하지는 않습니다. 이제 그러한 task 오류는 다른 task 준비를 즉시 취소하지 않고 기록하며, 미완성 pool을 평가에 넘기지 않습니다.

최종 관련 CPU 테스트: **131개 통과**. 변경 코드와 패키지의 Python 3.10 문법, 전체 source/tools checksum, OOD shell 문법, 보호된 Apptainer 실행부 보존도 확인했습니다. 테스트 실행 환경은 Python 3.12입니다.

이후 정상적인 재검증 실패를 제외하고 부족한 episode를 충원하도록 controller를 보완했습니다. 이 변경의 관련 테스트 **59개가 통과**했으며, 재검증 실패 후 새 seed 충원·episode_index 재부여·거절 결정 후 중단/resume·이전 progress 호환·실행 오류 중단·추가 ZIP 없는 OOD 파일 bind를 확인했습니다. simulator source와 수학 패치 ZIP은 변경하지 않아 기존 v3 진행을 유지합니다.

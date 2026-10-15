# v5 pool 진행 보존 및 빈 집기점 처리

**추가로 `Missing grasp contact point 2 for 021_cup`이 발생했다면 [집기점 번호 불일치까지 포함한 최신 패치](grasp-contact-recovery.md)를 사용한다.** 새 ZIP은 이 문서의 빈 목록 수정도 포함하며, 이전 패치를 먼저 적용할 필요가 없다. 아래 내용은 v1 수정의 기록이다.

**전체를 처음부터 생성하지 않는다.** 이번 오류는 `086_woodenblock`의 공식 metadata에 집기점이 없는데, 재현 코드가 이를 파일 손상처럼 처리한 문제다. 새 패치는 이미 저장된 v5의 성공·거절 기록을 보존하고, 중단된 후보부터 전체 50개 task의 pool 준비를 계속한다. 정책 평가는 실행하지 않는다.

## 원인 확인

[공식 objects.zip](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main)의 `086_woodenblock/model_data0.json`부터 `model_data4.json`까지 모두 `contact_points_pose: []`이다. ZIP 중앙 디렉터리와 해당 멤버를 HTTP Range로 읽고 각 파일의 CRC를 검증했다. 공식 저장소 API의 ZIP SHA256은 기존 다운로드 고정값 `6aa56b3cf1e1064f7c809308144da36b00815f8b137fef2d7e4de856f8becf27`과 같다. [파일별 근거](../reports/grasp-annotation-review.json)와 실제 JSON fixture를 보관했다.

[공식 place_a2b_left](https://github.com/RoboTwin-Platform/RoboTwin/blob/13c3c47ff4312dd62484bcd51be034af55c062d1/envs/place_a2b_left.py)와 right는 이 물체도 추출한다. 집기 후보가 없으면 원본 expert의 action 생성이 실패하고, [공식 평가 루프](https://github.com/RoboTwin-Platform/RoboTwin/blob/13c3c47ff4312dd62484bcd51be034af55c062d1/script/eval_policy.py#L221)는 예외를 잡아 다음 seed로 넘어간다. 독립 재현의 `PlanningMixin`이 빈 목록을 무조건 `ValueError`로 처리하면서 task가 중단되었다. 이 분류는 잘못되었다.

명시적인 빈 목록은 이제 `expert_failed`, 세부 유형 `no_grasp_annotations`로 기록된다. 같은 공통 함수가 적용되는 다른 task와 articulation도 동일하게 처리한다. 특정 물체를 샘플링 대상에서 제거하거나 임의로 집기점을 생성하지 않는다. 접촉점 key/파일 누락, null, 잘못된 행렬, CUDA 오류 등은 여전히 실행 오류다.

이 rejection은 **주어진 annotation으로 scripted expert가 행동을 구성할 수 없었다**는 뜻이다. 사람이든 다른 planner든 물리적으로 절대 집을 수 없다는 증명은 아니다. policy 평가 중 실패한 episode를 교체하는 동작도 아니다.

## OOD에서 이어서 실행

1. 현재 v5 pool 잡이 다른 task를 처리 중이면 그대로 완료하도록 두어도 된다. 바로 전환하려면 Job Composer에서 Stop/Cancel하고 **잡 종료를 확인한 뒤** 다음 잡을 제출한다. 동일 실험의 동시 실행은 lock으로 막는다.
2. [robotwin-empty-grasp-fix-v1.zip](../reports/robotwin-empty-grasp-fix-v1.zip)을 OOD Files의 `~/robotwin-data/patches/`에 업로드한다. 기존 ZIP·패치 폴더·manifest를 삭제하거나 수정하지 않는다. 새 ZIP은 새 디렉터리에 자동 해제된다.
3. [ood-pool-pro6000-v5-resume.txt](../cluster/ood-pool-pro6000-v5-resume.txt) 전체를 Job Composer에 붙여 넣어 제출한다. 기본값은 PRO 6000 **2장, GPU당 worker 4개**, CPU 32개, RAM 128G이다. 한 장만 쓰려면 `#SBATCH --gres=gpu:pro6000:1`, `#SBATCH --cpus-per-task=16`으로 바꾸면 된다.

기존 이미지 `docker://joowan1108/robotwin-phys:bundle-cu128-v5`, 같은 실험명 `phys-pro6000-100-v5`, task당 100개, 시작 seed 100000을 사용한다. **v2/v3/v4를 가져오거나 재검증하지 않는다.** 이후 중단·재개에도 이 새 스크립트를 사용한다.

실행은 먼저 기존 CUDA/Vulkan/CuRobo 점검을 수행하고, 전체 `objects/*/model_data*.json`을 검증된 ZIP의 크기·CRC와 비교한다. 빈 집기점 목록과 추출 손상을 구별한 보고서를 `reports/parallel/phys-pro6000-100-v5/fastwam-<새 잡번호>/grasp-assets.json`에 쓴다.

이후 다음과 같은 출력이 나온다.

```text
EMPTY_GRASP_PATCH_SOURCE_PASS: reviewed simulator snapshot verified
GRASP_ASSET_AUDIT: ... metadata files; ... explicit empty contact lists
POOL_PROGRESS_PRESERVED ...: accepted=...; rejected=...; expert replay=0
POOL_CONTINUATION_READY: ... same v5 pool, no older experiment import
Pool preparation: 8 workers on 2 GPUs; .../50 tasks already frozen
```

완료된 task는 그대로 사용한다. 미완료 task는 `.progress.json`에 기록된 마지막 후보 다음부터 진행한다. 에러로 종료한 seed는 아직 채택·거절 기록이 없으므로 그 seed를 다시 검사한다. 예를 들어 `place_a2b_right/seed-100092`가 마지막 오류이고 이후 기록이 없다면 그 후보를 다시 실행하여 명시적인 expert rejection을 기록한 뒤 100093으로 진행한다.

## 재검증 없이 보존할 수 있는 범위

새 ZIP은 이미 배포한 `robotwin-fresh-pool-v1.zip`에서 `planning.py`의 **기존에 반드시 예외로 종료하던 빈 접촉점 분기**만 바꾼다. 나머지 runtime source, CuRobo 수정, vendor 코드, 물리 설정은 그대로다. 테스트는 해당 분기를 제외한 Python AST 전체가 동일한지도 확인한다. 이전 성공·거절 기록을 반환한 실행은 이 오류 분기에 진입하지 않았으므로, 이번 변경 때문에 이전 expert를 다시 실행할 필요가 없다.

재개 도구는 이전·현재 source SHA, assets, 버전, GPU 종류/드라이버, CuRobo 설정 등 fingerprint 전체를 비교한다. 검토한 `planning.py`의 이전→이후 SHA 변경 한 개만 허용한다. 다른 차이가 있으면 중단하며 임의의 fingerprint 무시는 하지 않는다. 이미 평가가 시작됐거나 확정된 pool도 수정하지 않는다.

변경 전 모든 manifest/progress 원본을 다음 위치에 백업하고, 저장된 episode·rejection을 그대로 유지한 채 검토된 변경을 provenance에 반영한다.

```text
~/robotwin-data/reports/pool-preservation/phys-pro6000-100-v5/
  explicit-empty-grasp-annotations-v1/originals/
```

같은 폴더의 `*.receipt.json`에는 이전·현재 fingerprint, 원본 위치, 보존한 기록 수와 해시, `expert_reexecuted: false`가 남는다. 이는 이전 seed를 새 코드로 실제 재실행했다는 기록이 아니다. 도중 종료 후 재개해도 이미 처리한 파일은 다시 변경하지 않는다.

## 로그와 완료 확인

- task 진행: `~/robotwin-data/reports/parallel/phys-pro6000-100-v5/fastwam-<새 잡번호>/freeze-<task>.log`
- candidate 세부 사유: `~/robotwin-data/manifests/phys-pro6000-100-v5/physical_random/<task>-verification/seed-<seed>.result.json`
- 저장된 진행: 같은 폴더의 `<task>.progress.json`; 완료된 task는 `<task>.json`
- 전체 완료: `~/robotwin-data/manifests/phys-pro6000-100-v5/pool-index.json`

모든 task의 seed 로그를 모으려면 기존 [모니터 스크립트](../cluster/ood-pool-monitor-v5.txt)의 `POOL_JOB_ID`를 **새 pool 잡번호**로 설정하고 별도 제출한다. `pool-progress.log`와 모니터 잡 출력에서 통합 진행을 볼 수 있다. GPU는 추가로 요청하지 않는다.

## 반복 오류의 전체 원인

| 문제 | 이번까지 확인된 원인 | 대응 |
|---|---|---|
| Vulkan 렌더러 초기화 실패 | Apptainer가 주입한 호스트 EGL과 이미지 glibc 불일치 | 이미지의 EGL/GLdispatch 우선 사용 |
| `curobo_left.yml` 없음 | asset 압축 해제 후 경로 template 생성 누락 | 공식 경로 생성 단계 실행 |
| 집기 assertion·빈 집기점에서 task 중단 | 공식 루프의 후보 제외 동작을 엄격한 worker로 옮기면서 일부 expert 실패를 실행 오류로 분류 | 근거가 확인된 실패만 구조화해 rejection 처리 |
| placement NaN | 회전 계산의 수치 경계 처리 부족 | 공통 수학 helper 수정 |
| CuRobo shape mismatch | 고정된 SDK의 batch mask 공유 및 결과 형식 불일치 | 해당 SDK 버전용 호환 수정 |
| 반복 재검증/이전 실험 이동 오류 | 잦은 코드 변경과 엄격한 provenance 검사 사이의 운영 문제 | 이번에는 같은 v5의 검토된 변경 한 건만 기록하고 진행 보존 |

사전 검증이 CUDA·렌더링·FK 및 작은 expert 사례에 집중되어 50개 task와 다양한 asset의 expert 경로를 충분히 커버하지 못한 것이 공통적인 검증 한계다. 이번 오류를 논문의 잘못으로 볼 근거는 없다. 이 프로젝트는 RoboTwin-Phys 논문의 독립 재구현이며, 위 오류에는 RoboTwin/CuRobo 코드 경로와 우리 통합·예외 처리 문제가 함께 있었다.

## 검증과 한계

실제 공식 woodenblock 5종 metadata로 원본 오류를 재현하고, 새 분류·정상 집기 탐색 유지·손상 데이터 오류 유지·기존 기록 보존·다음 seed 진행·OOD 압축 해제와 bind·체크섬·Bash 문법을 검사한다. [검증 기록](../reports/empty-grasp-validation.json)을 확인할 수 있다.

로컬 검증은 CPU 환경이다. 원격 GPU의 실제 50-task rollout 완료를 이곳에서 확인하지는 못했으며, GPU/메모리/다른 task 구현 오류까지 모두 사라졌다고 보장하지 않는다. 알려지지 않은 실행 오류는 해당 task 로그에 남기고 나머지 task는 계속 진행한다.

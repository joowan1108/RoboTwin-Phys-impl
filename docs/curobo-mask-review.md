# CuRobo의 IK 성공 마스크 공유 오류

현재 실행은 [전체 50개 task의 새 v5 pool 구성](fresh-pool.md)을 사용합니다. 이전 pool의 재검증·이전 없이 시작하며, 아래 기록 이후 확인된 결과 누적 shape mismatch 수정도 포함합니다. 아래 v4 이전 절차는 당시 실행 기록입니다.

잡 1640에서 `adjust_bottle` seed 100000과 `beat_block_hammer` seed 100115가 CuRobo `motion_gen.py:3992`의 shape mismatch로 종료됐습니다. 이는 정상적인 expert infeasibility가 아니라 planner 내부의 tensor 소유권 오류입니다. 공식 저장소에도 [동일한 문제 #502](https://github.com/NVlabs/curobo/issues/502)가 보고돼 있습니다.

## 확인한 원인과 수정

고정한 CuRobo `v0.7.8`, revision `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`의 batch planner는 `MotionGenResult(ik_result.success, ...)`로 결과를 만듭니다. 두 객체가 같은 tensor를 가리키므로 graph search 실패 후 `result.success[:] = False`가 IK 성공 마스크도 지웁니다. 이미 성공한 IK 3개 또는 5개로 만든 `goal_config`는 남아 있는데, 그 값을 넣을 indexing 결과는 0개가 되어 오류가 발생합니다.

single/batch planner의 결과 생성 시 success tensor를 `clone()`하여 분리합니다. Graph 실패가 유효한 query이면 기존에 의도된 trajectory optimization 경로를 계속 시도합니다. 실제 IK·graph·trajectory 실패는 실패 결과로 남고, 이 tensor 예외를 무조건 `expert_failed`로 바꾸지는 않습니다. 기존 optimizer 설정, seed 수, 실패/성공 판정은 유지합니다.

`curobo_compat.py`가 원본 file SHA·revision·두 함수의 구문 구조를 검증한 뒤 두 생성 인수만 수정합니다. 원본 SDK 파일은 수정하지 않으며 각 episode 프로세스에서 동일하게 적용합니다. API descriptor와 기본 인수를 보존합니다. 코드 지문에 CuRobo revision·원본 file SHA·수정 ID도 기록해 planner 차이가 드러나도록 했습니다. 이 오류 경로는 다른 worker와 tensor를 공유하는 경로가 아니므로 GPU당 worker 수를 줄이는 것 자체가 이 버그를 고치는 방법은 아닙니다.

## OOD 적용

1. [새 CuRobo 패치 ZIP](../reports/robotwin-curobo-fix-v1.zip)을 기존 `~/robotwin-data/patches/`에 업로드합니다. 이전 파일은 보존합니다.
2. 이전 잡이 실행 중이면 Stop/Cancel 후 종료를 확인합니다.
3. [이전 helper 파일 부재를 처리하는 최신 병렬 준비 스크립트](../cluster/ood-pool-pro6000x2-v5-provenance-fix.txt)를 사용합니다. GPU 두 장·GPU당 worker 네 개가 기본값입니다. worker 수는 기존처럼 조절할 수 있습니다. 이미지 빌드는 필요 없습니다. 기존 `robotwin-curobo-fix-v1.zip`을 그대로 쓰며 추가 ZIP 업로드는 필요 없습니다.
4. 별도로 실패 seed를 확인하려면 [진단 스크립트](../cluster/ood-curobo-diagnose-pro6000-v5.txt)를 사용합니다. `adjust_bottle` 100000, `beat_block_hammer` 100115의 original/physical 조건을 각각 실행합니다.
5. 준비 완료 후 [v4 FastWAM](../cluster/ood-fastwam-pro6000-v5-provenance-fix.txt) / [v4 π0.5](../cluster/ood-pi05-pro6000-v5-curobo-fix.txt)를 사용합니다.

새 실험은 `phys-pro6000-100-v4`입니다. task별로 v3의 저장된 pool/progress를 우선 사용하고, 해당 task 자료가 없으면 v2에서 가져옵니다. 원본 파일은 읽기만 하고, 채택된 seed를 새 planner로 재검증합니다. 재검증의 정상적인 expert 실패/불안정은 제외하고 다음 사용하지 않은 seed로 부족한 수를 채웁니다. 실행 오류나 초기 상태·instruction 불일치는 다른 task 준비와 분리해 보고하며, 미완성 pool은 평가에 사용하지 않습니다.

v3에서 candidate 검사만 일부 했고 accepted episode가 없다면 그 자료도 progress에 보존합니다. v3의 reverification 중에만 저장된 자료가 있고 정규 `.progress.json`은 아직 없다면 v2 원본을 재검증합니다. 이런 경우 중간 reverification 일부가 다시 실행될 수 있습니다. 완료된 v4 검증과 재검증은 저장된 progress에서 이어집니다.

## 검증

원본의 `_plan_from_solve_state_batch` 본문을 그대로 사용하고 PyTorch **2.7.1 CPU tensor**로 `[3, 6] → [0, 6]`과 `[5, 6] → [0, 6]`을 재현했습니다. 수정 후 graph 실패의 trajectory fallback, 실제 trajectory 실패, invalid query, IK 성공 0개, graph 성공·부분 성공을 검사했습니다. Single planner의 마스크도 원본 IK 결과와 분리되는지 확인했습니다. 함수 identity/default 보존·idempotence·원본이 다른 경우 거부·Python 3.10/3.12 구문 차이 처리도 검사합니다.

원격 CUDA 실행이나 두 실제 seed의 GPU 재실행은 아직 이 로컬 환경에서 검증하지 않았습니다. 새 진단 잡이 그 검증을 수행합니다.

최종 관련 CPU 테스트 **164개 통과**. 새 OOD ZIP 추출과 source bind·저장 자료의 v3 우선/v2 fallback·재검증 실패 후 충원·작업 중단 후 resume도 확인했습니다. 변경 코드의 Python 3.10 문법과 패키지 checksum, 보호된 Apptainer 실행부 보존을 확인했습니다. 테스트 환경은 Python 3.12와 PyTorch 2.7.1+cpu입니다.

잡 1641의 `Unreviewed previous simulator source: src/robotwin_phys/numerics.py`에 대해, v2에 없던 파일이 v3에 추가된 경우를 이전 해시 목록 검사에서 거절하는 문제를 재현하고 수정했습니다. 이전 파일이 없고 현재 값이 알려진 helper 해시일 때만 허용하며, 존재하는 이전 파일의 알 수 없는 해시나 새로운 알 수 없는 helper 해시는 계속 거부합니다. 관련 테스트 56개 통과. simulator source와 ZIP은 그대로 유지하므로 v4 진행을 이어갈 수 있습니다.

정확한 이전 자료와 해시를 확인하려면 [읽기 전용 provenance 진단 잡](../cluster/ood-provenance-diagnose-pro6000-v5.txt)을 사용합니다. 모델/episode를 실행하지 않고, job 1641의 current provenance와 선택된 v3/v2 자료의 `numerics.py` 존재 여부·해시·허용 목록을 모아 `reports/provenance-diagnostics/<진단 잡번호>/summary.json`에 저장합니다.

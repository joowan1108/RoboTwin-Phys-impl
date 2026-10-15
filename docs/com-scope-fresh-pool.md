# CoM offset 적용 범위 수정과 v6 새 pool

**이 변경은 simulator source를 바꾸므로 `phys-pro6000-100-v5` pool과 저장된 progress를 이어받지 않는다.** 새 실험 이름 `phys-pro6000-100-v6`으로 전체 50개 task의 pool을 처음부터 준비한다. v5 자료는 삭제하지 않는다. 앞선 expert 수정([빈 집기점 v1](empty-grasp-recovery.md), [집기점 번호 v2](grasp-contact-recovery.md), [planner 실패 뒤 예외 v3](planner-failure-recovery.md))은 모두 포함한다.

## 확인한 문제

사용자의 별도 서버(RTX 2080 Ti 2장, Docker, 같은 `bundle-cu128-v5` 이미지에 v3 패치 적용)에서 50개 task를 seed 100000부터 각각 최대 5개 후보로 검사했다. 공식 안정성 검사(`check_stable`: 2000 step 정착 후 500 step 동안 자세 변화 3° 초과 여부)가 후보의 절반 이상을 `unstable_initialization`으로 제외했고, 제외율은 CoM offset 크기에 따라 올라갔다.

| 후보의 CoM offset 크기 | unstable 비율 (v3 snapshot, 148개 후보) | 수정 후 (v2 snapshot, 93개 후보) |
|---|---|---|
| 5 mm 미만 | 7 / 20 | 3 / 21 |
| 5–10 mm | 6 / 15 | 1 / 8 |
| 10–15 mm | 39 / 54 | 5 / 30 |
| 15 mm 초과 | 46 / 59 | 8 / 34 |

전체 unstable 비율은 66%에서 18%로 내려갔다. 같은 seed로 짝지은 88개 후보에서 수정 전 unstable이던 53개 중 24개는 verified, 16개는 expert 단계까지 진행해 `expert_failed`, 13개는 그대로 unstable이었다. 수정 전 verified였던 24개 중 21개는 그대로 verified였다. 5개 후보 안에 verified episode를 얻은 task는 36개에서 48개로 늘었고(남은 둘은 `beat_block_hammer`, `put_object_cabinet`으로 expert 실패가 원인), 남은 unstable 17건 중 7건은 세워진 task 병(001_bottle·114_bottle)으로 논문의 task 물체 CoM·질량 범위가 그대로 적용된 결과다. 두 실행 모두 crash는 없었다.

## 논문과의 대조

논문 Table 2의 CoM 범위([−1.5, 1.5] cm, x/y/z 독립)와 Table 3의 task별 축소(`beat_block_hammer`의 hammer처럼 얇은 물체는 ±2 mm)는 **조작 대상 물체**를 기준으로 서술되어 있으며, "전체 범위를 적용하면 CoM이 hammer 몸체 밖으로 나갈 수 있어 제외했다"고 설명한다. clutter 물체에 대한 언급은 없다. 기존 구현은 [로컬 선택](reproduction.md#미공개-사항에-대한-명시적-선택)으로 같은 episode offset을 clutter에도 적용했고, 그 결과 1.5 cm 이동이 펜·마커의 몸체 밖으로 CoM을 옮겨 장면 자체가 제외되었다. 이는 논문이 피하려 한 바로 그 상황을 task 물체가 아닌 배경 물체에서 재현한 것이며, 채택된 pool을 작은 CoM offset 쪽으로 치우치게 만든다.

## 수정 범위

`src/robotwin_phys/simulator.py`의 `_apply_physics`:

- **clutter에는 CoM offset을 적용하지 않는다.** task 물체는 종전과 같이 x/y/z 독립 offset을 받는다. 공식 `get_cluttered_table`은 반환 직전에 `self.cluttered_objs`를 비우므로, 우리 override가 그 호출 전후의 scene entity 집합 차이로 clutter(rigid actor와 urdf articulation link 모두)의 scene id를 기록해 둔다. 첫 구현은 비워진 목록을 읽어 clutter를 하나도 식별하지 못했고, 2080 Ti 서버의 paired 재실행에서 바로 드러나 수정했다.
- 같은 기준으로 **clutter는 질량 배율과 geometry scale도 받지 않고(명목 질량·크기), policy 단계 외력도 받지 않는다.** 논문 Table 2의 물체 속성(질량·CoM·크기)과 외력은 조작 대상 물체에 대한 서술이다. 특히 외력은 마찰이 0.01까지 내려간 episode에서 clutter를 rollout 내내 미끄러지게 했다. clutter의 pose는 종전과 같이 기울어진 table을 따른다.
- 마찰·반발 계수는 table·gripper와 공유하는 접촉 재질이므로 종전과 같이 모든 collision shape에 적용한다.
- 저장되는 `initial_state.physical.bodies`에 물체별 `clutter` 여부, 실제 적용한 `com_offset_m`·`mass_scale`, 외력 적용 여부 `external_force`를 기록한다.

task 코드, 샘플링 stream, 조건 파일(`configs/physical_domains.json`), planner, 평가 루프는 바꾸지 않았다. 사용자 결정(2026-10-14)으로 질량 배율·geometry scale·외력도 clutter에서 제외했다.

## OOD에서 실행

1. 실행 중인 v5 pool 잡이 있으면 종료를 확인한다(같은 노드 자원 외에는 v5 자료와 충돌하지 않는다).
2. [robotwin-fresh-pool-v2.zip](../reports/robotwin-fresh-pool-v2.zip)을 OOD Files의 `~/robotwin-data/patches/`에 업로드한다. 압축은 스크립트가 해제한다.
3. [ood-pool-pro6000-v6-fresh.txt](../cluster/ood-pool-pro6000-v6-fresh.txt)를 Job Composer에 붙여 넣어 제출한다. 기본값은 PRO 6000 2장, **GPU당 worker 8개(총 16 task 동시), CPU 64개**, 메모리 128G, 실험 `phys-pro6000-100-v6`이다. worker당 GPU 메모리 약 6 GB(96 GB 중 48 GB), CPU 4개를 기준으로 잡았다. 파티션이 CPU 64개를 거부하면 `--cpus-per-task=32`와 `APPTAINERENV_ROBOTWIN_FREEZE_WORKERS_PER_GPU=4`로 낮춘다. CUDA OOM이 나면 해당 task는 중단되지만 progress는 보존되므로 worker 수를 낮춰 같은 스크립트를 다시 제출하면 이어진다.

기대 로그:

```text
FRESH_POOL_V2_PATCH_SOURCE_PASS: reviewed simulator snapshot verified
FRESH_POOL_V2: CoM offset on task objects only; clutter keeps nominal CoM
FRESH_POOL: experiment=phys-pro6000-100-v6; all tasks; 100 episodes/task; start seed=100000; no older pool import
Pool preparation: 16 workers on 2 GPUs; 0/50 tasks already frozen
```

같은 v6 잡을 다시 제출하면 완료된 task와 `.progress.json`을 이어서 사용한다. 평가 launcher도 이 ZIP을 bind해야 provenance 검사를 통과한다.

## pool 생성 후 바로 FastWAM 평가 (같은 잡, PRO 6000 2장)

[ood-fastwam-pro6000x2-v6-all.txt](../cluster/ood-fastwam-pro6000x2-v6-all.txt)는 같은 ZIP으로 `phase=all`을 실행한다. 먼저 위와 동일하게 16 worker로 pool을 만들고 `pool-index.json`을 확정한 뒤, 같은 잡 안에서 FastWAM 평가를 시작한다. 평가 단계는 새 **dual layout**을 사용한다: 시뮬레이터와 FastWAM replica를 두 GPU에 번갈아 배치한다(replica i와 시뮬레이터 i는 같은 GPU). replica 하나가 시뮬레이터 하나를 담당하므로 `APPTAINERENV_ROBOTWIN_POLICY_REPLICAS=4`는 episode 4개 동시 진행, GPU당 replica 2개 + 시뮬레이터 2개를 뜻한다. 실측(job 1684, PRO 6000): FastWAM replica 하나가 로드 직후 약 23.9 GiB를 차지해 replica 4개를 한 GPU에 올리면 네 번째가 CUDA OOM으로 죽는다. 처음 배포(v2까지)는 replica를 전부 첫 GPU에 올렸고 이 실패로 v3에서 분산 배치로 바꿨다. GPU당 replica 2개(약 48 GiB) + 시뮬레이터 2개면 여유가 있고, 6 replica(GPU당 3개, 약 72 GiB)까지는 가능하다.

3일 제한으로 끝나면 **같은 스크립트를 다시 제출**한다. 완료된 pool과 검증된 episode를 재사용하고 남은 episode만 이어간다. pool만 먼저 만들고 싶으면 위의 `ood-pool-pro6000-v6-fresh.txt`를 쓰고, 이후 FastWAM은 이 스크립트를 제출하면 pool을 재사용한다. 다른 모델은 `APPTAINERENV_ROBOTWIN_MODEL`만 바꿔 같은 스크립트를 제출한다(Motus는 replica 2, FACT는 1).

scheduler 변경(`tools/parallel_eval.py`): GPU 2장 할당을 `dual` layout으로 받아들이고(종전에는 1장 또는 3장만 허용), replica 수를 1–8로 설정할 수 있다. 기존 1장(shared)·3장(split) layout과 결과 기록 형식은 바뀌지 않는다.

## 검증

- CPU 단위 검사(`tests/test_com_scope.py`): task 물체는 offset을 받고 clutter(rigid·articulated)는 명목 CoM을 유지하며, 질량 배율·재질은 양쪽 모두 적용되고 기록에 `clutter`/`com_offset_m`이 남는다.
- ZIP/launcher 검사(`tests/test_fresh_pool_launcher.py`): v6 launcher의 추출·체크섬·bind, 이전 ZIP 불변, v2 ZIP의 src/tools가 tree와 동일.
- 2080 Ti 서버 실측 비교: 같은 50개 task·같은 seed로 v3 snapshot과 v2 snapshot을 비교한 결과는 [reports/com-scope-validation.json](../reports/com-scope-validation.json)에 있다.

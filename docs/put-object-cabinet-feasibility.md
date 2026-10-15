# put_object_cabinet의 낮은 expert 채택률 (v6 pool, 2026-10-15)

잡 1681의 v6 pool에서 49개 task는 완료되었고 `put_object_cabinet`만 남아 후보 약 1,300개에서 24개 episode를 채택했다(채택률 약 2%). 결과 파일은 `plan_success: false`, `reason: expert_failed`로, 공식 CuRobo planner가 경로 계획 실패를 보고한 뒤 expert가 중단한 경우다. 공식 평가 루프가 seed를 건너뛰는 것과 같은 결과이며 오류가 아니다.

## 2080 Ti 서버 재현 (같은 snapshot, seed 101227부터)

| 설정 | 채택 / 후보 |
|---|---|
| Official Random (물리 randomization 없음) | 4 / 30 |
| Physical Random, table height 제외 | 2 / 30 |
| Physical Random, table tilt 제외 | 0 / 30 |
| Physical Random 전체 | 1 / 80 |

클러스터에서 제외된 seed 31개를 같은 snapshot으로 재실행하면 모두 같은 이유로 제외되어 GPU 차이가 아니다.

## 실패 위치

expert 동작을 추적한 결과(`reports/bs1-check/cabinet_diag.py`):

- seed 101227: 첫 동작인 **물체(stapler) 집기 계획**에서 실패. Official Random에서도 같은 seed가 같은 위치에서 실패하므로 공식 expert 자체의 한계다.
- seed 101228: 물체 집기는 성공하고 **반대쪽 팔의 서랍 손잡이 집기 계획**에서 실패. Official Random에서는 성공하며, 차이는 table height −7.8 cm로 손잡이 목표가 약 5 cm 낮아진 것뿐이다. 질량·CoM·마찰·damping은 경로 계획에 영향을 주지 않는다.

즉 이 task는 공식 scripted expert가 Official Random에서도 약 13%만 성공하는 어려운 task이고, 논문 Table 2의 table height [−8, 0] cm가 낮은 손잡이 위치에서 계획 실패를 더 늘린다. 논문 Table 3는 이 task에서 size와 arm damping만 고정하고 table height는 전역 범위를 유지하므로, 현재 구현은 논문 설정과 일치한다. 채택된 episode는 table height가 얕은 쪽으로 치우치며, 이는 논문의 feasibility filter("task-feasible operating conditions")가 갖는 성질이다.

## 조치

- 코드·domain 변경 없음. 후보 예산 10,000개 안에서 완료된다(예상 총 후보 약 5,000개, 후보당 20–40 s).
- 잡 1681은 이 task가 끝난 뒤 pool을 확정하고 FastWAM 평가로 넘어간다. 3일 제한에 걸리면 같은 스크립트를 다시 제출한다.
- 표에 보고할 때 이 task의 채택률과 table height 치우침을 함께 기록한다. 채택된 조건 분포는 `manifests/phys-pro6000-100-v6/physical_random/put_object_cabinet.json`의 `condition.table_height_offset_m`에서 바로 계산할 수 있다.

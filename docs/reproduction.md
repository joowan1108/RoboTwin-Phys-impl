# 원문·공식 코드 대조 기록

확인일: 2026-10-08. 기준 논문: [RoboTwin-Phys v1](https://arxiv.org/html/2609.26292v1), [PDF](https://arxiv.org/pdf/2609.26292v1). 참고한 공식 안내는 [RoboTwin-Phys 저장소](https://github.com/yefeng00/RoboTwin-Phys), [프로젝트](https://yefeng00.github.io/RoboTwinPhysTest/), [RoboTwin 설치 문서](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)입니다.

## 무엇이 공개되어 있는가

공식 Phys repository revision `c629e7f9d244ce862482847c17a8b9b0921355f3`에는 소개 페이지와 README만 있고 실행 환경·설정 파일은 추후 공개 예정입니다. dataset 다운로드도 TBD입니다. README 사본은 `references/official-phys-README.md`에 저장했습니다.

v1 원문은 전체 결과와 task별 결과, 부록 A의 전역 범위, 부록 B의 9개 예외를 제공합니다. 별도의 factor별 ablation 결과표나 ablation 전용 수치 범위는 확인되지 않았습니다. 따라서 임의의 추가 범위를 논문 ablation 범위로 만들지 않았습니다. `physical_only`, single-factor, leave-one-factor-out 기능은 동일한 부록 범위를 사용하는 자체 분석 옵션입니다.

현재 RoboTwin main의 새 policy 시스템 대신 공식 `stable_2.0` revision `13c3c47ff4312dd62484bcd51be034af55c062d1`을 고정했습니다. 이 revision의 설치 파일은 `scripts/`가 아닌 `script/` 아래에 있습니다. CuRobo는 해당 설치 스크립트의 v0.7.8, revision `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`를 사용합니다. 이는 Phys 저자가 사용한 정확한 미공개 commit을 알아냈다는 의미는 아닙니다.

## 부록 A: 전역 범위

기계가 읽는 원본은 [configs/physical_domains.json](../configs/physical_domains.json)입니다. 아래 수치는 부록 A Table 2의 사실 정보입니다.

| 속성 | 범위 | 적용 |
|---|---|---|
| 질량 | 0.125–4 배 | object mass와 inertia에 같은 배율 |
| CoM | 축별 −0.015–0.015 m | x/y/z 독립 offset |
| 형상 scale | 0.85–1.05 배 | collision·visual·grasp metadata 함께 변경 |
| table tilt | 0–2° | azimuth 90° |
| table height | −0.08–0 m | 논문의 nominal 0.74 m 기준 |
| external force | 0–0.1 mg | +X, policy rollout의 각 physics step |
| camera | orbit 0–0.10 m, translation 0–0.03 m, rotation 0–10° | 세 mode 중 하나 |
| friction | 0.01–1.25 | static/dynamic friction |
| restitution | 0–0.5 | collision material |
| object joint damping | 0–25 | articulated object active joint |
| arm/gripper damping | 0.5–3 배 | 두 종류에 동일한 한 배율 |

논문은 13개 physical attribute라고 부르지만 카메라와 arm/gripper를 묶어 표의 11개 행으로 제시하고 있습니다. 카메라 mode와 robot damping을 묶어 ablation factor family를 11개로 제공합니다. `com_offset_m`는 3차원이며 arm과 gripper는 원문대로 같은 배율을 공유합니다. 13개가 모두 독립 scalar라는 뜻으로 구현하지 않았습니다.

## 부록 B: 9개 민감 task

| task | 전역 설정에서 바뀌는 값 |
|---|---|
| dump_bin_bigbin | CoM ±1 mm |
| grab_roller | CoM ±2 mm |
| beat_block_hammer | CoM ±2 mm |
| put_bottles_dustbin | CoM ±2 mm |
| open_microwave | object damping 0–0.25 |
| turn_switch | object damping 0–1 |
| put_object_cabinet | geometry 1, arm/gripper damping 1 |
| scan_object | 카메라 세 magnitude 절반, CoM ±2 mm |
| place_bread_skillet | 카메라 magnitude 절반, CoM ±1 mm, geometry 0.9–1.05, tilt 0–1° |

명시되지 않은 값은 전역 범위를 유지합니다. task 이름은 공식 RoboTwin `_eval_step_limit.yml`의 50개 registry에서 읽습니다.

## seed와 평가 공정성

논문 §2.2는 episode 초기화 시 sampling하고 rollout 동안 고정하며 seed로 재현한다고 설명합니다. §2.4는 expert가 완료하지 못한 조건을 평가 pool에서 제외합니다. §4.1과 부록 D는 Clean/Official Random/Physical Random과 task당 100회 평가를 제공합니다. Clean/Official Random의 논문 수치는 각 모델의 기존 공개 결과에서 가져온 값이므로 동일 seed로 세 protocol을 저자가 모두 재평가했다는 근거는 없습니다.

[고정한 RoboTwin eval 코드](https://github.com/RoboTwin-Platform/RoboTwin/blob/13c3c47ff4312dd62484bcd51be034af55c062d1/script/eval_policy.py)는 run seed를 `100000 * (1 + seed)`로 바꾸고 candidate seed를 증가시키며 expert 통과 100개를 평가합니다. 같은 seed로 expert scene과 policy scene을 각각 초기화합니다. 하지만 일부 Python `random` 사용, runtime policy RNG 공유, 매번 다시 실행하는 expert filtering으로 인해 seed 하나만 저장하는 것은 충분하지 않습니다.

이 구현은 다음을 추가합니다.

1. policy를 연결하기 전에 expert pool을 확정합니다. 환경 오류는 제외 사유로 삼지 않습니다.
2. episode seed, task, parameter 이름에서 SHA-256로 분리한 PCG64 stream을 사용합니다. factor를 끄더라도 다른 factor의 값은 바뀌지 않습니다. Python의 process별 `hash()`를 쓰지 않습니다.
3. visual/layout의 NumPy·Python RNG와 policy 프로세스의 RNG를 분리합니다. 동적 조명은 episode seed와 simulation step으로 고정합니다.
4. instruction도 한 번 생성해 그대로 저장합니다. policy가 manifest의 물리 GT를 observation으로 받지는 않습니다.
5. 매 episode를 새 프로세스에서 시작하고 초기 scene을 manifest와 비교합니다. state float 비교 tolerance는 absolute 2e-6, relative 1e-6입니다.
6. frozen JSON의 SHA-256과 소스·asset 내용·패키지·GPU/driver fingerprint를 검사합니다. model run은 manifest hash와 server identity를 기록합니다.
7. policy 실패를 이유로 seed를 교체하거나 expert pool을 다시 만들지 않습니다. incomplete/error run을 100-episode 성공률로 표시하지 않습니다.

같은 조건을 보장하는 범위는 같은 task/protocol/ablation/manifest입니다. 조건을 바꾸는 ablation 사이에는 physics가 의도적으로 달라집니다. 다른 protocol에서 시작 seed는 같아도 expert filtering 결과가 다를 수 있으므로 paired analysis를 하려면 공통 feasible seed 교집합을 별도로 구성해야 합니다. 서로 다른 hardware/PhysX/정밀도에서도 전체 trajectory가 bitwise 일치한다고 보장하지 않습니다.

## 미공개 사항에 대한 명시적 선택

아래는 공식 구현으로 검증되지 않은 **로컬 구현 선택**입니다.

- 연속 범위는 uniform, 카메라 mode는 균등 확률, 방향/회전축은 isotropic으로 sampling합니다. 전역 episode factor는 task object에 적용합니다. 공식 clutter object는 **명목 질량·CoM·크기를 유지하고 policy 단계 외력도 받지 않습니다**. table·gripper와 공유하는 접촉 재질(마찰·반발 계수)만 clutter에도 적용합니다. 논문의 CoM 범위와 task별 축소(hammer 등 얇은 물체)는 조작 대상 물체를 기준으로 서술되며, 1.5 cm 이동은 펜·마커 같은 얇은 clutter의 몸체 밖으로 CoM을 옮겨 공식 안정성 검사가 장면 자체를 제외시켰습니다([v6 pool 안내](com-scope-fresh-pool.md)). 저자의 정확한 RNG algorithm/stream/seed 목록은 미공개입니다.
- tilt는 table surface normal이 +Y를 향하도록 X축 음의 회전으로 구현합니다. table 중심을 기준으로 초기 object pose를 함께 회전하고 scale에 따라 tabletop 위의 높이를 보정합니다. robot root는 움직이지 않습니다.
- geometry는 load 전에 collision/visual 및 grasp/functional metadata에 적용합니다. mass는 원래 task가 설정한 값에 배율을 곱하고, geometry 처리 후 계산된 inertia에도 같은 mass 배율을 곱합니다. robot 링크의 mass/CoM는 object randomization 대상이 아닙니다.
- friction/restitution은 object·table·gripper 등 collision material에 적용합니다. object joint damping은 drive damping으로 해석하며 stiffness와 force limit은 보존합니다.
- physical table height를 켜면 원래 `random_table_height`/task bias와 더하지 않고 0.74 m에 −8–0 cm를 적용합니다. 이로써 표의 전역 범위를 넘는 이중 randomization을 피합니다. `place_dual_shoes`의 upstream −0.1 m bias도 이 선택의 영향을 받습니다. 저자 코드 공개 후 가장 먼저 대조할 항목입니다.
- head camera만 변경합니다. orbit은 원래 optical ray와 tabletop의 교차점을 gaze target으로 삼아 radius를 유지하고 sampled chord distance만큼 이동합니다. translation은 gaze를 보정하지 않고 rotation은 제자리 회전입니다. 다른 camera들은 원래 설정을 유지합니다.
- Table 2가 external force를 policy stage로 명시하므로 expert/settling에는 force를 켜지 않고 policy stage에서만 +X 힘을 매 simulation step 적용합니다. 따라서 expert 성공은 force가 켜진 rollout의 성공 보장이 아닙니다. force timing과 feasibility의 정확한 저자 구현은 공개 후 확인해야 합니다.
- 공식 랜덤 배경, clutter, 광원은 유지하지만 repeated render call이 조명을 바꾸던 upstream 동작은 simulation-time 기반으로 보완했습니다. 이는 공정성 개선을 위한 로컬 패치이므로 stock RoboTwin의 모든 픽셀과 같다고 주장하지 않습니다.

upstream 변경은 [references/robotwin-local.patch](../references/robotwin-local.patch)에 남습니다. SAPIEN URDF encoding/.srdf와 MPLib screw planner 수정은 RoboTwin 공식 설치 안내에 따른 환경 내부 변경입니다.

## 현재 머신 호환성

Ubuntu 22.04.5, RTX 2080 Ti ×2 (sm_75), driver 580.178.04. `nvidia-smi`의 CUDA 13.0은 driver 지원 상한이며 이 프로젝트의 Toolkit/PyTorch runtime은 12.1입니다. [NVIDIA backward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)에 따라 최신 driver에서 CUDA 12.x 실행이 가능합니다.

CUDA kernel은 `TORCH_CUDA_ARCH_LIST=7.5`로 빌드했습니다. PhysX CPU simulation, CUDA CuRobo planning, Vulkan RTX rendering을 사용합니다. 화면이 필요 없는 renderer도 실제 NVIDIA Vulkan device를 명시합니다. RTX 2080 Ti는 native BF16 Tensor Core와 FlashAttention-2의 공식 지원 대상이 아닙니다. 이 Torch 버전의 `is_bf16_supported()`가 True를 반환해도 sm_75 native BF16 가속을 의미하지 않아 별도로 capability를 검사합니다. [NVIDIA Turing 문서](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html), [FlashAttention 지원표](https://github.com/Dao-AILab/flash-attention#nvidia-cuda-support).

두 장의 11GB GPU가 자동으로 하나의 22GB GPU가 되지 않습니다. 모델을 임의로 양자화하거나 정밀도를 바꾼 결과를 논문의 official-precision 결과와 같은 것으로 취급하지 않습니다.

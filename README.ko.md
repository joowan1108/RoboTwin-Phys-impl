<div align="center">

# RoboTwin-Phys · independent reimplementation

**논문을 바탕으로 다시 구현한 RoboTwin 2.0 물리 랜덤화 평가**

[![paper](https://img.shields.io/badge/arXiv-2609.26292-b31b1b.svg)](https://arxiv.org/abs/2609.26292)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![tests](https://github.com/joowan1108/RoboTwin-Phys-impl/actions/workflows/tests.yml/badge.svg)](https://github.com/joowan1108/RoboTwin-Phys-impl/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.10-3776ab.svg)

[구현 내용](#-구현-내용) · [결과](#-결과) · [우리 seed로 벤치마크하기](#-우리-seed로-벤치마크하기) · [데이터](data/README.md) · [English](README.md)

</div>

> [!NOTE]
> 작업을 시작한 2026-10-08 기준으로 [RoboTwin-Phys](https://arxiv.org/abs/2609.26292) 저자들은 코드, task 설정, seed 목록을 공개하지 않았습니다. 이 저장소의 모든 것은 논문을 바탕으로 다시 구현한 것이며 공식 구현이 아닙니다.

## ✨ 구현 내용

RoboTwin-Phys는 RoboTwin 2.0의 50개 양팔 조작 task에서 episode마다 물리 조건을 바꿔 가며 policy를 평가합니다. 이 저장소는 그 프로토콜 전체를 구현합니다.

- **물리 랜덤화.** 논문 Table 2의 11개 속성을 episode마다 seed에서 샘플링합니다. 민감한 9개 task는 Appendix B의 범위를 씁니다.

  | 속성 | 범위 | 적용 대상 |
  |---|---|---|
  | 물체 질량 | 0.125 ~ 4 × (관성도 함께) | task 물체 |
  | 질량 중심 | 축마다 ±0.015 m | task 물체 |
  | 크기 배율 | 0.85 ~ 1.05 × | task 물체 |
  | 테이블 기울기 | 0 ~ 2° | 테이블 |
  | 테이블 높이 | −0.08 ~ 0 m | 테이블 |
  | 외력 | 0 ~ 0.1 mg, policy 실행 중에만 | task 물체 |
  | 카메라 | 궤도 0–0.10 m, 이동 0–0.03 m 또는 회전 0–10° | head 카메라 |
  | 마찰 | 0.01 ~ 1.25 | 모든 재질 |
  | 반발 계수 | 0 ~ 0.5 | 모든 재질 |
  | 물체 관절 damping | 0 ~ 25 | 관절 물체 |
  | 팔·그리퍼 damping | 0.5 ~ 3 × | 로봇 |

- **expert로 거른 episode pool.**
  - task마다 seed 100000부터 차례로 올라가며 검사합니다.
  - seed마다 샘플링된 물리 조건에서 공식 RoboTwin 2.0 expert를 실행하고, expert가 성공한 seed만 남깁니다.
  - 100개가 채워지면 멈추고, 모든 policy를 정확히 이 100개 episode로 평가합니다.
- **policy 평가.** 각 policy는 자기 환경에서 서버로 실행되며, 공식 RoboTwin adapter(`get_model / eval / reset_model`)를 그대로 씁니다. policy 실패는 항상 분모에 남고 seed를 교체하지 않습니다.
- **논문의 baseline 5종.** Fast-WAM, G0.5, π0.5, Motus, FACT. checkpoint와 추론 설정은 [`configs/policies`](configs/policies)에 고정했습니다.
- **공개 pool.** 채택된 seed 5000개, 각 seed의 물리 조건, Fast-WAM의 모든 episode 결과가 [`data/`](data/README.md)에 있습니다.

논문에 명시되지 않은 부분의 선택은 [`docs/design.md`](docs/design.md)에 기록했습니다. 주요 선택은 두 가지입니다.
- 질량, 질량 중심, 크기, 외력은 task 물체에만 적용합니다.
- 테이블 높이는 upstream의 랜덤 테이블 높이를 대체합니다.

## 📊 결과

Physical Random, 50 task × 100 episode, task별 성공률의 평균(%).

| 모델 | 이 저장소 | 논문 (Table 1) |
|---|---:|---:|
| **Fast-WAM** | **45.18** | 44.24 |
| **G0.5** | **38.36** | 37.82 |

두 모델 모두 논문의 전체 성공률을 1%p 이내로 재현했습니다. π0.5, Motus, FACT의 adapter도 들어 있지만([Baseline](#-baseline) 참고) 이번 공개에서는 평가하지 않았습니다.

환경: NVIDIA RTX PRO 6000 Blackwell 2장, RoboTwin 2.0 `stable_2.0 @ 13c3c47`. checkpoint: Fast-WAM `yuanty/fastwam @ 8eaceeb`(`sigma_shift=5.0`), G0.5 `OpenGalaxea/G05 @ e312be81e90c`.

### 논문과 얼마나 비슷한가

| 지표 | Fast-WAM | G0.5 |
|---|---:|---:|
| 전체 성공률 (우리 / 논문) | 45.18 / 44.24 (Δ +0.94) | 38.36 / 37.82 (Δ +0.54) |
| task별 성공률 50개의 Pearson 상관 | 0.84 | 0.84 |
| Spearman 순위 상관 | 0.85 | 0.82 |
| task별 차이의 평균 절댓값 | 8.3%p | 6.9%p |
| task별 차이의 중앙값 | 6.0%p | 5.5%p |
| ±10%p 안에 든 task | 35 / 50 | 41 / 50 |
| ±15%p 안에 든 task | 44 / 50 | 45 / 50 |
| p < 0.05로 유의하게 다른 task (양측 검정) | 9 / 50 | 7 / 50 |
| Bonferroni 보정 뒤에도 유의한 task (p < 0.001) | 3 / 50 | 1 / 50 |

양쪽 모두 task당 100 episode로 추정한 값이라 task 하나에 약 ±10%p의 표본 잡음이 있고, 50개 중 2.5개 정도는 우연히 p < 0.05가 나옵니다. 보정 뒤에도 차이가 남는 task는 다음과 같습니다.

| task | Fast-WAM 우리 | Fast-WAM 논문 | G0.5 우리 | G0.5 논문 |
|---|---:|---:|---:|---:|
| `grab_roller` | **86** | **65** | 82 | 72 |
| `place_fan` | **64** | **38** | 54 | 32 |
| `place_dual_shoes` | **6** | **28** | **3** | **23** |

`place_dual_shoes`는 두 policy 모두 논문보다 크게 낮고, `place_fan`은 두 policy 모두 높습니다(G0.5는 p = 0.002). policy 문제보다는 환경 쪽의 공통 원인일 가능성이 큽니다. `place_dual_shoes`의 후보 하나는 [docs/design.md](docs/design.md#scope-decisions-where-the-paper-is-silent)의 테이블 높이 결정입니다. 이 결정은 이 task의 upstream −0.1 m 테이블 높이 보정도 대체합니다.

<details>
<summary><b>Fast-WAM task별 비교 (50개)</b> 🟩 |Δ| ≤ 10 · 🟨 ≤ 20 · 🟥 > 20 · ❗ p &lt; 0.05 · ‼️ p &lt; 0.001</summary>

| Task | Ours | Paper | Δ | |
|---|---:|---:|---:|:---:|
| `adjust_bottle` | 64 | 64 | +0 | 🟩 |
| `beat_block_hammer` | 11 | 17 | -6 | 🟩 |
| `blocks_ranking_rgb` | 51 | 57 | -6 | 🟩 |
| `blocks_ranking_size` | 43 | 56 | -13 | 🟨 |
| `click_alarmclock` | 54 | 69 | -15 | 🟨❗ |
| `click_bell` | 45 | 57 | -12 | 🟨 |
| `dump_bin_bigbin` | 38 | 32 | +6 | 🟩 |
| `grab_roller` | 86 | 65 | +21 | 🟥‼️ |
| `handover_block` | 30 | 33 | -3 | 🟩 |
| `handover_mic` | 62 | 54 | +8 | 🟩 |
| `hanging_mug` | 22 | 26 | -4 | 🟩 |
| `lift_pot` | 37 | 29 | +8 | 🟩 |
| `move_can_pot` | 42 | 50 | -8 | 🟩 |
| `move_pillbottle_pad` | 41 | 43 | -2 | 🟩 |
| `move_playingcard_away` | 54 | 59 | -5 | 🟩 |
| `move_stapler_pad` | 22 | 17 | +5 | 🟩 |
| `open_laptop` | 64 | 59 | +5 | 🟩 |
| `open_microwave` | 31 | 17 | +14 | 🟨❗ |
| `pick_diverse_bottles` | 15 | 17 | -2 | 🟩 |
| `pick_dual_bottles` | 17 | 33 | -16 | 🟨❗ |
| `place_a2b_left` | 61 | 54 | +7 | 🟩 |
| `place_a2b_right` | 54 | 49 | +5 | 🟩 |
| `place_bread_basket` | 61 | 50 | +11 | 🟨 |
| `place_bread_skillet` | 59 | 39 | +20 | 🟨❗ |
| `place_burger_fries` | 48 | 53 | -5 | 🟩 |
| `place_can_basket` | 26 | 17 | +9 | 🟩 |
| `place_cans_plasticbox` | 33 | 28 | +5 | 🟩 |
| `place_container_plate` | 68 | 66 | +2 | 🟩 |
| `place_dual_shoes` | 6 | 28 | -22 | 🟥‼️ |
| `place_empty_cup` | 46 | 57 | -11 | 🟨 |
| `place_fan` | 64 | 38 | +26 | 🟥‼️ |
| `place_mouse_pad` | 31 | 35 | -4 | 🟩 |
| `place_object_basket` | 35 | 41 | -6 | 🟩 |
| `place_object_scale` | 37 | 46 | -9 | 🟩 |
| `place_object_stand` | 58 | 53 | +5 | 🟩 |
| `place_phone_stand` | 41 | 52 | -11 | 🟨 |
| `place_shoe` | 62 | 63 | -1 | 🟩 |
| `press_stapler` | 64 | 58 | +6 | 🟩 |
| `put_bottles_dustbin` | 9 | 11 | -2 | 🟩 |
| `put_object_cabinet` | 42 | 39 | +3 | 🟩 |
| `rotate_qrcode` | 45 | 43 | +2 | 🟩 |
| `scan_object` | 44 | 29 | +15 | 🟨❗ |
| `shake_bottle` | 81 | 69 | +12 | 🟨 |
| `shake_bottle_horizontally` | 81 | 75 | +6 | 🟩 |
| `stack_blocks_three` | 40 | 30 | +10 | 🟩 |
| `stack_blocks_two` | 53 | 63 | -10 | 🟩 |
| `stack_bowls_three` | 42 | 52 | -10 | 🟩 |
| `stack_bowls_two` | 65 | 62 | +3 | 🟩 |
| `stamp_seal` | 24 | 24 | +0 | 🟩 |
| `turn_switch` | 50 | 34 | +16 | 🟨❗ |

논문 값: arXiv 2609.26292v1 Table 4 (Appendix D). 우리 값: [`data/physical-random-100/results/fastwam/episodes.csv`](data/physical-random-100/results/fastwam/episodes.csv).

</details>

<details>
<summary><b>G0.5 task별 비교 (50개)</b> 🟩 |Δ| ≤ 10 · 🟨 ≤ 20 · 🟥 > 20 · ❗ p &lt; 0.05 · ‼️ p &lt; 0.001</summary>

| Task | Ours | Paper | Δ | |
|---|---:|---:|---:|:---:|
| `adjust_bottle` | 45 | 65 | -20 | 🟨❗ |
| `beat_block_hammer` | 6 | 8 | -2 | 🟩 |
| `blocks_ranking_rgb` | 43 | 36 | +7 | 🟩 |
| `blocks_ranking_size` | 33 | 44 | -11 | 🟨 |
| `click_alarmclock` | 55 | 59 | -4 | 🟩 |
| `click_bell` | 36 | 36 | +0 | 🟩 |
| `dump_bin_bigbin` | 38 | 24 | +14 | 🟨❗ |
| `grab_roller` | 82 | 72 | +10 | 🟩 |
| `handover_block` | 24 | 23 | +1 | 🟩 |
| `handover_mic` | 53 | 43 | +10 | 🟩 |
| `hanging_mug` | 22 | 15 | +7 | 🟩 |
| `lift_pot` | 32 | 30 | +2 | 🟩 |
| `move_can_pot` | 31 | 34 | -3 | 🟩 |
| `move_pillbottle_pad` | 30 | 26 | +4 | 🟩 |
| `move_playingcard_away` | 37 | 43 | -6 | 🟩 |
| `move_stapler_pad` | 16 | 23 | -7 | 🟩 |
| `open_laptop` | 55 | 51 | +4 | 🟩 |
| `open_microwave` | 47 | 33 | +14 | 🟨❗ |
| `pick_diverse_bottles` | 17 | 27 | -10 | 🟩 |
| `pick_dual_bottles` | 17 | 35 | -18 | 🟨❗ |
| `place_a2b_left` | 44 | 51 | -7 | 🟩 |
| `place_a2b_right` | 49 | 57 | -8 | 🟩 |
| `place_bread_basket` | 50 | 50 | +0 | 🟩 |
| `place_bread_skillet` | 47 | 44 | +3 | 🟩 |
| `place_burger_fries` | 31 | 38 | -7 | 🟩 |
| `place_can_basket` | 30 | 20 | +10 | 🟩 |
| `place_cans_plasticbox` | 31 | 15 | +16 | 🟨❗ |
| `place_container_plate` | 47 | 48 | -1 | 🟩 |
| `place_dual_shoes` | 3 | 23 | -20 | 🟨‼️ |
| `place_empty_cup` | 42 | 35 | +7 | 🟩 |
| `place_fan` | 54 | 32 | +22 | 🟥❗ |
| `place_mouse_pad` | 23 | 22 | +1 | 🟩 |
| `place_object_basket` | 27 | 31 | -4 | 🟩 |
| `place_object_scale` | 33 | 34 | -1 | 🟩 |
| `place_object_stand` | 44 | 40 | +4 | 🟩 |
| `place_phone_stand` | 33 | 37 | -4 | 🟩 |
| `place_shoe` | 55 | 44 | +11 | 🟨 |
| `press_stapler` | 50 | 42 | +8 | 🟩 |
| `put_bottles_dustbin` | 39 | 38 | +1 | 🟩 |
| `put_object_cabinet` | 43 | 42 | +1 | 🟩 |
| `rotate_qrcode` | 37 | 33 | +4 | 🟩 |
| `scan_object` | 34 | 39 | -5 | 🟩 |
| `shake_bottle` | 76 | 71 | +5 | 🟩 |
| `shake_bottle_horizontally` | 79 | 76 | +3 | 🟩 |
| `stack_blocks_three` | 32 | 29 | +3 | 🟩 |
| `stack_blocks_two` | 45 | 50 | -5 | 🟩 |
| `stack_bowls_three` | 23 | 32 | -9 | 🟩 |
| `stack_bowls_two` | 41 | 36 | +5 | 🟩 |
| `stamp_seal` | 24 | 15 | +9 | 🟩 |
| `turn_switch` | 33 | 40 | -7 | 🟩 |

논문 값: arXiv 2609.26292v1 Table 4 (Appendix D)의 Galaxea-VLA 열. 우리 값: Fast-WAM과 같은 5000개 episode.

</details>

## 🎯 우리 seed로 벤치마크하기

우리 pool은 [`data/physical-random-100`](data/physical-random-100/)에 있습니다.

| 파일 | 내용 |
|---|---|
| `pool/seeds.json` | task별: episode 순서대로 채택된 seed 100개와, 탈락한 seed 전체 및 탈락 이유 |
| `pool/conditions.csv` | episode별: 샘플링된 물리 속성 11개와 언어 지시문 |
| `results/fastwam/episodes.csv` | episode별: Fast-WAM 성공 여부, action 수, 실행 시간 |

episode의 물리 조건은 `(task, seed)`만으로 정해지므로 어느 머신에서나 같습니다. 같은 5000개 episode로 내 policy를 평가하는 순서는 다음과 같습니다.

**1. simulator 준비.** 저장소를 clone하고 Docker 이미지를 빌드합니다. 이미지에는 고정된 revision의 RoboTwin 2.0, SAPIEN, CuRobo, 공식 assets가 들어갑니다([docs/setup.md](docs/setup.md)). 그다음 조건이 재현되는지 확인합니다.

```bash
git clone https://github.com/joowan1108/RoboTwin-Phys-impl.git && cd RoboTwin-Phys-impl
bash tools/docker.sh build
bash tools/docker.sh doctor      # GPU, Vulkan 렌더링, CUDA 확인
bash tools/docker.sh shell       # 이후 명령은 모두 이 컨테이너 안에서 실행
python3 analysis/verify_release.py data/physical-random-100   # 5000개 조건을 seed에서 다시 계산해 확인
```

**2. 같은 시작 seed로 pool 고정.** expert가 내 머신에서 seed를 하나씩 다시 검사하고 초기 장면을 저장합니다.

```bash
for task in $(robotwin-phys tasks); do
  robotwin-phys freeze --task $task --protocol physical_random --episodes 100 --seed 100000 \
    --output manifests/$task.json
done
```

채택된 seed가 우리와 같은지 확인합니다.

```bash
python3 - <<'EOF'
import json
ours = json.load(open('data/physical-random-100/pool/seeds.json'))['tasks']
for task, entry in ours.items():
    mine = [e['seed'] for e in json.load(open(f'manifests/{task}.json'))['episodes']]
    print(task, 'OK' if mine == entry['accepted'] else 'DIFFERS')
EOF
```

expert는 seed마다 결정적이지만, GPU나 드라이버가 다르면 경계에 있는 seed 몇 개가 바뀔 수 있습니다. 다르다면 결과와 함께 알려 주세요.

**3. policy 감싸기.** 공식 RoboTwin policy와 같은 함수 세 개로 adapter를 만들고, 이를 가리키는 profile을 작성합니다.

```python
# my_policy/deploy_policy.py
def get_model(args):               # args = profile의 "args" 블록
    return load_my_model(args["ckpt"])

def reset_model(model):            # episode 시작마다 호출
    model.reset()

def eval(env, model, observation): # episode가 끝날 때까지 반복 호출
    for action in model.predict(observation):
        env.take_action(action)
```

```yaml
# configs/policies/my_policy.yaml
name: my_policy
adapter: my_policy/deploy_policy.py
python_paths: [my_policy]
args:
  ckpt: /path/to/checkpoint
```

[`src/robotwin_phys/noop_policy.py`](src/robotwin_phys/noop_policy.py)가 가장 짧은 동작 예시이고, [`configs/policies`](configs/policies)에서 baseline 5종의 profile을 볼 수 있습니다.

**4. 평가.** policy의 Python 환경에서 policy 서버를 띄운 뒤 50개 task를 실행합니다.

```bash
PYTHONPATH=$PWD/src /path/to/my-policy-env/bin/python -m robotwin_phys.policy_server \
  --config configs/policies/my_policy.yaml --port 9000 &

for task in $(robotwin-phys tasks); do
  robotwin-phys evaluate --manifest manifests/$task.json --output runs/my_policy/$task --policy-port 9000
done
```

task 폴더마다 두 종류의 결과가 생깁니다.
- `runs/my_policy/<task>/summary.json`에는 task의 `success_rate`가 들어갑니다.
- `episode-NNNN.result.json`에는 seed별 결과가 들어갑니다.

episode별 MP4가 필요하면 `--video`를 붙이세요.

**5. 비교.** 벤치마크 점수는 task 50개 성공률의 평균입니다. 두 가지로 비교할 수 있습니다.
- episode 단위로 `results/fastwam/episodes.csv`의 Fast-WAM과 비교합니다.
- task 단위로 [`references/paper-table4-physical-random.json`](references/paper-table4-physical-random.json)의 논문 값과 비교합니다.

<details>
<summary><b>우리와 완전히 같은 환경에서 실행하기</b></summary>

Release 파일 [`physical-random-100-full.zip`](https://github.com/joowan1108/RoboTwin-Phys-impl/releases/tag/data-physical-random-100)에는 검증된 초기 장면을 포함한 manifest 50개가 들어 있습니다. `robotwin-phys evaluate`는 소스, assets, 패키지 버전, GPU 모델, 드라이버가 모두 같을 때만 이 manifest를 받아들입니다. 다른 머신에서는 2단계를 따르세요.

</details>

## 🤖 Baseline

| 모델 | Checkpoint | Adapter |
|---|---|---|
| Fast-WAM | `yuanty/fastwam` | 공식 FastWAM RoboTwin client |
| G0.5 | `OpenGalaxea/G05` (접근 승인 필요) | 공식 GalaxeaVLA RoboTwin adapter |
| π0.5 | `motus-robotics/pi0.5_robotwin2` | OpenPI PyTorch |
| Motus | `motus-robotics/Motus_robotwin2` | 공식 Motus adapter |
| FACT | `Bariona/fact-wam` | FACT 자체 서버와 공식 client |

이번 공개에서 평가한 모델은 Fast-WAM과 G0.5입니다. π0.5, Motus, FACT adapter는 구현과 CPU 테스트까지 되어 있지만 공개 결과는 없습니다. revision과 추론 인자는 [`configs/policies`](configs/policies)에, 준비 방법은 [`docs/policies.md`](docs/policies.md)에 있습니다. bf16 policy는 Ampere 이상 GPU가 필요합니다. 설계 결정과 구현 세부 사항은 [`docs/design.md`](docs/design.md)에 있습니다.

## 📄 라이선스와 인용

코드와 공개 데이터는 [MIT License](LICENSE)로 배포합니다. RoboTwin 2.0, CuRobo, 각 policy 저장소, 모델 checkpoint는 각자의 라이선스를 따릅니다.

이 재구현을 사용한다면 원 논문을 인용해 주세요.

```bibtex
@article{robotwinphys2026,
  title   = {RoboTwin-Phys},
  journal = {arXiv preprint arXiv:2609.26292},
  year    = {2026}
}
```

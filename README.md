<div align="center">

# RoboTwin-Phys · independent reimplementation

**Physics-randomized evaluation for RoboTwin 2.0, rebuilt from the paper.**

[![paper](https://img.shields.io/badge/arXiv-2609.26292-b31b1b.svg)](https://arxiv.org/abs/2609.26292)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![tests](https://github.com/joowan1108/RoboTwin-Phys-impl/actions/workflows/tests.yml/badge.svg)](https://github.com/joowan1108/RoboTwin-Phys-impl/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.10-3776ab.svg)

[What is implemented](#-what-is-implemented) · [Results](#-results) · [Benchmark your policy with our seeds](#-benchmark-your-policy-with-our-seeds) · [Data](data/README.md) · [한국어](README.ko.md)

</div>

> [!NOTE]
> The authors of [RoboTwin-Phys](https://arxiv.org/abs/2609.26292) had not released code, task settings or seed lists when this work started (2026-10-08). Everything here is reimplemented from the paper. It is not the official implementation.

## ✨ What is implemented

RoboTwin-Phys evaluates bimanual manipulation policies on the 50 RoboTwin 2.0 tasks while the physics of every episode is randomized. This repository implements the full protocol:

- **Physical randomization.** All 11 attributes of the paper's Table 2 are sampled per episode from the seed. The nine sensitive tasks use the Appendix B overrides.

  | Attribute | Range | Applies to |
  |---|---|---|
  | Object mass | 0.125 to 4 × (inertia scaled too) | task objects |
  | Center of mass | ±0.015 m per axis | task objects |
  | Geometry scale | 0.85 to 1.05 × | task objects |
  | Table tilt | 0 to 2° | table |
  | Table height | −0.08 to 0 m | table |
  | External force | 0 to 0.1 mg, during the policy rollout only | task objects |
  | Camera | orbit 0–0.10 m, translation 0–0.03 m or rotation 0–10° | head camera |
  | Friction | 0.01 to 1.25 | all materials |
  | Restitution | 0 to 0.5 | all materials |
  | Object joint damping | 0 to 25 | articulated objects |
  | Arm and gripper damping | 0.5 to 3 × | robot |

- **Expert-filtered episode pools.**
  - For each task, seeds are scanned upward from 100000.
  - The official RoboTwin 2.0 expert runs under each seed's sampled physics, and only seeds it solves are kept.
  - Scanning stops once 100 episodes are accepted, and every policy is evaluated on exactly these 100 episodes.
- **Policy evaluation.** Each policy runs as a server in its own environment, through its official RoboTwin adapter (`get_model / eval / reset_model`). Policy failures always count, and seeds are never replaced.
- **Five baselines from the paper.** Fast-WAM, G0.5, π0.5, Motus and FACT. Their checkpoints and inference settings are pinned in [`configs/policies`](configs/policies).
- **Published pool.** The 5000 accepted seeds, their sampled physical conditions and every Fast-WAM episode outcome are in [`data/`](data/README.md).

Where the paper is silent, the choice made here is documented in [`docs/`](docs/README.md). The main ones:
- Mass, center of mass, size and force apply to task objects only.
- Table height replaces the upstream random table height.

## 📊 Results

Physical Random, 50 tasks × 100 episodes, mean of per-task success rates (%).

| Model | This repo | Paper (Table 1) |
|---|---:|---:|
| **Fast-WAM** | **45.18** | 44.24 |
| G0.5 | in progress | 37.82 |
| π0.5 | in progress | 31.60 |
| Motus | not yet run | 39.60 |
| FACT | not yet run | 39.14 |

Setup: 2× NVIDIA RTX PRO 6000 Blackwell, RoboTwin 2.0 `stable_2.0 @ 13c3c47`, Fast-WAM checkpoint `yuanty/fastwam @ 8eaceeb` with `sigma_shift=5.0`.

### How close is Fast-WAM to the paper?

| Agreement metric | Value |
|---|---:|
| Overall success rate, ours vs. paper | 45.18 vs. 44.24 (Δ +0.94) |
| Pearson correlation of the 50 per-task rates | 0.84 |
| Spearman rank correlation | 0.85 |
| Mean absolute per-task difference | 8.3 points |
| Median absolute per-task difference | 6.0 points |
| Tasks within ±10 points | 35 / 50 |
| Tasks within ±15 points | 44 / 50 |
| Tasks that differ at p < 0.05, two-sample test | 9 / 50 (2.5 expected by chance) |
| Tasks that differ after Bonferroni correction (p < 0.001) | 3 / 50 |

Both sides are 100-episode estimates, so a single task carries roughly ±10 points of sampling noise. Three tasks still differ after correction:

| Task | Ours | Paper |
|---|---:|---:|
| `grab_roller` | 86 | 65 |
| `place_fan` | 64 | 38 |
| `place_dual_shoes` | 6 | 28 |

<details>
<summary><b>Per-task comparison (50 tasks)</b> 🟩 |Δ| ≤ 10 · 🟨 ≤ 20 · 🟥 > 20 · ❗ p &lt; 0.05 · ‼️ p &lt; 0.001</summary>

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

Paper values: arXiv 2609.26292v1, Table 4 (Appendix D). Ours: [`data/phys-pro6000-100-v6/results/fastwam/episodes.csv`](data/phys-pro6000-100-v6/results/fastwam/episodes.csv).

</details>

## 🎯 Benchmark your policy with our seeds

Our pool is published in [`data/phys-pro6000-100-v6`](data/phys-pro6000-100-v6/):

| File | Contents |
|---|---|
| `pool/seeds.json` | per task: the 100 accepted seeds in episode order and every rejected seed with its reason |
| `pool/conditions.csv` | per episode: the 11 sampled physical attributes and the language instruction |
| `results/fastwam/episodes.csv` | per episode: Fast-WAM success, number of actions and wall time |

An episode's physical condition depends only on `(task, seed)`, so it is identical on any machine. To evaluate your policy on the same 5000 episodes:

**1. Set up the simulator.** Clone this repository and build the Docker image. The image installs RoboTwin 2.0 at the pinned revision, SAPIEN, CuRobo and the official assets ([docs/docker.md](docs/docker.md)). Then check that the conditions reproduce:

```bash
git clone https://github.com/joowan1108/RoboTwin-Phys-impl.git && cd RoboTwin-Phys-impl
bash tools/docker.sh build
bash tools/docker.sh doctor      # GPU, Vulkan rendering and CUDA check
bash tools/docker.sh shell       # all following commands run inside this container
python3 analysis/verify_release.py data/phys-pro6000-100-v6   # recomputes all 5000 conditions from their seeds
```

**2. Freeze the pool from the same start seed.** The expert re-checks every seed on your machine and stores the initial scenes:

```bash
for task in $(robotwin-phys tasks); do
  robotwin-phys freeze --task $task --protocol physical_random --episodes 100 --seed 100000 \
    --output manifests/$task.json
done
```

Check that your accepted seeds are ours:

```bash
python3 - <<'EOF'
import json
ours = json.load(open('data/phys-pro6000-100-v6/pool/seeds.json'))['tasks']
for task, entry in ours.items():
    mine = [e['seed'] for e in json.load(open(f'manifests/{task}.json'))['episodes']]
    print(task, 'OK' if mine == entry['accepted'] else 'DIFFERS')
EOF
```

The expert is deterministic per seed, but a different GPU or driver can flip a few borderline seeds. Report any difference together with your results.

**3. Wrap your policy.** Write an adapter with the same three functions as the official RoboTwin policies, and a profile that points to it:

```python
# my_policy/deploy_policy.py
def get_model(args):               # args = the "args" block of the profile
    return load_my_model(args["ckpt"])

def reset_model(model):            # called before every episode
    model.reset()

def eval(env, model, observation): # called repeatedly until the episode ends
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

[`src/robotwin_phys/noop_policy.py`](src/robotwin_phys/noop_policy.py) is a minimal working example, and the profiles in [`configs/policies`](configs/policies) show the five baselines.

**4. Evaluate.** Start the policy server in your policy's own Python environment, then run the 50 tasks:

```bash
PYTHONPATH=$PWD/src /path/to/my-policy-env/bin/python -m robotwin_phys.policy_server \
  --config configs/policies/my_policy.yaml --port 9000 &

for task in $(robotwin-phys tasks); do
  robotwin-phys evaluate --manifest manifests/$task.json --output runs/my_policy/$task --policy-port 9000
done
```

Each task folder gets two kinds of output:
- `runs/my_policy/<task>/summary.json` has the task's `success_rate`.
- `episode-NNNN.result.json` holds the outcome of each seed.

Add `--video` to record an MP4 per episode.

**5. Compare.** The benchmark score is the mean of the 50 per-task success rates. You can compare it two ways:
- Episode by episode, against the Fast-WAM rows of `results/fastwam/episodes.csv`.
- Task by task, against the paper's values in [`references/paper-table4-physical-random.json`](references/paper-table4-physical-random.json).

<details>
<summary><b>Run it on exactly our runtime</b></summary>

The release asset [`phys-pro6000-100-v6-full.zip`](https://github.com/joowan1108/RoboTwin-Phys-impl/releases/tag/data-phys-pro6000-100-v6) contains our 50 sealed manifests, including the verified initial scenes. `robotwin-phys evaluate` accepts them only on an identical runtime: same source, assets, package versions, GPU model and driver. On any other machine, use step 2 instead.

</details>

## 🤖 Baselines

| Model | Checkpoint | Adapter |
|---|---|---|
| Fast-WAM | `yuanty/fastwam` | official FastWAM RoboTwin client |
| G0.5 | `OpenGalaxea/G05` (gated) | official GalaxeaVLA RoboTwin adapter |
| π0.5 | `motus-robotics/pi0.5_robotwin2` | OpenPI PyTorch |
| Motus | `motus-robotics/Motus_robotwin2` | official Motus adapter |
| FACT | `Bariona/fact-wam` | native FACT server and official client |

Revisions and inference arguments are pinned in [`configs/policies`](configs/policies), and preparation steps are in [`docs/policies.md`](docs/policies.md). The bf16 policies need an Ampere or newer GPU.

## 🗂 Repository layout

```text
src/robotwin_phys/   randomization, simulator hooks, episode pools, policy server and evaluation CLI
configs/             physical attribute ranges and policy profiles
data/                published pool and results
analysis/            per-task comparison with the paper, dataset export and verification
tools/               environment setup and multi-GPU schedulers
docker/              simulator and policy images
tests/               CPU tests (no simulator needed)
docs/                design notes and decisions (Korean)
```

## 📄 License and citation

The code and the published data are released under the [MIT License](LICENSE). RoboTwin 2.0, CuRobo, the policy repositories and the model checkpoints keep their own licenses.

If you use this reimplementation, please cite the original paper:

```bibtex
@article{robotwinphys2026,
  title   = {RoboTwin-Phys},
  journal = {arXiv preprint arXiv:2609.26292},
  year    = {2026}
}
```

<div align="center">

# RoboTwin-Phys · independent reimplementation

**Physics-randomized evaluation for RoboTwin 2.0, rebuilt from the paper.**

[![paper](https://img.shields.io/badge/arXiv-2609.26292-b31b1b.svg)](https://arxiv.org/abs/2609.26292)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![tests](https://github.com/joowan1108/RoboTwin-Phys-impl/actions/workflows/tests.yml/badge.svg)](https://github.com/joowan1108/RoboTwin-Phys-impl/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.10-3776ab.svg)
![tasks](https://img.shields.io/badge/tasks-50%20×%20100%20episodes-2ea44f.svg)

[Results](#-results) · [Data](data/README.md) · [Quick start](#-quick-start) · [How it works](#-how-it-works) · [Docs](docs/README.md) · [한국어](README.ko.md)

</div>

> [!NOTE]
> The authors of [RoboTwin-Phys](https://arxiv.org/abs/2609.26292) had not released code, task settings or seed lists when this work started (2026-10-08). Everything here is reimplemented from the paper and documented where the paper is silent. It is not the official implementation.

## ✨ Highlights

- **All 11 physical attributes** of the paper's Table 2, sampled per episode from independent, seeded streams, with the Appendix B overrides for the nine sensitive tasks.
- **Expert-filtered pools shared by every policy.** 50 tasks × 100 episodes, frozen once with the official RoboTwin 2.0 experts and replayed bit-exactly for each model.
- **Five official checkpoints** (Fast-WAM, G0.5, π0.5, Motus, FACT), each served from its own environment through the official `get_model / eval / reset_model` adapter.
- **Provenance you can audit.** Sealed manifests, runtime fingerprints and pinned policy identities refuse to mix results produced under different code, assets or GPUs.
- **Published pool and results.** Seeds, sampled conditions and every episode outcome are in [`data/`](data/README.md), verifiable on any machine.

## 📊 Results

Physical Random, 50 tasks × 100 episodes, mean of per-task success rates (%).

| Model | This repo | Paper (Table 1) | Status |
|---|---:|---:|---|
| **Fast-WAM** | **45.18** | 44.24 | ✅ complete |
| G0.5 (Galaxea-VLA) | – | 37.82 | 🔄 running |
| π0.5 | – | 31.60 | ⏳ queued |
| Motus | – | 39.60 | ⏳ not yet run |
| FACT | – | 39.14 | ⏳ not yet run |

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

Both sides are 100-episode estimates, so a single task carries roughly ±10 points of sampling noise. The three Bonferroni-significant tasks are `grab_roller` (86 vs. 65), `place_fan` (64 vs. 38) and `place_dual_shoes` (6 vs. 28). Different seeds, GPUs and the decisions listed under [How it works](#-how-it-works) can all move individual tasks.

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

**Setup:** 2× NVIDIA RTX PRO 6000 Blackwell, RoboTwin 2.0 `stable_2.0 @ 13c3c47`, Fast-WAM `yuanty/fastwam @ 8eaceeb` with `sigma_shift=5.0`, seeds scanned upward from 100000. The exact runtime fingerprint is in [`provenance.json`](data/phys-pro6000-100-v6/provenance.json).

## 🚀 Quick start

**Run the test suite** (no GPU, no simulator):

```bash
git clone --filter=blob:none https://github.com/RoboTwin-Platform/RoboTwin.git vendor/RoboTwin
git -C vendor/RoboTwin checkout 13c3c47ff4312dd62484bcd51be034af55c062d1
python -m venv .venv && .venv/bin/pip install numpy==1.26.4 PyYAML==6.0.2 "Pillow>=10,<12" transforms3d pytest
PYTHONPATH=src .venv/bin/python -m pytest -q tests \
  --ignore=tests/test_curobo_batch_results.py --ignore=tests/test_curobo_mask_compat.py \
  -k 'not test_force_is_policy_only_and_scales_as_mg and not test_all_five_configs_reference_real_official_adapters and not test_mp4_is_decodable'
```

The ignored and deselected tests need torch, SAPIEN, ffmpeg or the cloned policy repositories. This is also what [.github/workflows/tests.yml](.github/workflows/tests.yml) runs.

**Full environment.** The simulator needs SAPIEN 3.0.0b1, MPLib 0.2.1, CuRobo v0.7.8, PyTorch with CUDA and about 16 GB of official RoboTwin assets. Use the Docker images rather than a host install:

```bash
bash tools/docker.sh build      # CUDA 12.1 image for Turing GPUs; see docs/docker.md
bash tools/docker.sh doctor     # real Vulkan render + CUDA check, writes reports/hardware.json
bash tools/docker.sh validate   # smoke pool + no-op policy episode
```

For Blackwell GPUs and the five policy runtimes, `tools/build_bundle.sh` builds the single bundle image used on the cluster (published as `joowan1108/robotwin-phys:bundle-cu128-v5`). `tools/bootstrap.sh` records the original host installation and is kept for reference.

**Freeze a pool and evaluate one policy:**

```bash
robotwin-phys freeze --task stack_blocks_two --protocol physical_random --episodes 100 --seed 100000 \
  --output manifests/physical_random/stack_blocks_two.json

# in the policy's own environment
PYTHONPATH=$PWD/src /path/to/fastwam-env/bin/python -m robotwin_phys.policy_server \
  --config configs/policies/fastwam.yaml --port 9000

robotwin-phys evaluate --manifest manifests/physical_random/stack_blocks_two.json \
  --output runs/fastwam/stack_blocks_two --host 127.0.0.1 --policy-port 9000 --live --video
```

`--live` serves the head camera at http://127.0.0.1:8765; `--video` records 320×240 MP4 at 30 simulated Hz.

**Whole benchmark on a Slurm cluster.** `tools/parallel_eval.py` prepares all 50 task pools with several workers per GPU, then evaluates one model at a time with up to 8 policy replicas spread over the allocated GPUs. The launchers in [cluster/](cluster/README.md) mount a reviewed patch ZIP from [reports/](reports/) read-only over the image, pin its SHA-256 and the checksums of `src/` and `tools/`, and refuse to start otherwise. See [docs/cluster.md](docs/cluster.md) and [docs/com-scope-fresh-pool.md](docs/com-scope-fresh-pool.md).

### Use the published pool

```bash
python3 analysis/verify_release.py data/phys-pro6000-100-v6   # recomputes all 5000 conditions from their seeds
```

`pool/seeds.json` lists the accepted and rejected seeds per task and `pool/conditions.csv` the sampled attributes per episode. Conditions are a pure function of `(task, seed)` and identical on any machine. The 50 sealed manifests with the verified initial scenes are in the GitHub release asset `phys-pro6000-100-v6-full.zip`. See [`data/README.md`](data/README.md).

## 🔧 How it works

```text
seed ──► sample 11 attributes ──► official expert under that physics ──► accept / reject ──► sealed manifest
                                                                                                 │
             policy server (own env) ◄── RPC ── simulator replays the frozen initial state ◄─────┘
                                                   └─► success / failure stays in the denominator
```

1. **Sample conditions** (`src/robotwin_phys/conditions.py`). Independent PCG64 streams keyed by task, seed and attribute, so disabling one factor never changes another. Ranges live in [`configs/physical_domains.json`](configs/physical_domains.json).
2. **Freeze an expert pool** (`robotwin-phys freeze`, `tools/parallel_eval.py`). Only expert failures and unstable initializations are excluded; runtime errors abort instead of counting as failures.
3. **Evaluate policies** (`robotwin-phys evaluate`). The simulator refuses to run if the scene or fingerprint differs from the manifest; seeds are never replaced.
4. **Report** (`tools/experiment.py`, `analysis/`). Per-task and aggregate rates, a randomization audit of every worker request, paper-style tables.

<details>
<summary><b>Physical attributes</b></summary>

| Attribute | Range | Applies to |
|---|---|---|
| Object mass | 0.125 to 4 × (inertia scaled too) | task objects |
| Center of mass | ±0.015 m per axis | task objects |
| Geometry scale | 0.85 to 1.05 × | task objects |
| Table tilt | 0 to 2° (azimuth 90°) | table |
| Table height | −0.08 to 0 m from 0.74 m | table |
| External force | 0 to 0.1 mg along +X, policy stage only | task objects |
| Camera | one mode per episode: orbit 0–0.10 m, translation 0–0.03 m, rotation 0–10° | head camera |
| Friction | 0.01 to 1.25 | all materials |
| Restitution | 0 to 0.5 | all materials |
| Object joint damping | 0 to 25 | articulated objects |
| Arm and gripper damping | 0.5 to 3 × (one shared factor) | robot |

The paper says "13 physical attributes" but lists 11 rows; the count matches only if camera modes are counted separately. This implementation follows the table.

</details>

<details>
<summary><b>Decisions the paper leaves open</b></summary>

These are documented choices, not claims about the authors' code.

- Mass, center of mass, size and force apply to task objects only; the official clutter objects keep nominal properties, while friction and restitution are scene-wide ([docs/com-scope-fresh-pool.md](docs/com-scope-fresh-pool.md)).
- Table height replaces the upstream random table height rather than adding to it ([docs/reproduction.md](docs/reproduction.md)).
- Three official tasks read attributes in `check_success` that only the expert's `play_once` assigns. The evaluator derives them from the frozen initial scene instead of running the expert first ([docs/success-context-fix.md](docs/success-context-fix.md)).
- Fast-WAM is evaluated with `sigma_shift=5.0`, the upstream setting for its released checkpoint ([docs/policies.md](docs/policies.md)).
- On Blackwell GPUs the SAPIEN-bundled OIDN 2.0.1 denoiser fails silently and policies would see un-denoised frames; the cluster launcher binds OIDN 2.3.3 and proves it with a render probe ([docs/oidn-blackwell.md](docs/oidn-blackwell.md)).
- The pinned GalaxeaVLA adapter resolves the checkpoint path before reading `.hydra/config.yaml`, so a symlinked G0.5 runtime bundle made it read the original repository-relative config; the bundle is hard-linked instead ([docs/g05-runtime-bundle.md](docs/g05-runtime-bundle.md)).
- `put_object_cabinet` has a very low expert acceptance rate under the paper's ranges; it is kept as specified ([docs/put-object-cabinet-feasibility.md](docs/put-object-cabinet-feasibility.md)).

</details>

<details>
<summary><b>Provenance rules</b></summary>

- A manifest stores the sampled condition, its SHA-256, the verified initial state and its SHA-256, the instruction, the expert's result and a fingerprint of `src/robotwin_phys`, `configs/physical_domains.json`, the pinned RoboTwin envs/task configs/embodiment, package versions, GPU name and driver, CuRobo metadata and the asset tree.
- `pool-index.json` seals the manifests of one experiment; every evaluation run pins the manifest hash and the policy identity (name, config hash, adapter hash) in `run.json`. Results under a different identity are refused rather than silently replaced.
- Changing `src/` after a pool exists invalidates the fingerprint. The repository therefore ships explicit, reviewed continuation tools that advance saved records under a named rule with backups and receipts (`tools/continue_success_context.py`, `tools/resume_contact_pool.py`) instead of rebuilding pools.
- The patch ZIPs in `reports/` are immutable once a cluster job has mounted them. Packagers refuse to overwrite a delivered archive.

</details>

## 🗂 Repository layout

```text
src/robotwin_phys/   evaluation library and CLI (conditions, simulator hooks, manifests, worker, RPC, provenance)
configs/             physical_domains.json, policy profiles (checkpoint, adapter, inference args), requirements
tools/               environment setup, pool/evaluation schedulers, provenance continuations, patch packagers
analysis/            read-only analysis of evaluation records (per-task rates vs. the paper, log triage), dataset export/verification
data/                published evaluation pools and results (`data/<experiment>/`, see data/README.md)
cluster/             Slurm/Open OnDemand launchers for the CSI cluster (Apptainer image + reviewed patch ZIPs)
docker/              CUDA 12.1 (Turing) and CUDA 12.8 (Blackwell) images, per-policy runtime images, bundle image
tests/               319 CPU tests; the simulator is faked or driven through recorded fixtures
docs/                design notes, incident reviews and validation records (Korean, indexed in docs/README.md)
reports/             machine-readable validation results and the immutable patch ZIPs mounted by cluster jobs
references/          pinned upstream metadata (model cards, HF file lists, the local RoboTwin diff, paper Table 4 rates)
vendor/              pinned RoboTwin 2.0 (stable_2.0 @ 13c3c47) and CuRobo v0.7.8 checkouts (not committed)
```

## 🤖 Policies

| Model | Checkpoint | Adapter | Replicas per job |
|---|---|---|---:|
| Fast-WAM | `yuanty/fastwam` | official FastWAM RoboTwin client | 4 |
| G0.5 | `OpenGalaxea/G05` (gated) | official GalaxeaVLA RoboTwin adapter | 4 |
| π0.5 | `motus-robotics/pi0.5_robotwin2` | OpenPI PyTorch, card transforms | 4 |
| Motus | `motus-robotics/Motus_robotwin2` | official Motus adapter | 2 (VRAM) |
| FACT | `Bariona/fact-wam` | native FACT server + official client | 1 (native server) |

Revisions and inference arguments are pinned in [`configs/policies`](configs/policies); preparation and caveats are in [`docs/policies.md`](docs/policies.md). bf16 policies need Ampere or newer GPUs.

## 📄 License and citation

The code and the published data in this repository are released under the [MIT License](LICENSE). Vendor code (RoboTwin 2.0, CuRobo, the policy repositories) and model checkpoints keep their own licenses; copies of the RoboTwin and CuRobo licenses accompany the fixtures in `tests/fixtures/`.

If you use this reimplementation, please cite the original paper:

```bibtex
@article{robotwinphys2026,
  title   = {RoboTwin-Phys},
  journal = {arXiv preprint arXiv:2609.26292},
  year    = {2026}
}
```

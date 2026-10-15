# RoboTwin-Phys: independent reimplementation

An independent, paper-based reimplementation of **RoboTwin-Phys** ([arXiv:2609.26292v1](https://arxiv.org/abs/2609.26292)), the physics-randomized evaluation protocol for RoboTwin 2.0. The authors had not released evaluation code, task settings or seed lists when this work started (2026-10-08), so nothing here should be presented as the official implementation or as a bit-exact reproduction of the paper's numbers. What it does provide is a complete, auditable pipeline: physical condition sampling from the paper's appendix, expert-filtered episode pools shared by every policy, policy evaluation over an RPC bridge to each model's own environment, and provenance checks that refuse to mix results produced under different code, assets or GPUs.

The Korean project notes written during development are kept verbatim in [README.ko.md](README.ko.md) and [docs/](docs/README.md). This file is the English entry point.

## What the pipeline does

1. **Sample conditions** (`src/robotwin_phys/conditions.py`). Each episode seed draws the 11 attribute families of the paper's Table 2 from independent PCG64 streams keyed by task, seed and attribute name, so disabling one factor never changes another. Ranges come from [configs/physical_domains.json](configs/physical_domains.json), including the Appendix B overrides for the nine physics-sensitive tasks.
2. **Freeze an expert pool** (`robotwin-phys freeze`, `tools/parallel_eval.py`, `tools/parallel_freeze_task.py`). Seeds are scanned from 100000 upward; each candidate runs the official RoboTwin 2.0 scripted expert in a fresh process under the sampled physics. Only expert failures and unstable initializations are excluded. Runtime errors abort instead of being counted as failures. The accepted 100 episodes per task, with their full initial state, applied physics and a runtime fingerprint, are sealed in a content-addressed manifest.
3. **Evaluate policies** (`robotwin-phys evaluate`, `src/robotwin_phys/policy_server.py`). Every policy is served from its own Python environment through a bounded JSON/ndarray RPC that keeps the official `get_model / eval / reset_model` adapter interface. The simulator replays the exact frozen initial state, applies the policy-stage force, and refuses to run if the scene or fingerprint differs from the manifest. Policy failures stay in the denominator and seeds are never replaced.
4. **Report** (`tools/experiment.py`). Per-task and aggregate success rates, a randomization audit that cross-checks every worker request against the manifest, and paper-style tables.

### Physical attributes

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

The paper calls these "13 physical attributes" but lists 11 rows; the count reconciles only if the camera modes are counted separately. This implementation follows the table.

### Decisions the paper leaves open

These are documented choices, not claims about the authors' code. Each is explained in the linked note.

- Mass, center of mass, size and force apply to task objects only; the official clutter objects keep nominal properties, while friction and restitution are scene-wide ([docs/com-scope-fresh-pool.md](docs/com-scope-fresh-pool.md)).
- Table height replaces the upstream random table height rather than adding to it ([docs/reproduction.md](docs/reproduction.md)).
- Three official tasks read attributes in `check_success` that only the expert's `play_once` assigns. The evaluator derives them from the frozen initial scene instead of running the expert first ([docs/success-context-fix.md](docs/success-context-fix.md)).
- Fast-WAM is evaluated with `sigma_shift=5.0`, the upstream setting for its released checkpoint ([docs/policies.md](docs/policies.md)).
- `put_object_cabinet` has a very low expert acceptance rate under the paper's ranges; it is kept as specified ([docs/put-object-cabinet-feasibility.md](docs/put-object-cabinet-feasibility.md)).

## Repository layout

```text
src/robotwin_phys/   evaluation library and CLI (conditions, simulator hooks, manifests, worker, RPC, provenance)
configs/             physical_domains.json, policy profiles (checkpoint, adapter, inference args), requirements
tools/               environment setup, pool/evaluation schedulers, provenance continuations, patch packagers
cluster/             Slurm/Open OnDemand launchers for the CSI cluster (Apptainer image + reviewed patch ZIPs)
docker/              CUDA 12.1 (Turing) and CUDA 12.8 (Blackwell) images, per-policy runtime images, bundle image
tests/               304 CPU tests; the simulator is faked or driven through recorded fixtures
docs/                design notes, incident reviews and validation records (Korean, indexed in docs/README.md)
reports/             machine-readable validation results and the immutable patch ZIPs mounted by cluster jobs
references/          pinned upstream metadata (model cards, HF file lists, the local RoboTwin diff)
vendor/              pinned RoboTwin 2.0 (stable_2.0 @ 13c3c47) and CuRobo v0.7.8 checkouts (not committed)
```

## Quick start

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

## Provenance rules

- A manifest stores the sampled condition, its SHA-256, the verified initial state and its SHA-256, the instruction, the expert's result and a fingerprint of `src/robotwin_phys`, `configs/physical_domains.json`, the pinned RoboTwin envs/task configs/embodiment, package versions, GPU name and driver, CuRobo metadata and the asset tree.
- `pool-index.json` seals the manifests of one experiment; every evaluation run pins the manifest hash and the policy identity (name, config hash, adapter hash) in `run.json`. Results under a different identity are refused rather than silently replaced.
- Changing `src/` after a pool exists invalidates the fingerprint. The repository therefore ships explicit, reviewed continuation tools that advance saved records under a named rule with backups and receipts (`tools/continue_success_context.py`, `tools/resume_contact_pool.py`) instead of rebuilding pools.
- The patch ZIPs in `reports/` are immutable once a cluster job has mounted them. Packagers refuse to overwrite a delivered archive.

## Policies

Five RoboTwin 2.0 checkpoints are wired through their official adapters where one exists: Fast-WAM, Motus, FACT, π0.5 and G0.5 (Galaxea-VLA). Checkpoint repositories, revisions and inference arguments are pinned in [configs/policies](configs/policies); preparation steps and known caveats are in [docs/policies.md](docs/policies.md). bf16 policies need Ampere or newer GPUs.

## Status

The v6 experiment (`phys-pro6000-100-v6`: 50 tasks × 100 episodes, physical_random) is being evaluated on two RTX PRO 6000 GPUs. No success rates are published in this repository yet; the paper reports 44.24 (Fast-WAM), 39.60 (Motus), 39.14 (FACT), 31.60 (π0.5) and 37.82 (G0.5) under its Physical Random setting. Validation evidence for the infrastructure itself is in [docs/validation.md](docs/validation.md) and `reports/*.json`.

## Citation

```bibtex
@article{robotwinphys2026,
  title   = {RoboTwin-Phys},
  journal = {arXiv preprint arXiv:2609.26292},
  year    = {2026}
}
```

Vendor code (RoboTwin 2.0, CuRobo, the policy repositories) keeps its own licenses; copies of the RoboTwin and CuRobo licenses accompany the fixtures in `tests/fixtures/`. This repository does not yet declare a license for its own code.

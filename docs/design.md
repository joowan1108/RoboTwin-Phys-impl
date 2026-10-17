# Design and implementation notes

This document describes how this repository implements the RoboTwin-Phys protocol (arXiv 2609.26292v1) on top of RoboTwin 2.0. It covers the decisions taken where the paper is silent and the changes made to upstream behaviour that the evaluation depends on.

The authors' code, task configuration and seed lists were not public when this work was done. The official repository ([yefeng00/RoboTwin-Phys](https://github.com/yefeng00/RoboTwin-Phys), revision `c629e7f`) contains only a README, kept as [references/official-phys-README.md](../references/official-phys-README.md). Everything below is an independent reimplementation from the paper.

## Pinned upstream

| Component | Revision |
|---|---|
| RoboTwin 2.0 | `stable_2.0` @ `13c3c47ff4312dd62484bcd51be034af55c062d1` (installation scripts live under `script/`) |
| CuRobo | v0.7.8 @ `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`, the version installed by that RoboTwin revision |
| Simulator stack | Python 3.10, SAPIEN 3.0.0b1, MPLib 0.2.1 |

The pins are the public revisions closest to the paper. They are not a claim about the exact commits the authors used. The 50 task names come from the official `task_config/_eval_step_limit.yml` registry ([conditions.py](../src/robotwin_phys/conditions.py), `task_names`).

## Physical randomization

### Attributes and ranges

The ranges are read from [configs/physical_domains.json](../configs/physical_domains.json), which transcribes Appendix A (Table 2) of the paper.

| Factor | Range | How it is applied |
|---|---|---|
| `mass` | 0.125 to 4 × | Multiplies the mass the task assigned. Inertia, computed after geometry scaling, gets the same factor. |
| `com` | −0.015 to 0.015 m, per axis | Independent x/y/z offset of the body's centre of mass. |
| `geometry` | 0.85 to 1.05 × | Applied before loading to collision mesh, visual mesh and the grasp/functional point metadata together. |
| `table_tilt` | 0 to 2° | Rotation about −X, so the surface normal tilts towards +Y (azimuth 90°). Initial object poses are rotated with the table about its centre and their height above the tabletop is rescaled with the geometry factor. The robot base does not move. |
| `table_height` | −0.08 to 0 m | Offset from the nominal 0.74 m table height. |
| `force` | 0 to 0.1 mg | Constant +X force of `force_mg × mass × 9.81` N on each task object, added at every physics step of the policy stage only. |
| `camera` | orbit 0 to 0.10 m, translation 0 to 0.03 m, rotation 0 to 10° | One of three modes per episode, head camera only (see below). |
| `friction` | 0.01 to 1.25 | Static and dynamic friction of every collision material. |
| `restitution` | 0 to 0.5 | Restitution of every collision material. |
| `object_damping` | 0 to 25 | Drive damping of the active joints of articulated objects. Stiffness and force limit are kept. |
| `robot_damping` | 0.5 to 3 × | One factor that scales both arm joint damping and gripper damping. |

Continuous values are sampled uniformly. The camera mode is chosen with equal probability, and the camera direction (translation direction, rotation axis, orbit direction) is an isotropic unit vector. Robot links are never part of object randomization: their mass and CoM stay nominal. When a material or joint value is not randomized (clean and official protocols, or an ablation that disables the factor), the original upstream value is kept rather than replaced by a guessed nominal.

The camera modes, implemented in `PhysicsMixin.load_camera` ([simulator.py](../src/robotwin_phys/simulator.py)):

- **orbit**: the gaze target is the intersection of the nominal optical ray with the tabletop. The camera moves on the sphere around that target, keeping the radius, by the sampled chord distance, and is re-aimed at the target.
- **translation**: the camera is displaced along the sampled direction without re-aiming.
- **rotation**: the camera rotates in place about the sampled axis.

The wrist cameras and the other static cameras keep their upstream configuration.

### 13 versus 11 attributes

The paper speaks of 13 physical attributes, while Table 2 presents them in 11 rows: the three camera perturbation modes form one row, and arm and gripper damping share one factor. This implementation follows the table. It exposes 11 factors (`FACTORS` in [conditions.py](../src/robotwin_phys/conditions.py)), the centre-of-mass offset is a 3-vector, and arm and gripper share one damping scale. It does not treat the 13 as independent scalars.

### Per-task overrides (Appendix B)

Nine tasks narrow the global ranges. Values that are not listed keep the global range.

| Task | Override |
|---|---|
| `dump_bin_bigbin` | CoM ±1 mm |
| `grab_roller` | CoM ±2 mm |
| `beat_block_hammer` | CoM ±2 mm |
| `put_bottles_dustbin` | CoM ±2 mm |
| `open_microwave` | object damping 0 to 0.25 |
| `turn_switch` | object damping 0 to 1 |
| `put_object_cabinet` | geometry fixed at 1, arm/gripper damping fixed at 1 |
| `scan_object` | camera magnitudes halved, CoM ±2 mm |
| `place_bread_skillet` | camera magnitudes halved, CoM ±1 mm, geometry 0.9 to 1.05, tilt 0 to 1° |

### Independent, seeded streams

A condition is a pure function of `(task, seed, protocol, ablation)`. In `sample` ([conditions.py](../src/robotwin_phys/conditions.py)), every parameter draws from its own NumPy `PCG64` generator, seeded with the first four bytes of `SHA-256(["physics-v1", task, seed, parameter])`. Python's per-process `hash()` is never used. Consequences:

- The same `(task, seed)` gives the same physical condition on any machine.
- Disabling a factor (`--only` / `--without` ablations, or the `physical_only` protocol) does not change the values of the other factors.
- Model name, model order and the policy's own RNG consumption have no influence on the condition.

The condition is hashed (`condition_sha256`, canonical JSON) and stored with each episode.

The remaining upstream randomness (layout, clutter, textures, lighting) still uses NumPy's and Python's global generators. `EnvironmentRNG` ([rng.py](../src/robotwin_phys/rng.py)) gives the environment its own saved generator state, seeded from the episode seed. It swaps that state in around `setup_demo`, `get_obs`, `take_action` and `play_once` and restores the caller's state afterwards, so a policy that consumes random numbers cannot shift the scene. Upstream's `torch.manual_seed` in setup is isolated with `torch.random.fork_rng`.

### External force only during the policy stage

Table 2 places the external force in the policy rollout. The force is therefore off while the scene settles, during the stability check and during the expert demonstration. It is applied at every simulation step once the worker enters the policy stage (`_step_with_force` in [simulator.py](../src/robotwin_phys/simulator.py)). The manifest records `force_during_expert: false`. An expert success thus does not guarantee that the same episode is solvable with the force on.

### Protocols

| Protocol | Scene configuration | Physical factors |
|---|---|---|
| `clean` | official `demo_clean` | none |
| `official_random` | official `demo_randomized` | none |
| `physical_random` | official `demo_randomized` | all 11, with Appendix B overrides |
| `physical_only` | official `demo_randomized` | all 11 (analysis protocol for ablations) |

`--only` and `--without` are analysis tools of this implementation. The paper does not publish factor-wise ablation ranges, and results obtained with them are not paper numbers.

## Scope decisions where the paper is silent

These are local choices, not verified against the authors' code.

**Object attributes apply to task objects only.** Mass, centre of mass, geometry scale and the policy-stage force apply to the task objects. The official clutter distractors keep their nominal mass, CoM and size and receive no force. The paper describes these attributes, and the CoM reductions for thin objects such as the hammer, in terms of the manipulated objects. Applying a 1.5 cm CoM shift to thin clutter such as pens and markers moves the CoM outside the body, and the official stability check (2000 settling steps, then a pose change above 3° over 500 steps) rejects the scene before any manipulation. That biases the accepted pool towards small CoM offsets. On a 50-task, 5-candidate comparison, scoping these attributes to task objects lowered the `unstable_initialization` rate from 66% to 18% of candidates. Clutter poses still follow the tilted table.

Clutter is identified by the scene entities that the official `get_cluttered_table` adds, because that method clears `self.cluttered_objs` before returning. Both rigid actors and articulation links are covered. Each body in the stored `initial_state.physical.bodies` records `clutter`, the applied `com_offset_m` and `mass_scale`, and whether it receives `external_force`.

**Friction and restitution are scene-wide.** Contact material is shared by objects, clutter, table and grippers, so the sampled friction and restitution apply to every collision shape.

**Table height replaces the upstream random table height.** When the factor is active, the table is placed at 0.74 m plus the sampled offset, instead of adding the offset on top of RoboTwin's `random_table_height` and task-specific bias. This avoids a double randomization beyond the paper's range. It also replaces the −0.1 m upstream bias of `place_dual_shoes`. This is the first item to compare once the authors' code is public.

**Head camera only.** Only the head camera is perturbed (see the camera modes above).

**Visual randomization is kept, lighting is indexed by simulated time.** The official random backgrounds, clutter and lights are kept. Upstream's "crazy random light" drew from the global RNG on every render request, so the number of observations a policy requested changed the lighting. Lighting is now seeded from the episode seed and the simulation step (`_update_render` in [simulator.py](../src/robotwin_phys/simulator.py)). Pixels can therefore differ from stock RoboTwin.

**Instruction.** The language instruction is generated once, from the `unseen` instruction set, with an RNG seeded from `(task, seed)`, and stored in the manifest. The policy never receives the physical ground truth.

**Object joint damping** is interpreted as drive damping. **Mass scaling** multiplies the mass the task script set, and the inertia after geometry scaling.

## Episode pool construction

### Seed scan and expert filter

`robotwin-phys freeze` ([cli.py](../src/robotwin_phys/cli.py)) builds one pool per task:

1. Candidate seeds are scanned upward from the start seed, 100000 by default. This is the value RoboTwin's evaluation loop uses for run seed 0 (`100000 * (1 + seed)`).
2. Each candidate runs in a fresh worker process ([worker.py](../src/robotwin_phys/worker.py)). The condition is applied before the upstream stability check, and the official scripted expert (`play_once`) runs under the sampled physics.
3. A candidate is accepted if the planner succeeded and `check_success()` holds. Otherwise it is recorded as rejected with a reason:
   - `unstable_initialization`: the official stability check raised `UnStableError` while the scene was created.
   - `expert_failed`: the expert's plan failed, or it finished without success. Structured planning failures carry a `planning_failure` record (kind, actor, arm, contact ids, traceback where relevant; see [Fixes to upstream behaviour](#fixes-to-upstream-behaviour)).
4. Scanning stops at the requested number of accepted episodes (100), within a budget of 10000 candidates. Candidates are evaluated strictly in seed order, so worker timing never selects seeds.

Any other exception (missing asset, CUDA error, out-of-memory, unexpected assertion, worker crash or timeout) aborts the task as a runtime error. It is never counted as an infeasible seed. Upstream's evaluation loop catches every exception and moves to the next seed, which would hide environment errors such as a missing planner configuration as rejections. This implementation deliberately classifies only failures whose cause is understood.

Progress is saved after every candidate in a sealed `<task>.progress.json`, and `--resume` continues after the last recorded seed. A seed whose worker crashed has no record and is evaluated again.

### Sealed manifests

A finished task pool is written once to `<task>.json`. Frozen manifests are never overwritten. A manifest ([manifest.py](../src/robotwin_phys/manifest.py)) contains, per episode, the seed, the condition and its hash, the instruction, the expert's info, and the verified initial state: entity poses and velocities, articulation qpos/qvel, camera poses, table height, textures, lighting, and the physical values read back from the simulator (per-body mass, inertia, CoM and materials, object and robot joint damping). It also stores `initial_state_sha256`, the rejected seeds with their reasons, and the runtime fingerprint. The whole document is sealed with a SHA-256 over its canonical JSON. `load` refuses a manifest whose seal, condition hashes, episode indices or seeds are inconsistent, or that is not `verified`. Unverified `candidates` output, which previews conditions without running the simulator, is never accepted for evaluation.

When all 50 tasks are frozen, the scheduler writes `pool-index.json`. This sealed index binds the experiment suite hash, the protocol, the episode count and the SHA-256 of every task manifest. It also exports `conditions.csv`, derived from the manifests and never re-sampled. Later runs verify the index against the manifests and refuse to regenerate or mix pools.

### Runtime fingerprint

`fingerprint()` ([provenance.py](../src/robotwin_phys/provenance.py)) covers:

- the SHA-256 of every file in `src/robotwin_phys/*.py`, `configs/physical_domains.json`, RoboTwin's `envs/**/*.py`, `task_config/*.yml`, the `aloha-agilex` embodiment (`*.yml`, `*.urdf`, `*.srdf`) and `description/` (`*.json`, `*.py`);
- the RoboTwin revision;
- a content hash of all official assets (16 GB, cached by file stat);
- the versions of numpy, torch, sapien, mplib, scipy, warp-lang and nvidia-curobo;
- the GPU model, compute capability, driver version and torch CUDA version;
- the CuRobo revision, the SHA-256 of its two patched source files, and the ID of the compatibility repair.

`tools/` and `analysis/` are outside the fingerprint.

`robotwin-phys evaluate` refuses a manifest whose fingerprint differs from the current runtime. Evaluation reconstructs each episode from its stored condition and seed and compares the initial scene before the first action. That comparison only means something if the simulator, assets, physics engine and planner are the ones the expert was verified with. Full trajectories are not guaranteed to be bitwise identical across hardware, PhysX builds or precision. On a different machine, freeze a new pool from the same start seed: the condition of each seed is identical, and the accepted seed list can be compared seed by seed with the published one.

### Initial-state check

Every evaluation or replay episode starts in a new process, rebuilds the scene from the stored condition and seed, and compares the full initial state with the manifest before any policy action (`assert_same_state` in [simulator.py](../src/robotwin_phys/simulator.py)). Floats are compared with an absolute tolerance of 2e-6 and a relative tolerance of 1e-6. That tolerance admits serialization noise, not a different scene. Any mismatch stops the evaluation.

## Evaluation

### Policy server

Each policy runs in its own Python environment, as a server that imports nothing from the simulator ([policy_server.py](../src/robotwin_phys/policy_server.py)). It loads the model's official RoboTwin adapter and calls the upstream interface unchanged: `get_model(args)` once, `reset_model(model)` before every episode, and `eval(env, model, observation)` repeatedly. The simulator worker connects over a local TCP socket ([rpc.py](../src/robotwin_phys/rpc.py)). The server receives the observation (RGB cameras, 14-dimensional joint state) and a proxy `env` whose `get_obs`, `get_instruction`, `take_action` and `check_success` calls and a fixed set of attributes are forwarded to the simulator. Preprocessing, normalization and action chunking stay inside the adapter. Messages are length-prefixed JSON with base64 NumPy arrays and a 64 MiB limit. Pickle and object arrays are rejected.

Before loading, the server checks the adapter's git revision against `code_revision`, required checkpoint paths, the minimum VRAM, and Ampere-or-newer compute capability for profiles marked `requires_native_bf16`. Before each episode the worker sends a reset seed derived from `(policy_seed, episode seed)`. The server seeds NumPy, Python `random` and torch with it, then calls `reset_model`.

### Policy identity

On connection the server announces its identity: profile name, SHA-256 of the resolved profile (`config_sha256`), checkpoint repository/revision/file, code revision, and SHA-256 of the adapter file. `evaluate` writes the identity, the manifest hash, the policy seed and the mode into `run.json`. It refuses to resume an output directory whose `run.json` differs, and each worker refuses a server whose identity changed mid-run. When several replicas serve one model, they must report identical identities. Changing any inference argument therefore requires a new output directory.

### Scoring rules

- A policy that runs and does not succeed within the task's step limit is a failure and stays in the denominator.
- Seeds are never replaced, and the pool is never rebuilt because of policy results.
- Runtime errors (connection loss, out-of-memory, state mismatch) are reported as errors, not as task failures, and an incomplete task is not reported as a success rate. Re-running with `--resume` retries only episodes that did not complete.
- A run is benchmark-eligible only with 100 completed episodes per task, a non-ablated `clean`, `official_random` or `physical_random` manifest, no `--smoke-max-actions` limit, and a policy other than the no-op infrastructure test.
- The benchmark score is the arithmetic mean of the per-task success rates. Tables show unfinished cells as pending, never as 0%.

The scheduler records a per-episode audit (`episode-NNNN.randomization.json`) that compares the worker's request and result with the shared manifest. The aggregate tables (`results.md`, `results.tex`, `results.json`, `randomization.csv`, `randomization.jsonl`, `condition_consistency.csv`, `randomization_audit.json`) are produced by [tools/experiment.py](../tools/experiment.py) and [tools/randomization_audit.py](../tools/randomization_audit.py). A task whose evidence does not match the manifest is excluded from the success tables.

### Success context for three tasks

RoboTwin's `eval_policy.py` keeps one environment object alive: it runs the expert's `play_once` as the feasibility check and then evaluates the policy on the same object. Attributes that `play_once` assigns are therefore still present when `check_success` runs during the policy rollout. Here every episode starts a fresh process without the expert, so tasks whose `check_success` reads such attributes would raise `AttributeError` at the first step. An AST scan of the 50 pinned tasks finds exactly three:

| Task | Attribute read by `check_success` | Expression in `play_once` |
|---|---|---|
| `open_laptop` | `arm_tag` | `"left"` if `get_face_prod(laptop.q, [1,0,0], [1,0,0]) > 0` else `"right"` |
| `place_object_scale` | `arm_tag` | `"right"` if `object.p[0] > 0` else `"left"` |
| `put_object_cabinet` | `arm_tag`, `origin_z` | same arm rule; `origin_z = object.p[2]` |

[success_context.py](../src/robotwin_phys/success_context.py) sets these attributes with the official expressions, after the initial-state check and before the first policy action. The values are read from the verified initial scene, so they equal what the expert computed for the same seed. They are cross-checked against the arm the expert recorded in the manifest (`expert_info["info"]["{a}"]`), and a mismatch is an error. The result file records the derived `success_context`. The other 47 tasks are unaffected.

## Fixes to upstream behaviour

The evaluation depends on the following changes to RoboTwin 2.0 and CuRobo v0.7.8. Changes to RoboTwin source are applied by [tools/patch_robotwin.py](../tools/patch_robotwin.py), which checks the pinned revision and each anchor before editing. Changes to CuRobo are applied in-process by [curobo_compat.py](../src/robotwin_phys/curobo_compat.py), which verifies the CuRobo revision, the SHA-256 of `motion_gen.py` and `trajopt.py`, and the AST hash of every method it rewrites. The SDK files on disk are not modified.

### Exhausted grasp search

Upstream `choose_grasp_pose` returns `(None, None)` when no contact yields a reachable grasp. `grasp_actor` then builds a move to `None` and fails with the assertion `target_pose cannot be None`, and the official evaluation loop silently skips the seed through its catch-all `except`. `PlanningMixin` ([planning.py](../src/robotwin_phys/planning.py)) calls the official search and candidate selection unchanged. It first validates that every requested contact matrix exists and is finite. When the complete search returns no pose, it raises `ExpertPlanningFailure` (kind `no_reachable_grasp`), which the worker records as `expert_failed` together with the actor, its pose, the arm and the contacts checked. During a multi-contact search a single failed contact still returns `None`, so the next contact is tried. Tasks that query `get_grasp_pose` directly, such as `click_alarmclock`, get the same classification.

### Explicit empty grasp annotations

All five official `086_woodenblock` variants have `contact_points_pose: []`, and `place_a2b_left` and `place_a2b_right` can sample this object. The scripted expert has no grasp candidate. An explicitly empty list is recorded as `expert_failed` with kind `no_grasp_annotations`. A missing key, a missing file or null metadata remains a runtime error. The object is not removed from sampling and no grasp point is invented. Same file.

### Contact index unavailable in an asset variant

`place_container_plate` asks for contact 2 on the left arm, but `021_cup` variants 6 and 7 annotate only contacts 0 and 1. Upstream `get_grasp_pose` returns `None` and action construction fails. A request beyond a valid annotation list is recorded as `expert_failed` with kind `unavailable_grasp_annotation`, listing the requested and available ids. The expert's choice is preserved: the contact is not remapped. An audit of all 694 object metadata files against the 172 contact and functional point requests of the 50 tasks found only these two mismatches (`086_woodenblock` and `021_cup` 6/7). Same file.

### Exception after a recorded planner failure

Upstream `left/right/together_move_to_pose` only set `plan_success = False` when CuRobo fails, and the task script keeps running. Later `grasp_actor`/`place_actor` calls then return `(None, [])`, and `put_bottles_dustbin` indexes that result (`right_action[1][0]`) and raises `IndexError`. A `play_once` wrapper in `PlanningMixin` records any exception raised while `plan_success` is exactly `False` as `expert_failed` with kind `exception_after_planning_failure`, including the exception type, message and traceback. With `plan_success` still `True`, including after the "Try new way" retries that reset it (`open_microwave`, `place_can_basket`, `place_object_basket`), the exception propagates as a runtime error. One limitation follows from this rule: a genuine runtime error that happens after a planner failure is recorded as a rejection, but its traceback is kept in the seed's result file.

### Finite rotations in the transform helpers

`get_align_matrix` in `envs/utils/transforms.py` computes `arccos(dot(v1, v2))`. Float32 rounding can push the dot product slightly outside [−1, 1], which produces NaN and then an eigenvalue failure in `mat2quat`. Zero-length directions also produce NaN, and anti-parallel vectors returned the identity instead of a half turn. [numerics.py](../src/robotwin_phys/numerics.py) provides replacements for `get_align_matrix`, `cal_quat_dis`, `get_product_vector` and `rotate_cone`, plus guards in `rotate2rob` and `rotate_along_axis`: clamped cosines, a deterministic 180° rotation for anti-parallel vectors, explicit errors for zero, NaN or Inf directions, and a valid frame when the view and up directions are parallel. `patch_robotwin.py` binds them into the pinned `transforms.py`. In addition, `PlanningMixin.get_place_pose` validates the actor, target, end-effector and functional-point poses first. When valid input still has no defined placement direction, the candidate is recorded as `expert_failed` with kind `undefined_placement_direction`. Invalid input stays a runtime error.

### CuRobo exceptions in the RoboTwin planner wrapper

`envs/robot/planner.py` wrapped `motion_gen_batch.plan_batch` in `except Exception` and returned ten `"Failure"` results, which turned out-of-memory and similar errors into ordinary planning failures. The wrapper is removed by `patch_robotwin.py`. A failed plan is still reported through `result.success`, and runtime exceptions propagate.

### CuRobo IK success mask shared with the result

In CuRobo v0.7.8, `_plan_from_solve_state` and `_plan_from_solve_state_batch` build `MotionGenResult(ik_result.success, ...)`, so the result and the IK result share one tensor. After a graph-search failure, `result.success[:] = False` also clears the IK mask, and the later indexing of the already-computed goal configurations fails with a shape mismatch (also reported upstream as [NVlabs/curobo#502](https://github.com/NVlabs/curobo/issues/502)). `curobo_compat.py` passes `ik_result.success.clone()` instead. The intended trajectory-optimization fallback after a graph failure still runs, and real IK, graph and trajectory failures remain failures.

### CuRobo batch result consistency

When a batch plan's first attempt fails and a later attempt succeeds, CuRobo clones the first result and copies the successful rows into it. In v0.7.8 the copied pieces do not match. `TrajOptSolver._get_result(return_all_solutions=True)` returns `goalset_index` for every timestep while the finetune best-select returns only the last, and `optimized_dt` was not copied with the trajectory. `curobo_compat.py` (repair ID `consistent-batch-results-v2`) makes the following changes:

- the all-solutions branch returns the terminal goal index, as best-select does, and the per-timestep metrics are kept;
- `copy_idx` copies `optimized_dt` together with the selected trajectory;
- `clone` keeps an independent `path_buffer_last_tstep` list and carries `finetune_time`;
- `_plan_batch_attempts` retimes the accumulated result that is returned rather than the last attempt;
- the batch failure result for an invalid pose-cost query is built correctly.

Planner seeds, optimizer settings and success criteria are unchanged.

### Other RoboTwin patches

- `get_cluttered_table` used mutable default lists that accumulated the table bias across episodes in one process. They are copied on entry.
- The scene is created with an explicit PhysX CPU system and a render system on the device in `SAPIEN_RENDER_DEVICE` (default `cuda:0`), so a headless run renders on an actual NVIDIA Vulkan device.
- Object factories in `envs/utils/create_actor.py` call the geometry and pose hooks in [hooks.py](../src/robotwin_phys/hooks.py), which scale sizes, meshes and point metadata and place objects on the tilted table. Clutter is built at nominal size.
- The SAPIEN URDF loader (`.srdf` path, UTF-8 encoding) and the MPLib screw-planner collision check are changed as the official RoboTwin installation guide instructs.

[`tools/patch_robotwin.py`](../tools/patch_robotwin.py) applies these changes to the pinned checkout and refuses to run unless the checkout is at the pinned revision.

## Policy-specific notes

- **Fast-WAM** must run with `sigma_shift: 5.0` ([configs/policies/fastwam.yaml](../configs/policies/fastwam.yaml)). The pinned FastWAM README states that the released checkpoints are evaluated with `EVALUATION.sigma_shift=5.0`. Without it, `sim_robotwin.yaml` leaves the value null and the action scheduler falls back to `infer_shift: 1.0` from `configs/model/fastwam.yaml`, a later training default. The image sets `redirect_common_files: false` so the Wan files come from the pinned Hugging Face revisions.
- **G0.5**: the pinned GalaxeaVLA adapter calls `Path(ckpt_setting).resolve()` and reads `../../.hydra/config.yaml` next to the resolved checkpoint. The downloaded config points to the processor and action tokenizer with repository-relative paths. [tools/prepare_g05.py](../tools/prepare_g05.py) therefore builds `checkpoints/g05-runtime/` with a rewritten config that uses absolute paths, and hard-links (or copies) the checkpoint and dataset stats into it. A symlink would resolve back into the original download and load the original config. The profile also passes `sim_task: robotwin`. The upstream evaluator always passes the README's `task=robotwin` as `sim_task`, and the adapter turns it into the Hydra override that selects the `data`, `model` and `tokenizer` groups. Without it, `cfg.data` is missing.
- **Motus** needs about 41 GB of VRAM with the T5 encoder on the same GPU, or about 24 GB with the encoder elsewhere (model card). The profile enforces `minimum_vram_gb`, and Motus is run with fewer replicas than Fast-WAM (two in the dual layout). An optional, explicitly recorded placement (`t5_device`, patched into the official adapter by [docker/policies/patch_motus.py](../docker/policies/patch_motus.py)) puts only the T5 encoder on a second GPU.
- **FACT** runs as a native inference server plus the official RoboTwin client inside the RPC bridge. The native server's address is part of the policy identity, so the scheduler runs exactly one FACT replica. A small patch ([docker/policies/patch_fact.py](../docker/policies/patch_fact.py)) forwards the episode-derived seed to the native server instead of a time-based seed.
- **bf16 profiles** (Fast-WAM, G0.5, π0.5, Motus) require compute capability 8.0 or higher. The server refuses to start on older GPUs such as sm_75, because `torch.cuda.is_bf16_supported()` can report true without native bf16 support. Running a model at another precision would produce a different profile and is not equivalent to the official setting.

## Known platform issues

### OIDN denoiser on Blackwell

RoboTwin's `setup_scene` always enables ray tracing (32 spp, path depth 8) with the Intel Open Image Denoise (OIDN) denoiser. The SAPIEN 3.0.0b1 wheel bundles OIDN 2.0.1, which predates NVIDIA Blackwell (sm_120). On Blackwell GPUs the CUDA device cannot be created, every denoise call logs `[svulkan2] [error] OIDN Error: invalid handle`, and rendering continues without denoising. Policies then see noisy 32 spp frames, a distribution shift from the denoised images they were trained on. The expert pool does not depend on images, but policy results do. On a Blackwell run, the first-frame Laplacian standard deviation of the 320×240 head camera was 22.6, against 12.1 for the same task rendered on an RTX 2080 Ti.

OIDN 2.3.3 added Blackwell support (see also [SAPIEN issue #243](https://github.com/haosulab/SAPIEN/issues/243)). Install the 2.3.3 libraries in place of `sapien/oidn_library/` and replace `sapien/_oidn_tricks.py` with [docker/oidn/_oidn_tricks.py](../docker/oidn/_oidn_tricks.py), which preloads the 2.3.3 libraries. `libsvulkan2.so` only requires the soname `libOpenImageDenoise.so.2`. Verify the result with a render probe before evaluating: render an `rt` scene with the `oidn` denoiser, check that `libOpenImageDenoise_device_cuda.so.2.3.3` is mapped in the process, and check that no `OIDN Error` line appears. On an RTX 2080 Ti, OIDN 2.0.1 and 2.3.3 give images at the same noise level. The OIDN version is not part of the runtime fingerprint, so record the probe result with each run. [setup.md](setup.md#oidn-233-on-blackwell) describes the steps.

Ray-traced episodes are slow. Raise the per-episode `--timeout` (default 1800 s) when long-horizon tasks exceed it, for example to 3600 s.

### `put_object_cabinet` acceptance rate

Under the paper's ranges, the official expert accepts only about 2% of `put_object_cabinet` candidates. The rejections are `expert_failed` with `plan_success: false`, meaning CuRobo reported a planning failure. They are neither errors nor GPU-dependent: rejected seeds re-run on a different GPU fail for the same reason. The scripted expert is already weak on this task without physical randomization (about 13% acceptance under `official_random` in a 30-seed sample). The paper's table height range then lowers the drawer handle, and the opposite arm's grasp planning fails more often. Mass, CoM, friction and damping do not enter path planning. Appendix B fixes only size and arm damping for this task, so the full table height range is kept as specified. The accepted episodes are skewed towards shallow table offsets, which is a property of the paper's feasibility filter. The pool completes within the 10000-candidate budget. Report the acceptance rate with results for this task.

## Multi-GPU scheduling

[tools/parallel_eval.py](../tools/parallel_eval.py) prepares the shared pool and evaluates one model on the GPUs of a job or container allocation. GPU ids always refer to the allocation, never to host GPU order.

**Pool preparation** (`--phase freeze` or `all`) runs independent task processes, `--freeze-workers-per-gpu` per simulator GPU. Inside a task, candidates stay sequential, so the accepted seeds are the same for any worker count. Every process has its own CUDA context, renderer and CuRobo planner. GPU memory, CPU and host RAM grow with the worker count, and the speed-up is not proportional. The published pool was prepared with 8 workers per 96 GB GPU, budgeting about 6 GB of GPU memory and 4 CPU cores per worker. An out-of-memory error stops that task but keeps its progress. Re-run with fewer workers. A task that fails is reported in `freeze-errors.json` while the other tasks continue. The pool is sealed only when all tasks are complete, and the simulator GPUs must produce identical runtime fingerprints.

[tools/parallel_freeze_task.py](../tools/parallel_freeze_task.py) finishes a single remaining task faster. It evaluates the next N seeds concurrently, each in its own worker process, and commits the results strictly in seed order. Results past the seed that completes the pool are discarded. The saved rejected list, the accepted episodes and the final manifest are byte-identical to the sequential freeze.

**Evaluation** starts one policy server per replica, one after another to avoid concurrent checkpoint-loading peaks. Each replica drives one simulator slot, and idle slots take the next pending episode. Inference settings are never changed and no batching is added. The layout follows the allocation:

| Allocation | Layout | Policy replicas |
|---|---|---|
| 1 GPU | `shared`: one simulator and one replica on the same GPU | exactly 1 |
| 2 GPUs | `dual`: simulators on both GPUs, replicas round-robin over both; replica i shares a GPU with simulator slot i | default 4, at most 8 |
| 3 GPUs | `split`: two simulator GPUs, all replicas on a separate inference GPU | default 2, at most 2 |

A Fast-WAM replica holds about 24 GiB after loading, so four replicas do not fit on one 96 GB GPU. The dual layout places two per GPU. FACT always uses one replica, and Motus fewer than Fast-WAM because of its memory use. Re-submitting the same command resumes: verified episodes are reused, and only missing or failed episodes are run again with the same seeds and conditions. Concurrent runs of the same experiment are blocked by file locks.

For local Docker hosts, [tools/docker_parallel.py](../tools/docker_parallel.py) runs one simulator container per GPU and assigns whole tasks to them ([setup.md](setup.md)).

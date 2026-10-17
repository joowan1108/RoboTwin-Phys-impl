# Validation

This page lists what has been checked and what has not. Unit tests run on the CPU without a simulator (`pytest -q`, [tests/](../tests)). The GPU checks were run with the Docker images on NVIDIA RTX 2080 Ti (Turing) and RTX PRO 6000 Blackwell GPUs.

## Conditions and determinism

- **Ranges and overrides.** [tests/test_conditions.py](../tests/test_conditions.py) samples all 50 tasks × 100 seeds and checks every value against the global ranges and the Appendix B overrides.
- **Independence.** The same test file checks that a condition is identical in a separate process and after arbitrary consumption of the global NumPy and Python RNGs, that disabling a factor leaves the other factors' draws unchanged, that `EnvironmentRNG` restores the caller's RNG state, and that tampered or unverified manifests and changed initial scenes are rejected.
- **Published pool.** [analysis/verify_release.py](../analysis/verify_release.py) recomputes every published condition from its `(task, seed)` and compares it with the condition hash recorded in the sealed manifest.
- **External force.** [tests/test_force_physics.py](../tests/test_force_physics.py) checks in PhysX that the force is zero during the expert stage, and that during the policy stage it produces a 0.1 g acceleration independent of the body's mass.
- **Robot links.** Inspection of the sensitive tasks in the real simulator showed that robot link masses stay nominal while object properties change.
- **Clutter scope.** [tests/test_com_scope.py](../tests/test_com_scope.py) checks that task objects receive the CoM offset, mass scale and force while clutter (rigid and articulated) keeps nominal values, that materials apply to both, and that the per-body record says which is which. On real scenes (50 tasks, up to 5 candidates each), this scoping lowered `unstable_initialization` from 66% to 18% of candidates, and the number of tasks with an accepted episode within 5 candidates rose from 36 to 48.

## Frozen episodes

- **Fresh-process replay.** Episodes frozen by the expert were rebuilt in new processes and matched the stored initial state within the 2e-6 absolute / 1e-6 relative tolerance. This was checked on the host install, and in Docker on each of two GPUs.
- **Rejection reasons.** A rejected candidate keeps its reason in the manifest. In the first smoke pool, seed 100001 of `stack_blocks_two` was rejected as `unstable_initialization`.
- **Expert determinism across GPUs.** Rejected `put_object_cabinet` seeds from the Blackwell pool were re-run on an RTX 2080 Ti, and all were rejected again for the same reason.
- **Parallel freeze.** [tests/test_parallel_freeze_task.py](../tests/test_parallel_freeze_task.py) checks that the concurrent seed scanner with 1, 3 and 8 workers writes manifests and evidence files byte-identical to the sequential freeze, resumes from a saved prefix, and leaves progress untouched when a worker fails.
- **Resume.** Resuming an evaluation does not modify completed episode files, and a changed `run.json` identity is refused.

## Docker

On a two-GPU RTX 2080 Ti host, with the `robotwin-phys:cuda12.1` image:

| Check | Result |
|---|---|
| GPU selection and launcher | Single-GPU auto-selection, multi-GPU selection and rejection of duplicate GPU ids pass. |
| In-image test suite | Passes in separate non-root containers on GPU 0 and GPU 1, together with CUDA kernels and a real Vulkan camera render. |
| Frozen-episode replay | Initial states of seeds 100000 and 100002 match on each GPU. |
| Concurrent expert rollouts | `stack_blocks_two` on GPU 0 and `grab_roller` on GPU 1, both seed 100000, succeed at the same time. |
| Sequential use of one GPU | Two task pools processed in turn on GPU 0, with source and runtime fingerprints verified. |
| Isolation | Assets mounted read-only, no host environment mounted, no privileged mode, unprivileged UID. |

The bundle image was used for the full pool preparation and the Fast-WAM evaluation on two RTX PRO 6000 Blackwell GPUs, in the dual layout.

## Video and live view

- MP4 files from the Docker runs (`stack_blocks_two`, 522 frames; `grab_roller`, 184 frames) are 320×240 at 30 fps and decode with `ffprobe`.
- The headless MJPEG live server delivered real JPEG frames with task and seed status, on separate ports for the two GPUs.
- The native `--viewer` was not exercised on a display. Without `DISPLAY` it exits with an error that points to `--live`.

## Sensitive tasks

[tools/validate_sensitive_tasks.py](../tools/validate_sensitive_tasks.py) initializes the nine Appendix B tasks in the real simulator with their override ranges and checks the applied values. All nine initialize. `scan_object` needed seed 100002, because its first seeds were unstable. This is an initialization check, not a claim that the expert succeeds on every seed.

## Upstream fixes

Each fix in [design.md](design.md#fixes-to-upstream-behaviour) has CPU tests that reproduce the original failure from the pinned upstream code and check the repaired behaviour:

| Fix | Tests | What they check |
|---|---|---|
| Grasp classification | [test_expert_planning.py](../tests/test_expert_planning.py) | The `target_pose cannot be None` assertion is reproduced. Exhausted searches, explicitly empty annotations, out-of-range contact ids and exceptions after a planner failure become structured rejections. Successful searches produce identical actions. Missing or corrupt metadata and unexpected errors still propagate. |
| Annotation audit | (offline) | All 694 object metadata files of the official `objects.zip` against the 172 contact and functional point requests of the 50 tasks: the only mismatches are `086_woodenblock` and `021_cup` variants 6 and 7. |
| Transform helpers | [test_transform_numerics.py](../tests/test_transform_numerics.py) | The float32 out-of-range dot product and the `Eigenvalues did not converge` failure are reproduced. Parallel, anti-parallel and orthogonal directions, agreement with the original on ordinary rotations, quaternion sign invariance, and zero/NaN/Inf inputs are covered. |
| CuRobo IK mask | [test_curobo_mask_compat.py](../tests/test_curobo_mask_compat.py) | The shape mismatch (`[3, 6] → [0, 6]`, `[5, 6] → [0, 6]`) is reproduced on the original method body. Graph-failure fallback, real trajectory failures, invalid queries and refusal of unreviewed sources are covered. |
| CuRobo batch results | [test_curobo_batch_results.py](../tests/test_curobo_batch_results.py) | The goal-index shape conflict is reproduced. Trajectory, dt, goal index and path length survive clone/copy across failed and successful attempts. |
| Success context | [test_success_context.py](../tests/test_success_context.py) | The attributes are derived for the three tasks, a mismatch with the expert record is an error, other tasks are unaffected, and the AST scan of the pinned tasks finds exactly these three. In the simulator, frozen `open_laptop` and `place_object_scale` episodes run with the no-op policy complete, where they previously raised `AttributeError`. |
| G0.5 runtime bundle | [test_prepare_g05.py](../tests/test_prepare_g05.py) | The adapter's path resolution is reproduced: a symlinked bundle reads the original config, while the hard-linked bundle reads the rewritten config. Re-running is idempotent. |

## Rendering on Blackwell

With SAPIEN's bundled OIDN 2.0.1, every denoise call on Blackwell failed (`OIDN Error: invalid handle`), and the policy's head-camera frames were visibly noisier: first-frame Laplacian standard deviation 22.6, against 12.1 on an RTX 2080 Ti. With OIDN 2.3.3 bound in, the render probe loads `libOpenImageDenoise_device_cuda.so.2.3.3` and logs no OIDN errors. On the RTX 2080 Ti, OIDN 2.0.1 and 2.3.3 produce the same noise level. All published policy results were obtained with OIDN 2.3.3.

## Results

Fast-WAM on the published pool (50 tasks × 100 episodes, Physical Random) reaches 45.18%, against 44.24% in the paper. The Pearson correlation of the 50 per-task rates with the paper's Table 4 is 0.84. Per-episode outcomes are in [data/physical-random-100](../data/physical-random-100/README.md).

## Not validated

- Bitwise-identical trajectories across GPU models, drivers or PhysX builds. The runtime fingerprint deliberately refuses such comparisons.
- Peak memory and throughput of every model and layout. Only the configurations described in [design.md](design.md#multi-gpu-scheduling) were run.
- G0.5, π0.5, Motus and FACT on the full benchmark.

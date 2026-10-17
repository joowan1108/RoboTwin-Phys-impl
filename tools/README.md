# Tools

Command-line helpers around the `robotwin_phys` package. None of them is part of the runtime fingerprint.

## Environment and assets

| Script | Purpose |
|---|---|
| `docker.sh`, `docker_parallel.py` | Build, check (`doctor`) and validate the simulator image; run it on one or several local GPUs |
| `build_bundle.sh`, `build_cluster.sh` | Build the bundle image (simulator plus the five policy environments) used with Apptainer on Slurm |
| `bootstrap.sh`, `activate.sh` | Host installation without Docker (kept for reference) |
| `download_assets.py` | Download the official RoboTwin 2.0 assets and verify them against `references/assets-tree.json` |
| `clone_policy_repos.py`, `download_models.py` | Clone the pinned policy repositories and download the pinned checkpoints |
| `prepare_g05.py` | Assemble the G0.5 runtime bundle (hard-linked checkpoint and dataset statistics) |
| `patch_robotwin.py` | Apply the guarded local changes to the pinned RoboTwin checkout |
| `doctor.py`, `cluster_preflight.py`, `prepare_cluster.py` | Hardware and runtime checks; checksum-verified downloads inside a job |

## Pools and evaluation

| Script | Purpose |
|---|---|
| `experiment.py` | Single-machine experiment driver: `init`, `freeze`, `evaluate`, `report` |
| `parallel_eval.py` | Multi-GPU scheduler: pool preparation with several workers per GPU, then evaluation with several policy replicas |
| `parallel_freeze_task.py` | Finish one task's pool with concurrent seed workers, committed in seed order |
| `freeze_suite.sh` | Freeze all 50 tasks sequentially |
| `serve_policy.py`, `wait_policy.py`, `policy_runtime.py` | Start and supervise a policy server in the model's own environment |
| `randomization_audit.py` | Check that every simulator request applied exactly the sampled condition |
| `validate_sensitive_tasks.py`, `smoke_e2e.py` | Smoke tests for the sensitive tasks and the end-to-end pipeline |

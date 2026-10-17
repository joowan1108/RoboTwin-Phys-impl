# Tools

Scripts outside the runtime fingerprint. `tools/` is mounted from the patch ZIP on the cluster, and its checksum is pinned by every launcher, so new read-only helpers go to [`analysis/`](../analysis/README.md) instead (`analysis/analyze_episodes.py` summarizes every episode record and log of a model at once).

## Environment and assets

`bootstrap.sh`, `activate.sh` (host install record), `docker.sh`, `docker_parallel.py` (local Docker runs), `build_bundle.sh`, `build_cluster.sh` (bundle image for the cluster), `clone_policy_repos.py`, `download_models.py`, `prepare_g05.py` (policy repositories and checkpoints), `download_assets.py` (official RoboTwin assets, verified against `references/assets-tree.json`), `patch_robotwin.py` (guarded local diff of the pinned RoboTwin), `doctor.py`, `cluster_preflight.py` (hardware and runtime evidence), `prepare_cluster.py` (checksum-verified downloads inside a job).

## Pools, evaluation, reports

`parallel_eval.py` (scheduler: pool preparation with N workers per GPU, policy replicas, resume, reporting), `parallel_freeze_task.py` (finish one task's pool with concurrent seed workers, committed in seed order), `freeze_suite.sh`, `reverify_pool.py`, `serve_policy.py`, `wait_policy.py`, `policy_runtime.py` (policy server supervision in the model's own environment), `experiment.py` (tables and audits), `randomization_audit.py`, `validate_sensitive_tasks.py`, `smoke_e2e.py`.

## Provenance continuations

`continue_success_context.py`, `resume_contact_pool.py`, `diagnose_provenance.py`: advance saved manifests and progress under one named rule when a reviewed source change would otherwise invalidate the fingerprint. Each writes backups and receipts under `reports/pool-preservation/`.

## Delivery packagers

`package_*.py` build the immutable patch ZIPs in `reports/` and regenerate the matching launcher in `cluster/`. Each packager checks that only the reviewed files changed relative to its baseline ZIP and refuses to overwrite a delivered archive. The newest is `package_success_context_fix.py` (v6). `package_pool_parallel.py` embeds the scheduler into the v5 parallel launcher whenever `parallel_eval.py` changes.

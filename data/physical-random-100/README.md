# RoboTwin-Phys reproduction dataset: phys-pro6000-100-v6 (physical_random)

Independent reimplementation of RoboTwin-Phys (arXiv 2609.26292); not the authors' official pool.
50 tasks, expert-filtered episodes per task, shared by every evaluated policy.

## Files

- `pool/seeds.json`: for each task, the start seed, the accepted seeds in episode order, and every
  rejected seed with its reason (`expert_failed`, `unstable_initialization`). Scanning seeds upward from
  the start seed and keeping the accepted ones reproduces the pool.
- `pool/conditions.csv`: one row per accepted episode with the 11 sampled physical attributes, the
  condition hash, the initial-state hash and the language instruction. Conditions are a pure function of
  (task, seed): `robotwin_phys.conditions.sample(task, seed)` on the pinned repository recomputes them, and
  `analysis/verify_release.py` checks every row against that function.
- `results/<model>/episodes.csv`: per-episode policy outcome (success, actions taken, wall seconds).
  `identity.json` pins the policy name, profile and adapter hashes, and the manifest hashes used.
- `tables/`: aggregate tables as written by `tools/experiment.py` (if present).
- `full/`: the sealed manifests verbatim (sampled condition, verified initial scene state, expert
  result, runtime fingerprint), the pool index and the experiment suite. `provenance.json` summarises them.
  In the GitHub repository only `full/pool-index.json` and `full/suite.json` are committed; the manifests
  (several hundred MB) and `tables/randomization.jsonl` are in the release ZIP attached to the GitHub release.

## Using the pool elsewhere

`robotwin-phys evaluate` refuses a manifest whose runtime fingerprint (source, assets, package versions,
GPU, driver) differs from the current machine, so these manifests cannot be fed directly to a different
installation. On another machine, freeze the pool from the same start seed (`robotwin-phys freeze --seed
<start_seed>` or `tools/parallel_eval.py`); the expert is deterministic per seed on a given GPU class, so
the accepted seed list should match `pool/seeds.json`, and any difference is visible seed by seed.

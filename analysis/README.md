# Analysis

Read-only scripts over the evaluation records. They are **not** part of the patch ZIPs (`tools/` is checksum-pinned by every delivered archive, so new helpers live here) and need only the Python standard library, so a copy can be uploaded to the cluster and run from the login shell.

## `analyze_episodes.py`

Summarizes every episode of one model at once: per-task success with a 95% interval next to the paper's Table 4 rate (`references/paper-table4-physical-random.json`) and an exact binomial flag, log triage by failure signature (OIDN, CUDA, timeout, state mismatch, policy errors, unclassified error lines), which sampled physical attributes separate failures from successes, record anomalies, and an episode-by-episode comparison with another run folder.

```bash
# on the cluster (upload analysis/analyze_episodes.py and references/paper-table4-physical-random.json next to each other or pass --reference)
python3 analyze_episodes.py --root ~/robotwin-data --experiment phys-pro6000-100-v6 --model fastwam \
    --reference paper-table4-physical-random.json \
    --step-limits ~/robotwin-data/vendor/RoboTwin/task_config/_eval_step_limit.yml \
    --compare ~/robotwin-data/runs/phys-pro6000-100-v6/fastwam-oidn201-discarded
python3 analyze_episodes.py ... --errors        # error-like log lines of every failed episode
python3 analyze_episodes.py ... --json report.json
```

`references/paper-table4-physical-random.json` holds the per-task Physical Random rates of the five models from Appendix D of arXiv 2609.26292v1; the per-model means reproduce the paper's overall numbers.

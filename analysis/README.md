# Analysis

Read-only scripts over evaluation records and published datasets. They need only the Python standard library (plus this repository for `verify_release.py`).

## `analyze_episodes.py`

Summarizes every episode of one model at once:
- per-task success with a 95% interval next to the paper's Table 4 rate, with an exact binomial flag;
- log triage by failure signature (renderer, CUDA, timeout, state mismatch, policy errors);
- which sampled physical attributes separate failures from successes;
- an optional episode-by-episode comparison with another run folder.

```bash
python3 analysis/analyze_episodes.py --root <data-root> --experiment <experiment> --model fastwam \
    --reference references/paper-table4-physical-random.json
python3 analysis/analyze_episodes.py ... --errors        # error lines of every failed episode
python3 analysis/analyze_episodes.py ... --json report.json
```

`references/paper-table4-physical-random.json` holds the per-task Physical Random rates of the five models from Appendix D of arXiv 2609.26292v1.

## `export_release.py` and `verify_release.py`

`export_release.py` turns one experiment into a publishable dataset folder plus a ZIP:

| Output | Contents |
|---|---|
| `pool/seeds.json` | accepted seeds in episode order, rejected seeds with reasons |
| `pool/conditions.csv` | the 11 sampled attributes, condition and initial-state hashes, instruction per episode |
| `results/<model>/` | `episodes.csv` and `identity.json` per evaluated model |
| `tables/`, `provenance.json` | aggregate tables and the runtime fingerprint |
| `full/` | the sealed manifests verbatim |

```bash
python3 analysis/export_release.py --root <data-root> --experiment <experiment>
```

`verify_release.py` recomputes every condition hash of `pool/conditions.csv` from `(task, seed)`, checks the manifest and pool-index seals, and cross-checks the results against the pool:

```bash
python3 analysis/verify_release.py data/physical-random-100
```

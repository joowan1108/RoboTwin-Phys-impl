# Published evaluation data

| Dataset | Protocol | Tasks × episodes | Results | Full archive |
|---|---|---|---|---|
| [physical-random-100](physical-random-100/) | Physical Random | 50 × 100, seeds scanned from 100000 | Fast-WAM 45.18% (paper 44.24%) | [`physical-random-100-full.zip`](https://github.com/joowan1108/RoboTwin-Phys-impl/releases/tag/data-physical-random-100) (80 MB) |

The experiment ID `phys-pro6000-100-v6` that appears inside the files is recorded in the sealed manifests and pool index, so it is kept unchanged.

Runtime: 2× NVIDIA RTX PRO 6000 Blackwell, bundle image `joowan1108/robotwin-phys:bundle-cu128-v5`, RoboTwin 2.0 `stable_2.0 @ 13c3c47`, OIDN 2.3.3 denoiser. The exact fingerprint is in `provenance.json`.

## Contents

| Path | Contents |
|---|---|
| `pool/seeds.json` | per task: accepted seeds in episode order, and every rejected seed with its reason |
| `pool/conditions.csv` | per episode: the 11 sampled physical attributes, condition and initial-state hashes, instruction |
| `results/<model>/episodes.csv` | per episode: success, actions taken, wall time |
| `results/<model>/identity.json` | checkpoint revision, profile hash and adapter hash of the evaluated policy |
| `tables/`, `summary.md` | aggregate tables and the per-task comparison with the paper |
| `full/pool-index.json`, `full/suite.json` | the sealed pool index and experiment settings |

The full archive adds the 50 sealed manifests (verified initial scenes, expert results, runtime fingerprint) and `tables/randomization.jsonl`. Unpack it into `data/`.

## Verify

```bash
python3 analysis/verify_release.py data/physical-random-100
```

This recomputes all 5000 condition hashes from the seeds, checks the pool-index seal and cross-checks the results. With the full archive unpacked it also checks the 50 manifest seals.

To evaluate your own policy on these episodes, see [Benchmark your policy with our seeds](../README.md#-benchmark-your-policy-with-our-seeds).

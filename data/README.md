# Published evaluation data

Each folder is one experiment exported with `analysis/export_release.py` and checked with `analysis/verify_release.py`.

| Experiment | Protocol | Tasks × episodes | Models with results | Release asset |
|---|---|---|---|---|
| [phys-pro6000-100-v6](phys-pro6000-100-v6/) | physical_random | 50 × 100 (seeds from 100000) | Fast-WAM 45.18% (paper 44.24%) | `phys-pro6000-100-v6-full.zip` (80 MB): adds the 50 sealed manifests and `tables/randomization.jsonl` |

Runtime of `phys-pro6000-100-v6`: 2× NVIDIA RTX PRO 6000 Blackwell, Apptainer image `joowan1108/robotwin-phys:bundle-cu128-v5`, RoboTwin 2.0 at 13c3c47, OIDN 2.3.3 denoiser, success-context fix v5 ZIP for the Fast-WAM run. The exact fingerprint is in `provenance.json`.

## What you can do with it

- **Use the same episodes.** `pool/seeds.json` lists, per task, the accepted seeds in episode order and every rejected seed with its reason. `pool/conditions.csv` has the 11 sampled physical attributes per episode. Conditions are a pure function of `(task, seed)`, so they are identical on any machine.
- **Compare a policy episode by episode.** `results/<model>/episodes.csv` has success, actions taken and wall time per `(task, episode_index, seed)`; `identity.json` pins the checkpoint revision, profile hash and adapter hash.
- **Re-freeze the pool on your hardware.** The sealed manifests carry the verified initial scene and a runtime fingerprint; `robotwin-phys evaluate` only accepts them on an identical runtime. Elsewhere, freeze from the same start seed and compare the accepted seeds against `pool/seeds.json`.

## Verify

```bash
python3 analysis/verify_release.py data/phys-pro6000-100-v6
```

recomputes all 5000 condition hashes from the seeds and checks the pool-index seal, the per-task manifest hashes and the result rows. With the release ZIP unpacked over the folder it also checks the 50 manifest seals.

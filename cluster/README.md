# Slurm templates

Templates for running the benchmark on a Slurm cluster with Apptainer and the bundle image (`tools/build_bundle.sh`). Replace `YOUR_DOCKERHUB_ID`, the partition and the GRES names with your site's values.

| File | Purpose |
|---|---|
| `evaluate-1gpu.sbatch` | One GPU: prepare the pool, then evaluate one model (`ROBOTWIN_MODEL`) with `run_parallel.sh` |
| `evaluate-3gpu.sbatch` | Two simulator GPUs and one policy GPU on one node |
| `bundle-1gpu.sbatch`, `bundle-2gpu.sbatch` | Prepare the pool once, then evaluate all five models in turn with `run_bundle.sh` |
| `run_parallel.sh`, `run_bundle.sh` | Entry points that run inside the container |
| `default-protected.txt` | Common tail shared by every template (cache location and the single `apptainer exec`) |

Resubmitting the same template after a time limit resumes the run: sealed pools and completed episodes are reused, and nothing is re-sampled.

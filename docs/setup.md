# Setup

Everything runs in containers. The host needs Linux x86_64, an NVIDIA driver with Vulkan support, and Docker (or Apptainer on a cluster). No CUDA toolkit or Python packages are installed on the host.

There are two kinds of images:

| Image | Built by | Contents | Use |
|---|---|---|---|
| `robotwin-phys:cuda12.1` | [tools/docker.sh](../tools/docker.sh) `build` ([Dockerfile](../Dockerfile)) | Simulator only: Ubuntu 22.04, CUDA 12.1.1, Python 3.10, PyTorch 2.4.1, SAPIEN, CuRobo compiled for the local GPU | Pre-Blackwell GPUs (Turing to Hopper), pool preparation, replay |
| `<repo>:bundle-cu128-<tag>` | [tools/build_cluster.sh](../tools/build_cluster.sh) + [tools/build_bundle.sh](../tools/build_bundle.sh) | Simulator on CUDA 12.8 / PyTorch 2.7.1 (CuRobo built for sm_75 and sm_120) plus the five policy environments | Blackwell GPUs, full evaluation, Slurm with Apptainer |

## 1. Source checkouts

The images copy the pinned upstream sources from `vendor/`:

```bash
git clone https://github.com/joowan1108/RoboTwin-Phys-impl.git && cd RoboTwin-Phys-impl
git clone --branch stable_2.0 https://github.com/RoboTwin-Platform/RoboTwin.git vendor/RoboTwin
git -C vendor/RoboTwin checkout 13c3c47ff4312dd62484bcd51be034af55c062d1
git clone --branch v0.7.8 https://github.com/NVlabs/curobo.git vendor/curobo
git -C vendor/curobo checkout d64c4b005459db10c5dd867d8b30a87d5bda9bdb
python3 tools/clone_policy_repos.py   # GalaxeaVLA, Motus, FACT, FastWAM at their pinned revisions
```

[tools/clone_policy_repos.py](../tools/clone_policy_repos.py) refuses to touch an existing checkout at a different revision. The image build applies [tools/patch_robotwin.py](../tools/patch_robotwin.py) to the copied RoboTwin. The host checkout stays pristine.

## 2. Simulator image (Docker)

```bash
bash tools/docker.sh build      # build robotwin-phys:cuda12.1
bash tools/docker.sh doctor     # CUDA kernel, real Vulkan camera render, dependency check
bash tools/docker.sh shell      # interactive shell in the image
bash tools/docker.sh run robotwin-phys tasks
```

The build starts from a digest-pinned CUDA 12.1.1 devel image, creates a Python 3.10 environment from [docker/requirements.lock](../docker/requirements.lock), checks the source revisions, and compiles CuRobo for the compute capability of the local GPU ([docker/build_arch.py](../docker/build_arch.py)). Set `ROBOTWIN_CUDA_ARCH='8.0;8.6'` to target other GPUs, for example when building on a machine without a GPU. Python package versions and the base image are pinned. Ubuntu's apt state is not, so builds are not bitwise reproducible.

Other `tools/docker.sh` commands:

| Command | Purpose |
|---|---|
| `validate` | Image and revision checks, CUDA/Vulkan diagnostics, the CPU test suite, and a fresh-process replay of a frozen manifest with initial-state comparison and MP4 output. Pass `--manifest manifests/<file>.json`; the default is `manifests/verified-smoke-stack-blocks.json`. |
| `config` | Print the resolved Compose configuration. |
| `save [FILE]` | Export the image as a gzip archive (`docker image load -i ...` on the target machine). Assets, checkpoints and results are not included. |

GPU access:

- `ROBOTWIN_GPU_MODE=auto` (default) uses the NVIDIA Container Toolkit when Docker has the `nvidia` runtime. Otherwise it falls back to `bind`, which passes only the selected `/dev/nvidiaN`, the control devices and the host's CUDA/NVML/Vulkan/EGL driver files, read-only. The bind fallback targets common Linux x86_64 library layouts. Prefer the Toolkit elsewhere.
- `ROBOTWIN_GPU=<index|UUID>` selects the GPU. By default the second visible GPU is used if there is one (GPU 0 often drives a desktop), otherwise the first. Each container sees exactly one GPU as `cuda:0`. Do not set host GPU numbers in the container's `CUDA_VISIBLE_DEVICES`.
- Driver capabilities are `compute,utility,graphics`, so CUDA and Vulkan both work.

The container runs as the calling user's UID/GID, without privileged mode or a Docker socket. Its working directory is `/home/guest/robotwin-workspace`, a fixed path because the generated CuRobo planner YAMLs contain absolute paths. Mounts:

| Host path | In the container |
|---|---|
| `vendor/RoboTwin/assets` | read-only |
| `checkpoints/` | read-only |
| `manifests/`, `runs/`, `reports/` | read-write, persistent |
| `.container/cache` | runtime cache (`ROBOTWIN_CACHE_DIR`) |

Do not mount the whole repository over the workspace: it would hide the image's Python environment.

## 3. Assets

[tools/download_assets.py](../tools/download_assets.py) downloads `embodiments.zip`, `objects.zip` and `background_texture.zip` from the [RoboTwin 2.0 dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0), resumes partial downloads, verifies each archive against the Git LFS SHA-256 in [references/assets-tree.json](../references/assets-tree.json), and extracts it into `vendor/RoboTwin/assets` (about 15 GB to download, 16 GB extracted). Assets that differ from the pinned hashes change the runtime fingerprint and cannot be used with existing manifests.

The CuRobo planner configurations (`assets/embodiments/aloha-agilex/curobo_left.yml`, `curobo_right.yml`) are generated from templates by RoboTwin's `script/update_embodiment_config_path.py`. They contain absolute paths and must be generated at the in-container path. [tools/prepare_cluster.py](../tools/prepare_cluster.py) `--assets` performs the download, the verification and this generation step, and fails if a planner configuration is missing. Run it once inside a container whose asset directory is writable, for example:

```bash
export IMAGE=robotwin-phys:cuda12.1          # or the bundle image
export DATA="$HOME/robotwin-data"
mkdir -p "$DATA"/{assets,checkpoints,downloads,manifests,runs,reports,cache/home}
W=/home/guest/robotwin-workspace
docker run --rm -it --gpus all --shm-size=8g --user "$(id -u):$(id -g)" \
  -e HOME=/tmp/robotwin-home -v "$DATA/cache/home:/tmp/robotwin-home" \
  -v "$DATA/assets:$W/vendor/RoboTwin/assets" -v "$DATA/checkpoints:$W/checkpoints" \
  -v "$DATA/downloads:$W/downloads" -v "$DATA/manifests:$W/manifests" \
  -v "$DATA/runs:$W/runs" -v "$DATA/reports:$W/reports" \
  "$IMAGE" python tools/prepare_cluster.py --assets
```

The same `docker run` form, with `bash` as the command, works for all later steps when you keep data outside the repository.

## 4. Model checkpoints

Run these inside a container with `checkpoints/` writable (the `docker run` form above):

```bash
python tools/download_models.py --models pi05 fact motus fastwam
huggingface-cli login                         # after accepting the gated OpenGalaxea/G05 terms
python tools/download_models.py --models g05
python tools/prepare_g05.py
```

[tools/download_models.py](../tools/download_models.py) fetches only the RoboTwin weights and statistics at the revisions pinned in [configs/policies](../configs/policies), never optimizer states or other benchmarks. It verifies SHA-256 and writes `download-lock.json`. Re-running resumes partial files. The backbones listed in [configs/backbones.json](../configs/backbones.json) (Wan2.2-TI2V-5B, Qwen3-VL-2B-Instruct, Wan2.2-TI2V-5B-Diffusers, the Wan2.1 UMT5 tokenizer) and the π0.5 PaliGemma tokenizer are downloaded and verified by `python tools/prepare_cluster.py --models <name> ...`. That command also runs `prepare_g05.py` for G0.5. [policies.md](policies.md) describes each model.

## 5. Freeze, evaluate, watch

```bash
# Expert-verified pool for one task (never overwrites a frozen manifest; --resume continues)
robotwin-phys freeze --task stack_blocks_two --protocol physical_random \
  --episodes 100 --seed 100000 --output manifests/physical_random/stack_blocks_two.json

# Evaluate a running policy server on that pool
robotwin-phys evaluate --manifest manifests/physical_random/stack_blocks_two.json \
  --output runs/fastwam/stack_blocks_two --policy-port 9000 --live --video

# Rebuild the verified initial scenes in fresh processes, without a policy
robotwin-phys replay --manifest manifests/physical_random/stack_blocks_two.json \
  --output runs/replay/stack_blocks_two --video
```

- `--live` serves the head camera, seed and stage as MJPEG at `http://127.0.0.1:8765` (`--port` to change). It works headless. From another machine, tunnel with `ssh -N -L 8765:127.0.0.1:8765 user@server`.
- `--video` writes a 320×240, 30 fps MP4 per episode, captured on the simulation clock.
- `--viewer` opens the native SAPIEN viewer and requires `DISPLAY`.
- All services bind to localhost. A policy on another machine is reached with `ssh -L 9000:127.0.0.1:9000 user@policy-host`.

### Several GPUs on one Docker host

[tools/docker_parallel.py](../tools/docker_parallel.py) uses only the host's Python standard library. It starts one simulator container per GPU, with separate live ports (from 8765, `--base-port`), reports and caches, and assigns whole tasks to the containers:

```bash
python3 tools/docker_parallel.py validate --gpus auto
python3 tools/docker_parallel.py freeze --gpus 0 1 --live --video
```

`--gpus` takes `auto`, indices or UUIDs. Missing or duplicate GPUs are rejected before anything starts. Completed manifests are verified and reused, and partial tasks continue with `--resume`. When two evaluations run at the same time, give each its own policy server port (9000, 9001), output directory, live port, `ROBOTWIN_GPU`, `ROBOTWIN_REPORTS_DIR` and `ROBOTWIN_CACHE_DIR`. A stateful policy server must never be shared by two concurrent evaluations.

## 6. Blackwell bundle image

The CUDA 12.1 stack does not support Blackwell (sm_120). For Blackwell GPUs, and for running everything from one image, build the bundle on a development machine with Docker:

```bash
export HUB_REPO=<dockerhub-user>/robotwin-phys
bash tools/build_cluster.sh "$HUB_REPO" v1          # sim-cu128-v1 + policy-<model>-cu128-v1 (six images)
bash tools/build_bundle.sh "$HUB_REPO" v1 "$HUB_REPO" v1   # -> $HUB_REPO:bundle-cu128-v1
docker push "$HUB_REPO:bundle-cu128-v1"
```

- [docker/Dockerfile.blackwell](../docker/Dockerfile.blackwell) builds the simulator on CUDA 12.8.1 with `TORCH_CUDA_ARCH_LIST=7.5;12.0`.
- [docker/Dockerfile.policy](../docker/Dockerfile.policy) builds one environment per model from [docker/policies](../docker/policies). π0.5 uses Python 3.11, the others Python 3.10, all on PyTorch 2.7.1+cu128.
- [docker/Dockerfile.bundle](../docker/Dockerfile.bundle) copies the policy environments into `/opt/robotwin-policies/<model>/{venv,workspace,build}` on top of the simulator image. Packages are never merged across environments.

The scheduler starts each policy server with its own interpreter inside the same container. No nested containers or root access are needed. Keep the pushed digest: replacing an image in the middle of an experiment changes the runtime or policy identity, and the evaluation then refuses to continue.

Before evaluating policies on Blackwell, install OIDN 2.3.3 as described next.

### OIDN 2.3.3 on Blackwell

SAPIEN 3.0.0b1 bundles OIDN 2.0.1, which fails silently on Blackwell and leaves policy observations un-denoised ([design.md](design.md#oidn-denoiser-on-blackwell)). The fix needs no image rebuild:

1. Download the official `oidn-2.3.3.x86_64.linux.tar.gz` release, verify its checksum, and collect its `lib/*.so*` files in one directory.
2. Bind that directory read-only over `.../site-packages/sapien/oidn_library` of the simulator environment.
3. Bind [docker/oidn/_oidn_tricks.py](../docker/oidn/_oidn_tricks.py) over `.../site-packages/sapien/_oidn_tricks.py`, so that SAPIEN preloads the 2.3.3 libraries.
4. Probe before evaluating. Render one scene with the `rt` shader and the `oidn` denoiser, check that `/proc/self/maps` lists `libOpenImageDenoise_device_cuda.so.2.3.3`, and check that stderr contains no `OIDN Error`. During evaluation, worker logs must contain no `OIDN Error: invalid handle` lines.

## 7. Slurm with Apptainer

On a cluster, pull the bundle image with Apptainer and run [cluster/run_parallel.sh](../cluster/run_parallel.sh) inside it. The script calls [tools/parallel_eval.py](../tools/parallel_eval.py) with the simulator interpreter and passes extra arguments through. [cluster/](../cluster/README.md) has templates for the surrounding batch script:

- [evaluate-1gpu.sbatch](../cluster/evaluate-1gpu.sbatch): one GPU, pool preparation and one model;
- [evaluate-3gpu.sbatch](../cluster/evaluate-3gpu.sbatch): two simulator GPUs and one policy GPU on one node;
- `bundle-1gpu.sbatch` and `bundle-2gpu.sbatch`: evaluate all five models in turn through [cluster/run_bundle.sh](../cluster/run_bundle.sh).

Set `IMAGE` in the template to the tag you pushed, and adapt the partition and GRES names to your site. A batch script has to:

- request one node with 1, 2 or 3 GPUs (`--gres`), enough CPUs and RAM (8 pool workers per GPU used 4 CPUs each), and a time limit;
- keep Slurm's `CUDA_VISIBLE_DEVICES` and pass it into the container (`APPTAINERENV_CUDA_VISIBLE_DEVICES`). The scheduler only uses allocated GPUs;
- bind persistent directories into the workspace, for example with `APPTAINER_BINDPATH`:

  | `$DATA/...` | Container path |
  |---|---|
  | `assets` | `/home/guest/robotwin-workspace/vendor/RoboTwin/assets` |
  | `checkpoints`, `downloads`, `manifests`, `runs`, `reports` | `/home/guest/robotwin-workspace/<same name>` |
  | `cache/home` | a writable `HOME`, for example `/tmp/robotwin-home` |

  Also bind the host Vulkan ICD directories (`/usr/share/vulkan/icd.d`, `/etc/vulkan/icd.d`) read-only if present;
- set `APPTAINER_CLEANENV=1`, `APPTAINER_NO_HOME=1` and a node-local `APPTAINER_CACHEDIR`;
- configure the run through `APPTAINERENV_` variables read by `run_parallel.sh`:

  | Variable | Meaning | Default |
  |---|---|---|
  | `ROBOTWIN_MODEL` | `fastwam`, `g05`, `pi05`, `motus` or `fact` | `fastwam` |
  | `ROBOTWIN_EXPERIMENT` | experiment directory name shared by all models | — set it explicitly |
  | `ROBOTWIN_PHASE` | `all` (assets, preflight, pool, model download, evaluation, tables), `freeze`, `evaluate`, `report` | `all` |
  | `ROBOTWIN_EPISODES` | episodes per task | 100 |
  | `ROBOTWIN_VIDEO_SELECTION` | `first` (episode 0 of each task), `none`, `all` | `first` |
  | `ROBOTWIN_POLICY_REPLICAS` | policy replicas ([design.md](design.md#multi-gpu-scheduling)) | by layout |

- start the container:

  ```bash
  apptainer exec --nv docker://<dockerhub-user>/robotwin-phys:bundle-cu128-v1 \
    bash /home/guest/robotwin-workspace/cluster/run_parallel.sh --freeze-workers-per-gpu 8 --timeout 3600
  ```

A gated G0.5 download needs `APPTAINERENV_HF_TOKEN`. Keep tokens out of images and configuration files.

With `phase=all`, a job prepares assets, checks CUDA, Vulkan and CuRobo on every allocated GPU, freezes or verifies the pool, downloads the selected model, evaluates it, and writes tables to `$DATA/runs/<experiment>/tables/`. Logs are written to `$DATA/reports/parallel/<experiment>/<model>-<jobid>/`. When a job hits its time limit, submit the same script again: sealed pools and verified episodes are reused. To evaluate the next model on the same pool, change only `ROBOTWIN_MODEL` and set `ROBOTWIN_PHASE=evaluate`. Keep the experiment name, image, assets and episode settings. `evaluate` never samples new seeds: it refuses to run if the pool index, a manifest or the runtime fingerprint differs.

Pools prepared on different GPU models or drivers have different fingerprints. Keep them in separate experiments.

## Publishing and checking a pool

`python3 analysis/export_release.py --root <data root> --experiment <id>` exports seeds, conditions, per-episode results, policy identity and the sealed manifests. `python3 analysis/verify_release.py <folder>` recomputes every condition from its `(task, seed)` and checks the hashes ([analysis/README.md](../analysis/README.md)). The published pool is in [data/physical-random-100](../data/physical-random-100/README.md) (experiment ID `phys-pro6000-100-v6`).

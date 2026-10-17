# Policies

The five baselines of the paper are connected through their official RoboTwin 2.0 adapters. Every policy runs as a server in its own Python environment and is evaluated on the same frozen episode manifests ([design.md](design.md#evaluation)). That shared manifest is what makes the environment conditions identical across models. Precision, normalization and action chunking follow each model's own specification and are pinned in its profile under [configs/policies](../configs/policies).

## Checkpoints and code

| Model | Checkpoint (Hugging Face) | Files used | Code | Adapter |
|---|---|---|---|---|
| Fast-WAM | [`yuanty/fastwam`](https://huggingface.co/yuanty/fastwam) @ `8eaceeb24c3c` | `robotwin_uncond_3cam_384.pt`, `robotwin_uncond_3cam_384_dataset_stats.json` | [FastWAM](https://github.com/yuantianyuan01/FastWAM) @ `7faa71108368` | `experiments/robotwin/fastwam_policy/deploy_policy.py` |
| G0.5 | [`OpenGalaxea/G05`](https://huggingface.co/OpenGalaxea/G05) @ `e312be81e90c` (gated) | `g05-robotwin20/` (checkpoint, dataset stats, Hydra config), `action_tokenizer.pt`, `qwen3_5_2b_base_processor/` | [GalaxeaVLA](https://github.com/OpenGalaxea/GalaxeaVLA) @ `89f2322b4ad0` | `experiments/robotwin/galaxeafm_policy/deploy_policy.py` |
| π0.5 | [`motus-robotics/pi0.5_robotwin2`](https://huggingface.co/motus-robotics/pi0.5_robotwin2) @ `effc6e1a99bd` | `model.safetensors`, `assets/pi0.5_clean_randomize_joint_training/norm_stats.json` | OpenPI from RoboTwin's `policy/pi05` (pinned RoboTwin revision) | [src/robotwin_phys/pi05_policy.py](../src/robotwin_phys/pi05_policy.py) |
| Motus | [`motus-robotics/Motus_robotwin2`](https://huggingface.co/motus-robotics/Motus_robotwin2) @ `ab74bc214f7f` | `mp_rank_00_model_states.pt` | [Motus](https://github.com/thu-ml/Motus) @ `f771216802b8` | `inference/robotwin/Motus/deploy_policy.py` |
| FACT | [`Bariona/fact-wam`](https://huggingface.co/Bariona/fact-wam) @ `887f66bc0be8` | `transformer/`, `norm_stats_delta.json` | [FACT](https://github.com/Bariona/FACT) @ `5b299a2059f0` | `evaluation/robotwin/model2robotwin_interface.py` (client of FACT's native server) |

The full revisions are in the profiles. The Hugging Face metadata captured at those revisions is kept in `references/*-info.json`, and the public model cards in `references/*-card.md`.

Backbones, from [configs/backbones.json](../configs/backbones.json):

| Model | Additional files |
|---|---|
| Fast-WAM | `Wan-AI/Wan2.2-TI2V-5B`, UMT5 tokenizer from `Wan-AI/Wan2.1-T2V-1.3B` |
| Motus | `Wan-AI/Wan2.2-TI2V-5B` (including the T5 encoder and VAE, which the code loads), `Qwen/Qwen3-VL-2B-Instruct` (config and tokenizer files) |
| FACT | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (VAE, text encoder, tokenizer, scheduler) |
| π0.5 | PaliGemma tokenizer (`big_vision/paligemma_tokenizer.model`) |

## Inference settings

| Model | Native bf16 GPU required | Action execution | Other pinned arguments |
|---|---|---|---|
| Fast-WAM | yes (`mixed_precision: bf16`) | `replan_steps: 24` | `sigma_shift: 5.0`, `sim_task: robotwin_uncond_3cam_384_1e-4`, `sim_cfg_name: sim_robotwin.yaml` |
| G0.5 | yes (`mixed_precision: bf16`) | `replan_steps: 24` | `sim_task: robotwin`, `sim_cfg_name: sim_robotwin.yaml`; inference steps from the model config (10) |
| π0.5 | yes | `replan_steps: 32` (horizon 32) | `num_inference_steps: 10`, `adapt_to_pi=True`, delta joint actions with absolute grippers |
| Motus | yes, plus `minimum_vram_gb: 24` | official adapter default | Wan and Qwen backbone paths |
| FACT | not checked by the bridge | `execute_actions_per_plan: 48` | native server at `127.0.0.1:8093` |

These values come from the public evaluation code and checkpoint cards. They are not confirmed to match the RoboTwin-Phys authors' unpublished hyperparameters. Changing any of them changes the profile hash, and therefore the policy identity, so results must go to a new run directory.

Notes per model:

- **Fast-WAM.** The pinned README states that the released checkpoints are evaluated with `EVALUATION.sigma_shift=5.0`. Upstream `sim_robotwin.yaml` leaves it null, and the action scheduler then falls back to `infer_shift: 1.0` from `configs/model/fastwam.yaml`, a later training default. The profile sets 5.0 explicitly. The image sets `redirect_common_files: false` in the FastWAM model configs, so the Wan files load from the pinned Hugging Face downloads instead of an automatically downloaded ModelScope conversion. Loader and weights are unchanged.
- **G0.5.** The profile points to `checkpoints/g05-runtime/`, built by [tools/prepare_g05.py](../tools/prepare_g05.py). The adapter resolves the checkpoint path and reads `../../.hydra/config.yaml` next to the resolved file. The runtime bundle therefore contains a rewritten config with absolute processor and action tokenizer paths, and hard links (or copies) of the checkpoint and dataset stats. Symlinks would resolve back into the original download, whose config uses repository-relative paths. The original download and its checksums stay untouched. `sim_task: robotwin` reproduces what the upstream evaluator `eval_robotwin_single.py` always passes: the adapter turns it into the Hydra override `task=robotwin`, which selects the `data`, `model` and `tokenizer` groups. The upstream training config's `seed: 7` has no effect here, because the policy server reseeds every episode.
- **π0.5.** [pi05_policy.py](../src/robotwin_phys/pi05_policy.py) implements the transforms documented on the checkpoint card: Aloha data config, `adapt_to_pi=True`, delta joints with absolute grippers, action horizon 32 and the supplied normalization statistics. RoboTwin's stock pi05 configuration uses `adapt_to_pi=False` and is not used. The model runs as a PyTorch checkpoint. JAX is restricted to the CPU so it does not reserve GPU memory.
- **Motus.** The model card gives about 41 GB of GPU memory with the T5 encoder on the same device and about 24 GB with T5 pre-encoding. The profile enforces `minimum_vram_gb`. An optional placement puts only the T5 encoder on a second GPU. It is patched into the official adapter as an explicit `t5_device` argument ([docker/policies/patch_motus.py](../docker/policies/patch_motus.py)) and recorded in the profile hash. The default keeps the original placement.
- **FACT.** The RPC bridge runs FACT's official RoboTwin client, which talks to FACT's native inference server. The native server holds the model's RNG, so [docker/policies/patch_fact.py](../docker/policies/patch_fact.py) forwards an episode-derived seed to it instead of a time-based seed. Keep the native server's settings with the results. The scheduler runs exactly one FACT replica, because the native port is part of the policy identity.

## Environments

Each model needs its own environment, because the stacks conflict with each other and with the simulator (Python 3.10, NumPy 1.26, SAPIEN 3):

| Model | Upstream environment | Notes |
|---|---|---|
| Fast-WAM | PyTorch 2.7.1+cu128, NumPy 2.2.6 | |
| G0.5 | Python 3.10, PyTorch 2.7.1, Transformers 4.57.1 | Its default SAPIEN 2.x dependency conflicts with the simulator. Benchmark and training-only dependencies are left out, and the code's default attention fallback is allowed. |
| π0.5 | Python ≥ 3.11, OpenPI with JAX, Transformers, PyTorch | Transformers 4.53.2 with OpenPI's `transformers_replace`. LeRobot is installed with `--no-deps` for OpenPI's type imports, so its missing training extras are expected in `pip check`. |
| Motus | CUDA ≥ 12.6 plus model dependencies | FlashAttention 2.8.3.post1 wheel, verified by SHA-256 and checked for sm_120 kernels |
| FACT | PyTorch 2.6 / cu124 upstream | Raised to 2.7.1 / cu128 in the Blackwell image |

The bundle image ([setup.md](setup.md#6-blackwell-bundle-image)) contains these environments under `/opt/robotwin-policies/<model>/venv`, built from [docker/Dockerfile.policy](../docker/Dockerfile.policy) and the requirement files in [docker/policies](../docker/policies). Package freezes and code hashes of each environment are recorded with the image. They are not claimed to equal the authors' environment.

The bridge itself needs only NumPy and PyYAML and never imports SAPIEN or CuRobo. In a model environment, expose it with `PYTHONPATH=<repo>/src` rather than `pip install -e .`, because the project package is restricted to the simulator's Python 3.10.

## Preparation

```bash
python tools/clone_policy_repos.py                              # policy code at pinned revisions
python tools/download_models.py --models pi05 fact motus fastwam
huggingface-cli login                                           # after accepting the G05 terms
python tools/download_models.py --models g05
python tools/prepare_cluster.py --models fastwam                # backbones (+ prepare_g05.py for g05)
```

[tools/download_models.py](../tools/download_models.py) downloads only the RoboTwin weights and statistics, never optimizer states or other benchmarks. It verifies SHA-256 against the captured metadata and writes `download-lock.json`. Partial files are resumed by the same command and must not be used before it completes. G0.5 is gated: access must be granted on Hugging Face before the download works.

## Running a policy server

```bash
export PYTHONPATH=/path/to/RoboTwin-Phys-impl/src
/path/to/fastwam-env/bin/python -m robotwin_phys.policy_server \
  --config configs/policies/fastwam.yaml --port 9000
```

Use the same form with `g05.yaml`, `pi05.yaml` and `motus.yaml`. One server can process many tasks sequentially, but serves only one evaluation at a time. FACT needs its native server first, in the FACT environment:

```bash
cd vendor/FACT
/path/to/fact-env/bin/python -m scripts.inference_server \
  --model_id checkpoints/backbones/Wan2.2-TI2V-5B-Diffusers \
  --transformer_path checkpoints/fact/transformer \
  --stats_path checkpoints/fact/norm_stats_delta.json --port 8093
# second terminal, same environment
PYTHONPATH=/path/to/RoboTwin-Phys-impl/src /path/to/fact-env/bin/python -m robotwin_phys.policy_server \
  --config configs/policies/fact.yaml --port 9000
```

Use paths that match your checkout. Then connect the simulator with `robotwin-phys evaluate --manifest ... --policy-port 9000`. Inside the bundle image, [tools/parallel_eval.py](../tools/parallel_eval.py) starts and supervises the servers itself.

On startup the server checks the adapter's git revision, the required checkpoint paths, `minimum_vram_gb` and, for the bf16 profiles, a GPU of compute capability 8.0 or newer. On an older GPU such as the RTX 2080 Ti (sm_75) it exits with an error instead of running at a different precision. An FP16 or CPU-offload variant would be a different profile and would need its own validation. Apart from `minimum_vram_gb`, memory fit is not checked without actually loading the model.

## Status

Fast-WAM is fully evaluated on the published pool: 45.18% against 44.24% in the paper's Table 1 (Physical Random, mean of 50 per-task rates). Per-episode outcomes are in [data/physical-random-100](../data/physical-random-100/README.md). The other baselines have not been fully evaluated yet.

## Adding your own policy

1. Write an adapter module with the three functions of the official RoboTwin policies:

   ```python
   # my_policy/deploy_policy.py
   def get_model(args):                 # args = the "args" block of the profile; called once
       return load_my_model(args["ckpt"])

   def reset_model(model):              # called before every episode, after the server reseeds RNGs
       model.reset()

   def eval(env, model, observation):   # called until the episode ends; must take at least one action
       for action in model.predict(observation, env.get_instruction()):
           env.take_action(action)       # 14-dim joint target (qpos) by default
   ```

   `observation` is RoboTwin's observation dictionary (camera RGB images and `joint_action`). `env` is a proxy that forwards `get_obs`, `get_instruction`, `take_action` and `check_success`, plus attributes such as `take_action_cnt`, `step_lim` and `eval_success`, to the simulator. [src/robotwin_phys/noop_policy.py](../src/robotwin_phys/noop_policy.py) is a minimal working adapter that holds the current joint positions.

2. Write a profile, following [configs/policies/noop.yaml](../configs/policies/noop.yaml):

   ```yaml
   # configs/policies/my_policy.yaml
   name: my_policy
   adapter: my_policy/deploy_policy.py      # relative to the repository root
   python_paths: [my_policy]                # prepended to sys.path
   args:
     ckpt: /path/to/checkpoint
   ```

   Optional keys: `working_directory`, `code_revision` (checked against the adapter's git checkout), `checkpoint` (recorded in the identity), `required_paths`, `requires_native_bf16` and `minimum_vram_gb`. `${ROOT}` expands to the repository root.

3. Start the server in your policy's own environment and evaluate:

   ```bash
   PYTHONPATH=$PWD/src /path/to/my-env/bin/python -m robotwin_phys.policy_server \
     --config configs/policies/my_policy.yaml --port 9000
   robotwin-phys evaluate --manifest manifests/<task>.json --output runs/my_policy/<task> --policy-port 9000
   ```

The profile hash and adapter hash become part of the run's identity, so editing either requires a new output directory.

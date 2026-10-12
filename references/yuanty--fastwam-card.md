---
pretty_name: FastWAM Checkpoints
library_name: pytorch
tags:
- robotics
- robot-learning
- imitation-learning
- embodied-ai
- libero
- robotwin
- world-action-model
- optional-idm
---

# FastWAM Checkpoints

Official checkpoints for **Fast-WAM: Do World Action Models Need Test-time Future Imagination?**

This repository includes the original FastWAM checkpoints and the new
**Optional IDM** variant: one model with two inference modes for studying when
test-time future imagination helps.

## Files

```text
README.md
libero_uncond_2cam224.pt
libero_uncond_2cam224_dataset_stats.json
libero_optional_idm_2cam224.pt
libero_optional_idm_2cam224_dataset_stats.json
robotwin_uncond_3cam_384.pt
robotwin_uncond_3cam_384_dataset_stats.json
```

## Included checkpoints

- `libero_uncond_2cam224.pt`: FastWAM Base for
  `task=libero_uncond_2cam224_1e-4`.
- `libero_optional_idm_2cam224.pt`: Optional IDM for
  `task=libero_optional_idm_2cam224_1e-4`.
- `robotwin_uncond_3cam_384.pt`: FastWAM Base for
  `task=robotwin_uncond_3cam_384_1e-4`.
- Each checkpoint has a matching `*_dataset_stats.json` file for evaluation.

## Optional IDM: one model, two thinking modes

The Optional IDM checkpoint supports two action-inference paths without
retraining:

- `idm`: imagine the future video first, then predict actions.
- `first_frame`: predict actions directly from the current observation without
  test-time future imagination.

Evaluate either mode with the FastWAM repository:

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_optional_idm_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_optional_idm_2cam224.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_optional_idm_2cam224_dataset_stats.json \
  EVALUATION.sigma_shift=1.0 \
  +EVALUATION.action_infer_mode=idm \
  MULTIRUN.num_gpus=8
```

Replace `idm` with `first_frame` to use the Fast-WAM inference mode. The
released checkpoint uses an action scheduler shift of `1.0` for training and
evaluation.

On the full LIBERO benchmark (40 tasks, 50 episodes per task), this checkpoint
achieves:

| Inference mode | Spatial | Goal | Object | Long | Average |
| --- | ---: | ---: | ---: | ---: | ---: |
| IDM | 99.0% | 98.6% | 99.6% | 97.0% | **98.55%** |
| First frame (Fast-WAM) | 98.2% | 97.8% | 99.2% | 95.8% | **97.75%** |

## Project

- Project page: https://yuantianyuan01.github.io/FastWAM/
- Paper: https://arxiv.org/abs/2603.16666
- Code: https://github.com/yuantianyuan01/FastWAM

## Citation

```bibtex
@misc{yuan2026fastwam,
  title={Fast-WAM: Do World Action Models Need Test-time Future Imagination?},
  author={Tianyuan Yuan and Zibin Dong and Yicheng Liu and Hang Zhao},
  year={2026},
  note={arXiv preprint arXiv:2603.16666}
}
```

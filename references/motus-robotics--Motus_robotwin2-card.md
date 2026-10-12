---
license: apache-2.0
language:
- en
pipeline_tag: robotics
library_name: transformers
tags:
- Motus
- Vision-Language-Action
- World-Model
- Bimanual
- Manipulation
- RoboTwin
- Simulation
- Flowmatching
- Diffusion
---

# Motus: RoboTwin 2.0 Fine-Tuned Checkpoint

**Motus** is a **unified latent action world model** that leverages existing pretrained models and rich, sharable motion information. Motus introduces a **Mixture-of-Transformers (MoT)** architecture to integrate three experts (understanding, action, and video generation) and adopts a **UniDiffuser-style scheduler** to enable flexible switching between different modeling modes (World Models, Vision-Language-Action Models, Inverse Dynamics Models, Video Generation Models, and Video-Action Joint Prediction Models). Motus further leverages **optical flow** to learn **latent actions** and adopts a **three-phase training pipeline** and **six-layer data pyramid**, thereby extracting pixel-level "delta action" and enabling large-scale action pretraining.

This checkpoint is fine-tuned on **RoboTwin 2.0** benchmark (50+ manipulation tasks).

[**Homepage**](https://motus-robotics.github.io/motus) | [**GitHub**](https://github.com/thu-ml/Motus.git) | [**arXiv**](https://arxiv.org/abs/2512.13030) | [**Feishu**](https://motus-robotics.github.io/assets/motus/png/feishu.jpg) | [**WeChat**](https://motus-robotics.github.io/assets/motus/png/wechat.jpg)

---

## Table of Contents

- [Highlights](#highlights)
- [Model Details](#model-details)
- [Performance](#performance)
- [Hardware & Software Requirements](#hardware--software-requirements)
- [Quickstart (Inference)](#quickstart-inference)
- [Citation](#citation)

---

## Highlights

- **87.02%** average success rate on RoboTwin 2.0 (+15% over X-VLA, +45% over π₀.₅)
- **50+ Manipulation Tasks**: Trained on diverse bimanual manipulation scenarios
- **Multi-Task Capable**: Single model handles all 50+ tasks
- **Ready for Deployment**: Direct inference or further fine-tuning

---

## Model Details

### Architecture

| Component | Base Model | Parameters |
|-----------|------------|------------|
| **VGM (Video Generation Model)** | WAN 2.2 | ~5.00B |
| **VLM (Vision-Language Model)** | Qwen3-VL-2B | ~2.13B |
| **Action Expert** | - | ~641.5M |
| **Understanding Expert** | - | ~253.5M |
| **Total** | - | **~8B** |

### Training Details

- **Base Checkpoint**: [`motus-robotics/Motus`](https://huggingface.co/motus-robotics/Motus) (Stage 2 pretrained)
- **Fine-Tuning Data**: RoboTwin 2.0 (2,500 clean + 25,000 randomized demonstrations)
- **Training Steps**: 40k steps

### Action Representation

- **Control frequency**: 30Hz (default)
- **Action chunk size**: 48 steps (default)
- **Action dimension**: 14 (bimanual: 7 per arm)

---

## Performance

### RoboTwin 2.0 Benchmark (50+ Tasks)

| Method | Clean | Randomized |
|--------|-------|------------|
| π₀.₅ | 42.98% | 43.84% |
| X-VLA | 72.80% | 72.84% |
| **Motus (Ours)** | **88.66%** | **87.02%** |

**Key improvements:**
- +15% over X-VLA
- +45% over π₀.₅

---

## Hardware & Software Requirements

| Mode | VRAM | Recommended GPU |
|------|------|-----------------|
| Inference (with pre-encoded T5) | ~ 24 GB | RTX 5090 |
| Inference (without pre-encoded T5) | ~ 41 GB | A100 (40GB) / A100 (80GB) / H100 / B200 |

---

## Quickstart (Inference)

### RoboTwin 2.0 Simulation

```bash
cd inference/robotwin/Motus

# Single task evaluation
bash eval.sh place_dual_shoes

# Multi-task batch evaluation
bash auto_eval.sh
```

### Offline Inference (No Environment)

```bash
python inference/real_world/Motus/inference_example.py \
  --model_config inference/real_world/Motus/utils/robotwin.yml \
  --ckpt_dir ./pretrained_models/Motus_robotwin2 \
  --wan_path /path/to/pretrained_models \
  --image /path/to/input_frame.png \
  --instruction "pick up the cube and place it on the right" \
  --use_t5 \
  --output result.png
```

### Python API

```python
import torch
import yaml
from models.motus import Motus, MotusConfig

# Load config
with open("configs/robotwin.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize model
model_config = MotusConfig(
    wan_checkpoint_path=config['model']['wan']['checkpoint_path'],
    vae_path=config['model']['wan']['vae_path'],
    wan_config_path=config['model']['wan']['config_path'],
    vlm_checkpoint_path=config['model']['vlm']['checkpoint_path'],
    action_dim=14,
    load_pretrained_backbones=False,
)

model = Motus(model_config).to("cuda").eval()
model.load_checkpoint("./pretrained_models/Motus_robotwin2", strict=False)

# Inference
with torch.no_grad():
    predicted_frames, predicted_actions = model.inference_step(
        first_frame=frame_tensor,
        state=state_tensor,
        num_inference_steps=20,
        language_embeddings=t5_embeddings,
        vlm_inputs=[vlm_inputs],
    )

# Action chunk: [1, 48, 14]
actions = predicted_actions.squeeze(0).cpu().numpy()
```

---

## Citation

```bibtex
@misc{bi2025motusunifiedlatentaction,
            title={Motus: A Unified Latent Action World Model}, 
            author={Hongzhe Bi and Hengkai Tan and Shenghao Xie and Zeyuan Wang and Shuhe Huang and Haitian Liu and Ruowen Zhao and Yao Feng and Chendong Xiang and Yinze Rong and Hongyan Zhao and Hanyu Liu and Zhizhong Su and Lei Ma and Hang Su and Jun Zhu},
            year={2025},
            eprint={2512.13030},
            archivePrefix={arXiv},
            primaryClass={cs.CV},
            url={https://arxiv.org/abs/2512.13030}, 
}
```

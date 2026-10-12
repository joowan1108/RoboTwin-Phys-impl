---
base_model: Wan-AI/Wan2.2-TI2V-5B-Diffusers
license: apache-2.0
pipeline_tag: robotics
tags:
- world-action-model
- robotics
- robotwin
---

# FACT — RoboTwin Checkpoint

Action-transformer checkpoint for **FACT** (Failure-Aware Causal Training for World-Action Models), fine-tuned from [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) on the RoboTwin 2.0 benchmark ([data](https://huggingface.co/datasets/Bariona/robotwin-v2)).

**Paper**: [FACT: Failure-Aware Causal Training for World-Action Models](https://huggingface.co/papers/2608.10232)  
**Project page**: [https://fact-wam.github.io](https://fact-wam.github.io)  
**Code**: [https://github.com/Bariona/FACT](https://github.com/Bariona/FACT)

## Usage

The repo ships the matching normalization stats (`norm_stats_delta.json`), so inference/eval needs no dataset download or training:

```bash
huggingface-cli download Bariona/fact-wam --local-dir ./models/fact-wam

python -m scripts.inference_server \
  --model_id ./models/Wan2.2-TI2V-5B-Diffusers \
  --transformer_path ./models/fact-wam/transformer \
  --stats_path ./models/fact-wam/norm_stats_delta.json \
  --port 8093
```

For closed-loop RoboTwin evaluation, set `TRANSFORMER_PATH=./models/fact-wam/transformer` and `STATS_PATH=./models/fact-wam/norm_stats_delta.json` in `evaluation/robotwin/launch_config.yml`.
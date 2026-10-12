#!/usr/bin/env python3
"""Create a deployment bundle with local processor/tokenizer paths.

The gated downloaded originals and their checksums are preserved. Requires G05
access approval and a completed `download_models.py --models g05` first.
"""
from pathlib import Path
import copy
import yaml

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "checkpoints/g05"
bundle = ROOT / "checkpoints/g05-runtime"
original = source / "g05-robotwin20/.hydra/config.yaml"
if not original.exists():
    raise SystemExit("G05 config missing. Obtain HF access and download the pinned RoboTwin checkpoint first.")
config = yaml.safe_load(original.read_text())
processor = str(source / "qwen3_5_2b_base_processor")
tokenizer = str(source / "action_tokenizer.pt")
config["model"]["model_arch"]["hf_processor_path"] = processor
config["model"]["processor"]["tokenizer_params"]["pretrained_model_name_or_path"] = processor
config["tokenizer"]["vq_config"]["ckpt_dir"] = tokenizer
if isinstance(config["model"].get("tokenizer"), dict):
    config["model"]["tokenizer"]["vq_config"]["ckpt_dir"] = tokenizer
for relative in ("checkpoints/model_state_dict.pt", "dataset_stats.json"):
    target = bundle / relative
    origin = source / "g05-robotwin20" / relative
    if not origin.exists():
        raise FileNotFoundError(origin)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.symlink_to(origin)
    elif target.resolve() != origin.resolve():
        raise RuntimeError(f"Not replacing existing unrelated file: {target}")
(bundle / ".hydra").mkdir(exist_ok=True)
(bundle / ".hydra/config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
print(bundle)

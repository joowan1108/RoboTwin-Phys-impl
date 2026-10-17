#!/usr/bin/env python3
"""Create a deployment bundle with local processor/tokenizer paths.

The gated downloaded originals and their checksums are preserved. Requires G05
access approval and a completed `download_models.py --models g05` first.

The bundle files are HARD links (or copies), never symlinks: the pinned
GalaxeaVLA adapter calls Path(ckpt).resolve() and reads `.hydra/config.yaml`
next to the resolved file, so a symlink would make it read the original
download config, whose relative `checkpoints/...` paths only exist in the
upstream repository layout (job 1690: "HF processor tokenizer.json not found").
"""
from pathlib import Path
import os
import shutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "checkpoints/g05"
bundle = ROOT / "checkpoints/g05-runtime"
original = source / "g05-robotwin20/.hydra/config.yaml"
if not original.exists():
    raise SystemExit("G05 config missing. Obtain HF access and download the pinned RoboTwin checkpoint first.")
config = yaml.safe_load(original.read_text())
processor = source / "qwen3_5_2b_base_processor"
tokenizer = source / "action_tokenizer.pt"
for required in (processor / "tokenizer.json", tokenizer):
    if not required.is_file():
        raise FileNotFoundError(required)
config["model"]["model_arch"]["hf_processor_path"] = str(processor)
config["model"]["processor"]["tokenizer_params"]["pretrained_model_name_or_path"] = str(processor)
config["tokenizer"]["vq_config"]["ckpt_dir"] = str(tokenizer)
if isinstance(config["model"].get("tokenizer"), dict):
    config["model"]["tokenizer"]["vq_config"]["ckpt_dir"] = str(tokenizer)
for relative in ("checkpoints/model_state_dict.pt", "dataset_stats.json"):
    target = bundle / relative
    origin = source / "g05-robotwin20" / relative
    if not origin.exists():
        raise FileNotFoundError(origin)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        # Earlier bundles linked symbolically; the adapter resolves that link away.
        if target.resolve() != origin.resolve():
            raise RuntimeError(f"Not replacing existing unrelated link: {target}")
        target.unlink()
    if target.exists():
        same = os.path.samefile(target, origin) or target.stat().st_size == origin.stat().st_size  # hard link or earlier copy
        if not same:
            raise RuntimeError(f"Not replacing existing unrelated file: {target}")
    else:
        try:
            os.link(origin, target)
        except OSError:
            shutil.copy2(origin, target)
    if target.is_symlink() or not target.resolve().is_relative_to(bundle.resolve()):
        raise RuntimeError(f"Bundle file must resolve inside {bundle}: {target}")
(bundle / ".hydra").mkdir(exist_ok=True)
(bundle / ".hydra/config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
print(f"{bundle}: checkpoint {'hard-linked' if os.path.samefile(bundle / 'checkpoints/model_state_dict.pt', source / 'g05-robotwin20/checkpoints/model_state_dict.pt') else 'copied'}; "
      f"processor={processor}; action tokenizer={tokenizer}")

#!/usr/bin/env python3
"""Download only the requested RoboTwin evaluation weights/stats, at pinned revisions.

G05 requires the user's Hugging Face access approval/token. No license is accepted
automatically. This command never downloads optimizer state or LIBERO checkpoints.
"""
import argparse
import hashlib
import json
from pathlib import Path
import fnmatch

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "g05": ("OpenGalaxea/G05", ["g05-robotwin20/*", "action_tokenizer.pt", "qwen3_5_2b_base_processor/*"]),
    "pi05": ("motus-robotics/pi0.5_robotwin2", ["model.safetensors", "assets/*", "README.md"]),
    "fact": ("Bariona/fact-wam", ["transformer/*", "norm_stats_delta.json", "README.md"]),
    "motus": ("motus-robotics/Motus_robotwin2", ["mp_rank_00_model_states.pt", "config.json", "README.md"]),
    "fastwam": ("yuanty/fastwam", ["robotwin_uncond_3cam_384.pt", "robotwin_uncond_3cam_384_dataset_stats.json", "README.md"]),
}


def main():
    from huggingface_hub import hf_hub_download
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", choices=SPECS, default=["pi05", "fact", "motus", "fastwam"])
    args = p.parse_args()
    for name in args.models:
        repo, patterns = SPECS[name]
        info = json.loads((ROOT / f"references/{repo.replace('/', '--')}-info.json").read_text())
        files = [f for f in info["siblings"] if any(fnmatch.fnmatch(f["rfilename"], pat) for pat in patterns)]
        records = []
        for f in files:
            filename = f["rfilename"]
            print(f"Downloading {repo}@{info['sha']}: {filename}", flush=True)
            path = Path(hf_hub_download(repo, filename, revision=info["sha"], local_dir=ROOT / "checkpoints" / name))
            checksum = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    checksum.update(block)
            expected = f.get("lfs", {}).get("sha256")
            if expected and checksum.hexdigest() != expected:
                raise RuntimeError(f"Checkpoint SHA256 mismatch: {path}")
            records.append({"file": filename, "size": path.stat().st_size, "sha256": checksum.hexdigest()})
        lock = ROOT / "checkpoints" / name / "download-lock.json"
        lock.write_text(json.dumps({"repo": repo, "revision": info["sha"], "files": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()

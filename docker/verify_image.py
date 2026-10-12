"""Validate versions and link rebuilt CUDA extensions without requiring a GPU."""
import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "RoboTwin": "13c3c47ff4312dd62484bcd51be034af55c062d1",
    "curobo": "d64c4b005459db10c5dd867d8b30a87d5bda9bdb",
    "GalaxeaVLA": "89f2322b4ad016e192437adc1a2c253b05bab246",
    "FastWAM": "7faa71108368fbb3b6885649f112af607427a2d4",
    "FACT": "5b299a2059f00c043fedc5c9dd17bf43fa14cbec",
    "Motus": "f771216802b8a1601599422f12088bee3c068c14",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--requirements", default="docker/requirements.lock")
    parser.add_argument("--cuda", default="12.1")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Simulator requires Python 3.10")
    versions = {}
    for line in (ROOT / args.requirements).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==")
        versions[name] = metadata.version(name)
        if versions[name] != expected:
            raise RuntimeError(f"{name}: {versions[name]} != {expected}")
    for repo, pin in PINS.items():
        # Repositories in the image are root-owned, while the runtime user is not.
        path = ROOT / "vendor" / repo
        revision = subprocess.check_output(["git", "-c", f"safe.directory={path}",
            "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        if revision != pin:
            raise RuntimeError(f"{repo} revision changed: {revision}")
    import torch
    if torch.version.cuda != args.cuda:
        raise RuntimeError(f"Unexpected torch CUDA: {torch.version.cuda}")
    for extension in ("lbfgs_step_cu", "kinematics_fused_cu", "line_search_cu", "tensor_step_cu", "geom_cu"):
        importlib.import_module(f"curobo.curobolib.{extension}")
    if metadata.version("nvidia-curobo") != "0.0.0":
        raise RuntimeError("CuRobo package metadata differs from original manifest; source revision is checked separately")
    result = {"python": sys.version, "versions": versions, "source_revisions": PINS,
        "cuda_extension_imports": "passed", "gpu_execution_tested": False,
        "requirements_sha256": hashlib.sha256((ROOT / args.requirements).read_bytes()).hexdigest()}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Python/package pins, source revisions and CuRobo extension imports: passed (no GPU execution)")


if __name__ == "__main__":
    main()

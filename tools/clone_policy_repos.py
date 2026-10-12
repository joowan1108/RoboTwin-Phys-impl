#!/usr/bin/env python3
"""Fetch the reviewed official policy code without installing into simulator env."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOS = {
    "GalaxeaVLA": ("https://github.com/OpenGalaxea/GalaxeaVLA", "89f2322b4ad016e192437adc1a2c253b05bab246"),
    "Motus": ("https://github.com/thu-ml/Motus", "f771216802b8a1601599422f12088bee3c068c14"),
    "FACT": ("https://github.com/Bariona/FACT", "5b299a2059f00c043fedc5c9dd17bf43fa14cbec"),
    "FastWAM": ("https://github.com/yuantianyuan01/FastWAM", "7faa71108368fbb3b6885649f112af607427a2d4"),
}
for name, (url, revision) in REPOS.items():
    path = ROOT / "vendor" / name
    if not path.exists():
        subprocess.run(["git", "clone", "--no-checkout", url, str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", revision], check=True)
    sha = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if sha != revision:
        raise RuntimeError(f"Existing {name} is at {sha}, expected {revision}; not overwriting")
    print(name, sha)

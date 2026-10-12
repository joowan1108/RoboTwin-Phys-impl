"""Verify the container, preserving host reports and the frozen episode manifest."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="manifests/verified-smoke-stack-blocks.json")
    args = parser.parse_args()
    if not Path("/.dockerenv").exists():
        raise RuntimeError("Run this validation inside the Docker image")
    if os.getuid() == 0:
        raise RuntimeError("Validation must run as the unprivileged runtime user")
    folder = ROOT / "reports/docker-validation" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    folder.mkdir(parents=True)
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    commands = [
        [sys.executable, "docker/verify_image.py", "--output", str(folder / "environment.json")],
        [sys.executable, "tools/doctor.py", "--output", str(folder / "hardware.json")],
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        [sys.executable, "-m", "robotwin_phys.cli", "replay", "--manifest", str(manifest),
         "--output", str(ROOT / "runs/docker-replay" / folder.name), "--video"],
    ]
    summary = {"status": "running", "manifest": str(manifest), "uid": os.getuid(),
               "gpu_uuid": os.environ.get("ROBOTWIN_PHYSICAL_GPU_UUID"), "checks": []}
    try:
        import torch
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Each simulator container must see exactly one CUDA GPU")
        for i, command in enumerate(commands):
            print("Running:", " ".join(command), flush=True)
            with (folder / f"check-{i}.log").open("w") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            summary["checks"].append({"command": command, "exit_code": result.returncode})
            if result.returncode:
                raise RuntimeError(f"Validation failed; see {folder / f'check-{i}.log'}")
        summary["status"] = "passed"
    except Exception:
        summary["status"] = "failed"
        raise
    finally:
        (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Container validation report: {folder}", flush=True)


if __name__ == "__main__":
    main()

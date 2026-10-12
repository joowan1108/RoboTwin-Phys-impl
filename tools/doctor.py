#!/usr/bin/env python3
"""Write machine/runtime evidence and run a real Vulkan render + CUDA calculation."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def command(args):
    p = subprocess.run(args, text=True, capture_output=True)
    return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}


def main():
    import numpy as np
    import torch
    import sapien
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=str(ROOT / "reports/hardware.json"))
    args = p.parse_args()
    info = {"python": sys.version, "platform": platform.platform(),
        "nvidia_smi": command(["nvidia-smi"]), "nvcc": command(["nvcc", "--version"]),
        "display": os.environ.get("DISPLAY"), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packages": {name: importlib.metadata.version(name) for name in
            ("torch", "torchvision", "numpy", "scipy", "sapien", "mplib", "warp-lang", "nvidia-curobo")},
        "pip_check": command([sys.executable, "-m", "pip", "check"])}
    assert torch.cuda.is_available()
    gpu = torch.cuda.get_device_properties(0)
    info["cuda"] = {"runtime": torch.version.cuda, "gpu": gpu.name, "capability": [gpu.major, gpu.minor],
                    "memory_gib": gpu.total_memory/2**30, "compiled_arches": torch.cuda.get_arch_list(),
                    "native_bf16_tensor_cores": gpu.major >= 8,
                    "torch_is_bf16_supported": torch.cuda.is_bf16_supported()}
    a = torch.eye(16, device="cuda")
    assert torch.equal(a @ a, a)
    info["cuda_matmul_pass"] = True
    info["vulkan_devices"] = sapien.render.get_device_summary()
    rs = sapien.render.RenderSystem(os.environ.get("SAPIEN_RENDER_DEVICE", "cuda:0"))
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), rs])
    scene.add_ground(0)
    scene.set_ambient_light([.5,.5,.5])
    camera = scene.add_camera("doctor", 64, 64, 1., .01, 10.)
    scene.update_render(); camera.take_picture()
    image = camera.get_picture("Color")
    assert image.shape == (64, 64, 4) and np.isfinite(image).all()
    info["vulkan_render_pass"] = True
    info["render_device"] = str(rs.device)
    info["assets_present"] = {name: (ROOT / "vendor/RoboTwin/assets" / name).exists()
                                for name in ("objects", "embodiments", "background_texture")}
    Path(args.output).write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: info[k] for k in ("cuda", "cuda_matmul_pass", "vulkan_render_pass", "render_device", "pip_check")}, indent=2))


if __name__ == "__main__":
    main()

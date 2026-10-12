"""Generate per-container driver bindings without installing host packages.

Fallback for this Ubuntu x86_64 server when NVIDIA Container Toolkit is absent.
Uses ordinary Docker read-only bind mounts and device permissions, not privileged
containers, host-wide mounts, daemon configuration, or copied driver libraries.
"""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import tempfile
import os

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output", type=Path, default=ROOT / ".container/compose.gpu-bind.json")
    args = parser.parse_args()
    if platform.machine() != "x86_64":
        raise RuntimeError("The driver-bind fallback supports Linux x86_64 only; use NVIDIA Container Toolkit")
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,driver_version",
        "--format=csv,noheader"], text=True)
    matches = [row.strip().split(", ") for row in rows.splitlines()
               if args.gpu in row.strip().split(", ")[:2]]
    if len(matches) != 1:
        raise ValueError(f"Select one physical GPU index or UUID, got {args.gpu!r}")
    index, uuid, driver = matches[0]
    devices = [f"/dev/nvidia{index}", "/dev/nvidiactl", "/dev/nvidia-uvm"]
    for optional in ("/dev/nvidia-uvm-tools", "/dev/nvidia-modeset"):
        if Path(optional).exists():
            devices.append(optional)
    for device in devices:
        if not Path(device).is_char_device():
            raise RuntimeError(f"GPU device missing: {device}")
    libraries = Path("/usr/lib/x86_64-linux-gnu")
    mounts = []

    def mount(source, target):
        source = Path(source).resolve(strict=True)
        mounts.append({"type": "bind", "source": str(source), "target": str(target),
            "read_only": True, "bind": {"create_host_path": False}})

    for pattern in ("libcuda.so*", "libnvidia*.so*", "libEGL_nvidia.so*", "libGLX_nvidia.so*", "libGLES*_nvidia.so*"):
        for library in sorted(libraries.glob(pattern)):
            mount(library, Path("/usr/local/nvidia/lib64") / library.name)
    required = {"libcuda.so.1", "libnvidia-ml.so.1", "libGLX_nvidia.so.0", "libEGL_nvidia.so.0"}
    if not required.issubset({Path(item["target"]).name for item in mounts}):
        raise RuntimeError("Required CUDA/NVML/Vulkan/EGL host driver libraries are missing")
    mount("/usr/bin/nvidia-smi", "/usr/bin/nvidia-smi")
    for descriptor in ("/usr/share/vulkan/icd.d/nvidia_icd.json", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"):
        mount(descriptor, descriptor)
    for extra in ("/usr/share/nvidia/nvoptix.bin",):
        if Path(extra).exists():
            mount(extra, extra)
    data = {"services": {"sim": {
        "devices": [f"{device}:{device}:rw" for device in devices],
        "environment": {
            "NVIDIA_VISIBLE_DEVICES": "void",
            "CUDA_VISIBLE_DEVICES": uuid,
            "ROBOTWIN_PHYSICAL_GPU_UUID": uuid,
            "ROBOTWIN_HOST_DRIVER_VERSION": driver,
            "LD_LIBRARY_PATH": "/usr/local/nvidia/lib64:/usr/local/cuda/lib64",
            "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
        },
        "volumes": mounts,
    }}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as output:
        output.write(json.dumps(data, indent=2) + "\n")
        temporary = output.name
    os.replace(temporary, args.output)
    print(f"GPU {index} ({uuid}), driver {driver}: per-container device/read-only driver bindings", flush=True)


if __name__ == "__main__":
    main()

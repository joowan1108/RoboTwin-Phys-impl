import hashlib
import importlib.metadata
import subprocess
import json
from .conditions import ROOT, digest


def asset_fingerprint():
    # Cache content hashes by file stat; changed files are rehashed. This catches
    # ordinary asset edits without re-reading 16GB for every task evaluation.
    cache_path = ROOT / "reports/asset-hash-cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    updated, hashes = {}, {}
    for path in sorted((ROOT / "vendor/RoboTwin/assets").rglob("*")):
        if not path.is_file() or ".cache" in path.parts or "__pycache__" in path.parts:
            continue
        key = str(path.relative_to(ROOT))
        stat = path.stat()
        stamp = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        entry = cache.get(key, {})
        if entry.get("stat") == stamp:
            checksum = entry["sha256"]
        else:
            h = hashlib.sha256()
            with path.open("rb") as file:
                for block in iter(lambda: file.read(8*1024*1024), b""):
                    h.update(block)
            checksum = h.hexdigest()
        updated[key] = {"stat": stamp, "sha256": checksum}
        hashes[key] = checksum
    from .manifest import atomic_json
    atomic_json(cache_path, updated)
    return {"sha256": digest(hashes), "file_count": len(hashes)}


def fingerprint():
    """Fail evaluation if source/config/runtime changed after expert filtering."""
    files = {}
    repo = ROOT / "vendor/RoboTwin"
    # Hash source and all benchmark config; model adapter revisions are separate.
    for base, patterns in ((ROOT / "src/robotwin_phys", ("*.py",)),
                           (ROOT / "configs", ("physical_domains.json",)),
                           (repo / "envs", ("*.py",)),
                           (repo / "task_config", ("*.yml",)),
                           (repo / "assets/embodiments/aloha-agilex", ("*.yml", "*.urdf", "*.srdf")),
                           (repo / "description", ("*.json", "*.py"))):
        for pattern in patterns:
            for path in sorted(base.rglob(pattern)):
                files[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    versions = {}
    for package in ("numpy", "torch", "sapien", "mplib", "scipy", "warp-lang", "nvidia-curobo"):
        versions[package] = importlib.metadata.version(package)
    import torch
    gpu = {"name": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
           "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).splitlines()[0],
           "torch_cuda": torch.version.cuda}
    return {"source_sha256": digest(files), "files": files, "versions": versions, "gpu": gpu,
            "robotwin_revision": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
            "assets": asset_fingerprint()}

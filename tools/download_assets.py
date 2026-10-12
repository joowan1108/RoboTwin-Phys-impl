#!/usr/bin/env python3
"""Resume and checksum official assets against the captured HF LFS hashes."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    downloads = ROOT / "downloads"
    downloads.mkdir(exist_ok=True)
    records = []
    for entry in json.loads((ROOT / "references/assets-tree.json").read_text()):
        name = entry["path"]
        if name not in ("embodiments.zip", "objects.zip", "background_texture.zip"):
            continue
        path = downloads / name
        if not path.exists() or path.stat().st_size != entry["size"]:
            subprocess.run(["curl", "--fail", "--location", "--retry", "4", "--continue-at", "-",
                f"https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/main/{name}", "-o", str(path)], check=True)
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b""):
                checksum.update(block)
        if checksum.hexdigest() != entry["lfs"]["oid"]:
            raise RuntimeError(f"Official asset SHA256 mismatch: {path}; do not use changed assets with old manifests")
        destination = ROOT / "vendor/RoboTwin/assets"
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise ValueError("Invalid archive path")
            # A directory can exist after an interrupted unzip. Fill missing or
            # truncated members instead of assuming the whole archive exists.
            for member in archive.infolist():
                target = destination / member.filename
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                elif not target.exists() or target.stat().st_size != member.file_size:
                    archive.extract(member, destination)
        records.append({"file": name, "sha256": checksum.hexdigest(), "size": path.stat().st_size})
        print(f"Verified {name}", flush=True)
    (ROOT / "reports/assets-verified.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()

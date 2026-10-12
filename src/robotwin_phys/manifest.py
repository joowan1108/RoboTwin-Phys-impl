"""Content-addressed episode pools, frozen before policy evaluation."""
import json
import os
from pathlib import Path
from .conditions import digest


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def seal(manifest):
    body = {k: v for k, v in manifest.items() if k != "sha256"}
    return {**body, "sha256": digest(body)}


def load(path, require_verified=True):
    data = json.loads(Path(path).read_text())
    if data.get("sha256") != seal(data)["sha256"]:
        raise ValueError("Manifest hash mismatch; conditions were modified")
    if data.get("schema") != "robotwin-phys-local-v1":
        raise ValueError("Unsupported manifest schema")
    eps = data["episodes"]
    if not eps or len({e["seed"] for e in eps}) != len(eps):
        raise ValueError("Empty pool or duplicate episode seeds")
    for i, e in enumerate(eps):
        if e["episode_index"] != i or e["seed"] != e["condition"]["episode_seed"]:
            raise ValueError("Inconsistent episode identity")
        if e["condition"]["task"] != data["task"] or e["condition"]["protocol"] != data["protocol"]:
            raise ValueError("Inconsistent task or protocol")
        if e["condition_sha256"] != digest(e["condition"]):
            raise ValueError("Physical condition hash mismatch")
        if require_verified and (e.get("expert_success") is not True or not e.get("initial_state") or not e.get("instruction")):
            raise ValueError("Evaluation requires a completed expert-verified manifest")
    if require_verified and data.get("status") != "verified":
        raise ValueError("Candidate manifests cannot be used for scored evaluation")
    return data

"""Task domains and independent, stable streams. All distances use meters."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DOMAIN_FILE = ROOT / "configs/physical_domains.json"
PROTOCOLS = ("clean", "official_random", "physical_random", "physical_only")
FACTORS = ("mass", "com", "geometry", "table_tilt", "table_height", "force",
           "camera", "friction", "restitution", "object_damping", "robot_damping")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def stable_seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).digest()[:4], "big")


def domains(task: str) -> dict:
    data = json.loads(DOMAIN_FILE.read_text())
    return {**data["global"], **data["task_overrides"].get(task, {})}


def sample(task: str, seed: int, protocol="physical_random", only=(), without=()) -> dict:
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol: {protocol}")
    if not 0 <= seed < 2**32:
        raise ValueError("episode seed must be in [0, 2**32)")
    if set(only + without) - set(FACTORS) or (only and without):
        raise ValueError(f"Use only OR without with factors in {FACTORS}")
    if (only or without) and protocol not in ("physical_random", "physical_only"):
        raise ValueError("Ablations require a physical protocol")
    active = set(only or FACTORS) - set(without)
    if protocol in ("clean", "official_random"):
        active = set()
    bounds = domains(task)

    def rng(key):
        return np.random.Generator(np.random.PCG64(stable_seed("physics-v1", task, seed, key)))

    def val(key, factor, nominal):
        return float(rng(key).uniform(*bounds[key])) if factor in active else nominal

    mode = str(rng("camera_mode").choice(["orbit", "translation", "rotation"])) if "camera" in active else "none"
    direction = rng("camera_direction").normal(size=3)
    direction /= np.linalg.norm(direction)
    # Arm and gripper deliberately share this one factor (Appendix A).
    damping = val("robot_damping_scale", "robot_damping", 1.0)
    return {
        "task": task, "episode_seed": seed, "protocol": protocol,
        "active_factors": sorted(active), "domains": bounds,
        "mass_scale": val("mass_scale", "mass", 1.0),
        "com_offset_m": rng("com").uniform(*bounds["com_offset_m"], size=3).tolist() if "com" in active else [0., 0., 0.],
        "geometry_scale": val("geometry_scale", "geometry", 1.0),
        "table_tilt_deg": val("table_tilt_deg", "table_tilt", 0.0),
        "table_tilt_azimuth_deg": 90.0,
        "table_height_offset_m": val("table_height_offset_m", "table_height", None),
        "force_mg": val("force_mg", "force", 0.0), "force_direction": [1., 0., 0.],
        "camera": {"mode": mode, "direction": direction.tolist(),
                   "orbit_m": val("camera_orbit_m", "camera", 0.0),
                   "translation_m": val("camera_translation_m", "camera", 0.0),
                   "rotation_deg": val("camera_rotation_deg", "camera", 0.0)},
        # None means preserve each original material/joint value, not substitute a guessed nominal value.
        "friction": val("friction", "friction", None),
        "restitution": val("restitution", "restitution", None),
        "object_joint_damping": val("object_joint_damping", "object_damping", None),
        "arm_damping_scale": damping, "gripper_damping_scale": damping,
    }


def task_names() -> list[str]:
    import yaml
    return sorted(yaml.safe_load((ROOT / "vendor/RoboTwin/task_config/_eval_step_limit.yml").read_text()))

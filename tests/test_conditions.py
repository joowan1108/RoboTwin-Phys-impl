import copy
import json
import random
import subprocess
import sys

import numpy as np
import pytest

from robotwin_phys.conditions import sample, domains, digest, FACTORS, task_names
from robotwin_phys.manifest import load, seal, atomic_json
from robotwin_phys.rng import EnvironmentRNG
from robotwin_phys.simulator import assert_same_state


def test_cross_process_and_policy_rng_independence():
    expected = sample("stack_blocks_two", 100000)
    code = "import json; from robotwin_phys.conditions import sample; print(json.dumps(sample('stack_blocks_two',100000)))"
    assert json.loads(subprocess.check_output([sys.executable, "-c", code])) == expected
    np.random.seed(819); np.random.random(3000)
    random.seed(12); [random.random() for _ in range(100)]
    assert sample("stack_blocks_two", 100000) == expected


@pytest.mark.parametrize("task,key,wanted", [
    ("dump_bin_bigbin", "com_offset_m", [-.001, .001]),
    ("grab_roller", "com_offset_m", [-.002, .002]),
    ("beat_block_hammer", "com_offset_m", [-.002, .002]),
    ("put_bottles_dustbin", "com_offset_m", [-.002, .002]),
    ("open_microwave", "object_joint_damping", [0., .25]),
    ("turn_switch", "object_joint_damping", [0., 1.]),
    ("put_object_cabinet", "geometry_scale", [1., 1.]),
    ("put_object_cabinet", "robot_damping_scale", [1., 1.]),
    ("scan_object", "camera_orbit_m", [0., .05]),
    ("place_bread_skillet", "table_tilt_deg", [0., 1.]),
    ("place_bread_skillet", "geometry_scale", [.9, 1.05]),
])
def test_paper_task_domains(task, key, wanted):
    assert domains(task)[key] == wanted


def test_all_50_tasks_and_100_episodes_within_bounds():
    assert len(task_names()) == 50
    for task in task_names():
        for seed in range(100000, 100100):
            c = sample(task, seed)
            for key, bounds in domains(task).items():
                if key.startswith("camera_"):
                    v = c["camera"][key.removeprefix("camera_")]
                elif key == "robot_damping_scale":
                    v = c["arm_damping_scale"]
                else:
                    v = c[key]
                assert np.all(np.asarray(v) >= bounds[0]) and np.all(np.asarray(v) <= bounds[1])
            assert c["arm_damping_scale"] == c["gripper_damping_scale"]


def test_ablation_keeps_unrelated_draws_and_nominal_values():
    base = sample("grab_roller", 42)
    cut = sample("grab_roller", 42, without=("mass",))
    assert cut["mass_scale"] == 1
    for key in base.keys() - {"active_factors", "mass_scale"}:
        assert base[key] == cut[key]
    for protocol in ("clean", "official_random"):
        c = sample("grab_roller", 42, protocol)
        assert c["active_factors"] == []
        assert c["friction"] is None and c["object_joint_damping"] is None


def test_rng_restores_caller_and_isolates_nested_environment_calls():
    random.seed(31); np.random.seed(31)
    expected = (random.random(), np.random.rand())
    random.seed(31); np.random.seed(31)
    env = EnvironmentRNG(57)
    with env.use():
        first = np.random.rand()
        with env.use():
            second = np.random.rand()
    assert (random.random(), np.random.rand()) == expected
    other = EnvironmentRNG(57)
    with other.use():
        assert (np.random.rand(), np.random.rand()) == (first, second)


def test_manifest_rejects_tampering_and_unverified_pool(tmp_path):
    c = sample("grab_roller", 42)
    data = {"schema": "robotwin-phys-local-v1", "status": "candidates", "task": "grab_roller", "protocol": "physical_random",
            "episodes": [{"episode_index": 0, "seed": 42, "condition": c, "condition_sha256": digest(c), "expert_success": False}]}
    path = tmp_path / "pool.json"
    atomic_json(path, seal(data))
    assert load(path, require_verified=False)["episodes"][0]["seed"] == 42
    with pytest.raises(ValueError, match="expert-verified"):
        load(path)
    edited = json.loads(path.read_text()); edited["episodes"][0]["seed"] = 43
    atomic_json(path, edited)
    with pytest.raises(ValueError, match="hash mismatch"):
        load(path)


def test_initial_state_detects_scene_change():
    assert_same_state({"p": [0., 1.]}, {"p": [1e-7, 1.]})
    with pytest.raises(ValueError, match="differs"):
        assert_same_state({"p": [0., 1.]}, {"p": [.01, 1.]})

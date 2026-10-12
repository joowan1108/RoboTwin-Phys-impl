#!/usr/bin/env python3
"""Exercise the nine Appendix B configurations in real SAPIEN (not scored eval)."""
import json
from pathlib import Path
from robotwin_phys.conditions import ROOT, sample, domains
from robotwin_phys.cli import worker
from robotwin_phys.manifest import atomic_json

folder = ROOT / "reports/sensitive-tasks"
tasks = json.loads((ROOT / "configs/physical_domains.json").read_text())["task_overrides"]
results = {}
for task in tasks:
    for seed in range(100000,100005):
        c = sample(task, seed, "physical_only")
        result = worker({"mode":"inspect","episode":{"condition":c,"seed":seed}}, folder, f"{task}-{seed}", 300)
        if result["status"] == "rejected":
            continue
        state = result["initial_state"]
        b = domains(task)
        assert all(body["mass"] == 1.0 for body in state["physical"]["robot_bodies"])
        object_names = {body["name"] for body in state["physical"]["bodies"]}
        assert not object_names.intersection(body["name"] for body in state["physical"]["robot_bodies"])
        for joint in state["physical"]["object_joints"]:
            assert b["object_joint_damping"][0] <= joint["damping"] <= b["object_joint_damping"][1]
        assert result["extra_observations_preserve_lights"]
        results[task] = {"status":"passed", "seed":seed,"object_bodies":len(state["physical"]["bodies"]),
                         "object_joints":len(state["physical"]["object_joints"]),"render_device":result["render_device"]}
        break
    else:
        results[task] = {"status":"no_stable_candidate_in_smoke_budget"}
    atomic_json(ROOT / "reports/sensitive-tasks-summary.json", results)
    print(task, results[task], flush=True)
if any(x["status"] != "passed" for x in results.values()):
    raise SystemExit(1)

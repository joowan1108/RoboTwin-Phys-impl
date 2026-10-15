"""check_success attributes that only the expert assigns are derived from the initial scene."""
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from robotwin_phys import success_context


class ArmTag:
    def __init__(self, value):
        self.arm = value

    @property
    def opposite(self):
        return ArmTag("left" if self.arm == "right" else "right")

    def __eq__(self, other):
        return self.arm == (other.arm if isinstance(other, ArmTag) else other)

    def __str__(self):
        return self.arm


@pytest.fixture
def fake_envs(monkeypatch):
    utils = types.ModuleType("envs.utils")
    utils.ArmTag = ArmTag
    utils.get_face_prod = lambda q, local_axis, target_axis: float(np.sign(q[0]))  # stand-in: sign of the first quaternion term
    envs = types.ModuleType("envs")
    envs.utils = utils
    monkeypatch.setitem(sys.modules, "envs", envs)
    monkeypatch.setitem(sys.modules, "envs.utils", utils)
    return utils


def pose(p, q=(1.0, 0.0, 0.0, 0.0)):
    return SimpleNamespace(p=np.asarray(p, dtype=float), q=np.asarray(q, dtype=float))


@pytest.mark.parametrize("x,arm", [(0.25, "right"), (-0.25, "left")])
def test_cabinet_and_scale_take_the_arm_from_the_object_side(fake_envs, x, arm):
    for task in ("put_object_cabinet", "place_object_scale"):
        env = SimpleNamespace(object=SimpleNamespace(get_pose=lambda: pose([x, -0.1, 0.69])))
        context = success_context.prepare(env, task, {"info": {"{a}": arm}})
        assert env.arm_tag == arm and context["arm_tag"] == arm
        if task == "put_object_cabinet":
            assert env.origin_z == pytest.approx(0.69) and context["origin_z"] == pytest.approx(0.69)
        else:
            assert not hasattr(env, "origin_z") and "origin_z" not in context


@pytest.mark.parametrize("q0,arm", [(0.9, "left"), (-0.9, "right")])
def test_open_laptop_uses_the_laptop_facing_direction(fake_envs, q0, arm):
    env = SimpleNamespace(laptop=SimpleNamespace(get_pose=lambda: pose([0.0, 0.0, 0.8], [q0, 0.0, 0.0, 0.1])))
    context = success_context.prepare(env, "open_laptop")
    assert env.arm_tag == arm and context == {"arm_tag": arm}


def test_mismatch_with_the_expert_record_is_an_error_not_a_silent_fix(fake_envs):
    env = SimpleNamespace(object=SimpleNamespace(get_pose=lambda: pose([0.25, 0.0, 0.7])))
    with pytest.raises(ValueError, match="expert used left"):
        success_context.prepare(env, "put_object_cabinet", {"info": {"{a}": "left"}})


def test_other_tasks_are_untouched_and_need_no_official_imports(monkeypatch):
    monkeypatch.delitem(sys.modules, "envs.utils", raising=False)
    env = SimpleNamespace()
    assert success_context.prepare(env, "adjust_bottle", {"info": {"{a}": "left"}}) == {}
    assert not hasattr(env, "arm_tag")


def test_affected_task_list_matches_the_official_scan():
    assert success_context.TASKS == ("open_laptop", "place_object_scale", "put_object_cabinet")


def test_scan_of_pinned_official_tasks_finds_exactly_these_three():
    """Attributes read in check_success but assigned only in play_once, from the vendored sources."""
    import ast
    from pathlib import Path
    envs = Path(__file__).parents[1] / "vendor/RoboTwin/envs"
    if not (envs / "_base_task.py").exists():
        pytest.skip("vendored RoboTwin sources are not available")
    found = {}
    for path in sorted(envs.glob("*.py")):
        if path.name.startswith("_"):
            continue
        tree = ast.parse(path.read_text())
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        if not classes:
            continue
        methods = {m.name: m for m in classes[0].body if isinstance(m, ast.FunctionDef)}
        if "play_once" not in methods or "check_success" not in methods:
            continue

        def assigned(fn):
            return {t.attr for n in ast.walk(fn) if isinstance(n, ast.Assign) for t in n.targets
                    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self"}

        setup = set().union(*(assigned(methods[m]) for m in ("load_actors", "setup_demo") if m in methods))
        read = {n.attr for n in ast.walk(methods["check_success"]) if isinstance(n, ast.Attribute)
                and isinstance(n.value, ast.Name) and n.value.id == "self" and isinstance(n.ctx, ast.Load)}
        needed = sorted((assigned(methods["play_once"]) & read) - setup)
        if needed:
            found[path.stem] = needed
    assert found == {"open_laptop": ["arm_tag"], "place_object_scale": ["arm_tag"],
                     "put_object_cabinet": ["arm_tag", "origin_z"]}

"""Replay the official grasp methods and exercise worker error classification."""
import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from robotwin_phys.planning import ExpertPlanningFailure, PlanningMixin
from robotwin_phys import worker

spec = importlib.util.spec_from_file_location(
    "official_expert", Path(__file__).parent / "fixtures/robotwin_expert.py")
official = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)


class Actor:
    def __init__(self, contacts=2):
        self.contacts = contacts

    def get_name(self):
        return "001_bottle"

    def get_pose(self):
        return SimpleNamespace(p=[0.1, -0.1, 0.75], q=[1., 0., 0., 0.])

    def get_contact_point(self, index, ret="list"):
        matrix = np.eye(4)
        matrix[0, 3] = index * 0.01
        return matrix if ret == "matrix" else [*matrix[:3, 3], 1., 0., 0., 0.]

    def iter_contact_points(self):
        for i in range(self.contacts):
            yield i, self.get_contact_point(i)


def environment(cls, statuses):
    env = cls()
    env.plan_success = env.need_plan = True
    env.phase = "expert"
    calls = []
    results = iter(statuses)

    def plan(poses):
        status = next(results)
        calls.append(status)
        return {"status": [status], "position": [np.zeros((2, 7))]}

    env.robot = SimpleNamespace(
        get_grasp_perfect_direction=lambda arm: "side",
        create_target_pose_list=lambda pose, center, arm: [pose],
        left_plan_multi_path=plan, right_plan_multi_path=plan)
    return env, calls


class Adapted(PlanningMixin, official.OfficialBaseTask):
    pass


def test_original_implementation_reproduces_reported_assertion():
    env, calls = environment(official.OfficialBaseTask, ["Fail", "Fail"])
    with pytest.raises(AssertionError, match="target_pose cannot be None"):
        env.grasp_actor(Actor(), "left")
    assert calls == ["Fail", "Fail"]


def test_exhausted_valid_search_is_an_explicit_planning_failure():
    env, calls = environment(Adapted, ["Fail", "Fail"])
    with pytest.raises(ExpertPlanningFailure) as error:
        env.grasp_actor(Actor(), "left")
    assert calls == ["Fail", "Fail"] and env.plan_success is False
    evidence = error.value.diagnostics
    assert evidence["kind"] == "no_reachable_grasp"
    assert evidence["actor"] == "001_bottle" and evidence["contact_point_ids"] == [0, 1]
    json.dumps(evidence, allow_nan=False)


def test_failed_first_contact_does_not_prevent_success_at_next_contact():
    original, _ = environment(official.OfficialBaseTask, ["Fail", "Success"])
    adapted, calls = environment(Adapted, ["Fail", "Success"])
    original_arm, original_actions = original.grasp_actor(Actor(), "right")
    arm, actions = adapted.grasp_actor(Actor(), "right")
    assert calls == ["Fail", "Success"] and adapted.plan_success is True
    assert arm == original_arm
    assert [(a.action, a.target_pose) for a in actions] == [
        (a.action, a.target_pose) for a in original_actions]


@pytest.mark.parametrize("bad", [None, np.eye(3), np.full((4, 4), np.nan)])
def test_bad_contact_metadata_is_not_excluded_as_infeasible(bad):
    env, calls = environment(Adapted, ["Fail"])
    actor = Actor(1)
    actor.get_contact_point = lambda *args: bad
    with pytest.raises(ValueError, match="grasp contact point"):
        env.grasp_actor(actor, "left")
    assert calls == []


def test_empty_iterator_without_explicit_asset_metadata_remains_an_error():
    env, calls = environment(Adapted, [])
    with pytest.raises(ValueError, match="Missing grasp contact points"):
        env.grasp_actor(Actor(0), "left")
    assert calls == []


@pytest.mark.parametrize('variant', range(5))
def test_official_woodenblock_empty_annotations_are_expert_failures(variant):
    config = json.loads((Path(__file__).parent / f'fixtures/woodenblock/model_data{variant}.json').read_text())
    assert config['contact_points_pose'] == []
    actor = Actor(0)
    actor.config, actor.POINTS = config, {'contact': 'contact_points_pose'}
    actor.get_name = lambda: '086_woodenblock'
    original, _ = environment(official.OfficialBaseTask, [])
    with pytest.raises(AssertionError, match='target_pose cannot be None'):
        original.grasp_actor(actor, 'left')
    env, calls = environment(Adapted, [])
    with pytest.raises(ExpertPlanningFailure) as error:
        env.grasp_actor(actor, 'left')
    assert calls == [] and env.plan_success is False
    assert error.value.diagnostics['kind'] == 'no_grasp_annotations'


@pytest.mark.parametrize('config', [None, {}, {'contact_points_pose': None},
                                   {'contact_points_pose': {}}, {'contact_points_pose': ''}])
def test_empty_iterator_with_missing_or_invalid_metadata_is_not_rejected(config):
    actor = Actor(0)
    actor.config, actor.POINTS = config, {'contact': 'contact_points_pose'}
    env, _ = environment(Adapted, [])
    with pytest.raises(ValueError, match='Missing grasp contact points'):
        env.grasp_actor(actor, 'left')


def test_explicit_empty_contact_selection_remains_a_task_configuration_error():
    actor = Actor(0)
    actor.config, actor.POINTS = {'contact_points_pose': []}, {'contact': 'contact_points_pose'}
    env, _ = environment(Adapted, [])
    with pytest.raises(ValueError, match='Missing grasp contact points'):
        env.grasp_actor(actor, 'left', contact_point_id=[])


def test_empty_articulation_annotations_follow_same_rule():
    actor = Actor(0)
    actor.config, actor.POINTS = {'contact_points': []}, {'contact': 'contact_points'}
    env, _ = environment(Adapted, [])
    with pytest.raises(ExpertPlanningFailure) as error:
        env.grasp_actor(actor, 'left')
    assert error.value.diagnostics['kind'] == 'no_grasp_annotations'


class AnnotatedActor(Actor):
    """Use official metadata and Actor's out-of-range -> None contract."""
    POINTS = {'contact': 'contact_points_pose'}

    def __init__(self, asset, variant):
        self.config = json.loads((Path(__file__).parent /
            f'fixtures/grasp_contacts/{asset}/model_data{variant}.json').read_text())
        self.contacts = len(self.config[self.POINTS['contact']])
        self.asset = asset

    def get_name(self):
        return self.asset

    def get_contact_point(self, index, ret='list'):
        points = self.config[self.POINTS['contact']]
        try:
            points[index]
        except (IndexError, TypeError):
            return None
        return super().get_contact_point(index, ret)


@pytest.mark.parametrize('asset,variant', [('021_cup', i) for i in range(1, 8)] +
                         [('002_bowl', i) for i in (1, 2, 3, 5)])
@pytest.mark.parametrize('arm', ['left', 'right'])
def test_all_official_container_variants_with_both_task_contact_choices(asset, variant, arm):
    actor = AnnotatedActor(asset, variant)
    contact = 2 if arm == 'left' else 0
    original, original_calls = environment(official.OfficialBaseTask, ['Success'])
    adapted, calls = environment(Adapted, ['Success'])
    if asset == '021_cup' and variant in (6, 7) and arm == 'left':
        with pytest.raises(AssertionError, match='target_pose cannot be None'):
            original.grasp_actor(actor, arm, contact_point_id=contact)
        with pytest.raises(ExpertPlanningFailure) as error:
            adapted.grasp_actor(actor, arm, contact_point_id=contact)
        assert error.value.diagnostics['kind'] == 'unavailable_grasp_annotation'
        assert error.value.diagnostics['contact_point_ids'] == [2]
        assert error.value.diagnostics['available_contact_point_ids'] == [0, 1]
        assert error.value.diagnostics['arm'] == 'left'
        assert calls == original_calls == []
    else:
        old_arm, old_actions = original.grasp_actor(actor, arm, contact_point_id=contact)
        new_arm, new_actions = adapted.grasp_actor(actor, arm, contact_point_id=contact)
        assert old_arm == new_arm and original_calls == calls == ['Success']
        assert [(a.action, a.target_pose) for a in old_actions] == [
            (a.action, a.target_pose) for a in new_actions]


def test_worker_classifies_missing_selected_contact_and_retains_available_ids(monkeypatch):
    from robotwin_phys import simulator
    env, calls = environment(Adapted, [])
    env.initial_state, env.dispose = lambda: {}, lambda: None
    env.play_once = lambda: env.grasp_actor(AnnotatedActor('021_cup', 6), 'left', contact_point_id=2)
    monkeypatch.setattr(simulator, 'create_env', lambda *a, **kw: env)
    result = worker.run({'mode': 'expert', 'episode': {'condition': {}}})
    assert result['status'] == 'rejected' and result['reason'] == 'expert_failed'
    assert result['planning_failure']['kind'] == 'unavailable_grasp_annotation'
    assert result['planning_failure']['available_contact_point_ids'] == [0, 1]
    assert result['plan_success'] is False and calls == []


def test_direct_contact_query_uses_same_unavailable_annotation_classification():
    env, calls = environment(Adapted, [])
    with pytest.raises(ExpertPlanningFailure) as error:
        env.get_grasp_pose(AnnotatedActor('021_cup', 7), 'left', contact_point_id=np.int64(2))
    assert error.value.diagnostics['kind'] == 'unavailable_grasp_annotation' and calls == []
    json.dumps(error.value.diagnostics, allow_nan=False)


@pytest.mark.parametrize('points', [None, {}, [None], [np.eye(3).tolist()],
                                   [np.full((4, 4), np.nan).tolist()]])
def test_missing_selected_contact_does_not_hide_corrupt_metadata(points):
    actor = AnnotatedActor('021_cup', 6)
    actor.config['contact_points_pose'] = points
    actor.get_contact_point = lambda *a: None
    env, calls = environment(Adapted, [])
    with pytest.raises(ValueError):
        env.grasp_actor(actor, 'left', contact_point_id=2)
    assert calls == []


@pytest.mark.parametrize('index', [-3, 2.0, '2', None])
def test_invalid_index_types_do_not_become_expected_expert_failure(index):
    actor = AnnotatedActor('021_cup', 6)
    actor.get_contact_point = lambda *a: None
    env, calls = environment(Adapted, [])
    with pytest.raises(ValueError):
        env.get_grasp_pose(actor, 'left', contact_point_id=index)
    assert calls == []


def test_in_range_contact_returning_none_remains_an_execution_error():
    actor = AnnotatedActor('021_cup', 1)  # 2 is a valid annotated index.
    actor.get_contact_point = lambda *a: None
    env, _ = environment(Adapted, [])
    with pytest.raises(ValueError, match='Missing grasp contact point 2'):
        env.grasp_actor(actor, 'left', contact_point_id=2)


def test_selected_contact_does_not_validate_unselected_metadata():
    env, calls = environment(Adapted, ["Success"])
    actor = Actor()
    original = actor.get_contact_point
    actor.get_contact_point = lambda index, ret="list": original(index, ret) if index == 1 else None
    env.grasp_actor(actor, "left", contact_point_id=1)
    assert calls == ["Success"]


def test_direct_grasp_query_is_an_explicit_failure_instead_of_none():
    env, calls = environment(Adapted, ["Fail"])
    with pytest.raises(ExpertPlanningFailure) as error:
        env.get_grasp_pose(Actor(), "left", contact_point_id=1)
    assert error.value.diagnostics["contact_point_ids"] == [1]
    assert calls == ["Fail"]


def test_search_context_is_restored_after_unexpected_planner_error():
    env, _ = environment(Adapted, [])

    def fail(poses):
        raise RuntimeError("CUDA failure")

    env.robot.left_plan_multi_path = fail
    with pytest.raises(RuntimeError, match="CUDA failure"):
        env.grasp_actor(Actor(), "left")
    assert env._grasp_search_depth == 0


def test_previous_plan_failure_preserves_official_short_circuit():
    original, _ = environment(official.OfficialBaseTask, [])
    adapted, calls = environment(Adapted, [])
    original.plan_success = adapted.plan_success = False
    assert original.grasp_actor(Actor(), "left") == adapted.grasp_actor(Actor(), "left")
    assert calls == []


def placement_environment(monkeypatch):
    spec = importlib.util.spec_from_file_location('placement_math',
            Path(__file__).parent / 'fixtures/robotwin_transforms.py')
    maths = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(maths)
    from robotwin_phys import numerics
    for name in ('get_align_matrix', 'cal_quat_dis', 'get_product_vector', 'rotate_cone'):
        setattr(maths, name, getattr(numerics, name))
    monkeypatch.setitem(__import__('sys').modules, 'envs.utils.transforms', maths)
    monkeypatch.setattr(official, 'transforms', maths, raising=False)
    monkeypatch.setattr(official, 'get_place_pose', maths.get_place_pose, raising=False)
    env, _ = environment(Adapted, [])
    env.robot.get_left_ee_pose = env.robot.get_right_ee_pose = lambda: [0, -.3, .75, 1, 0, 0, 0]
    actor = Actor()
    actor.get_pose = lambda: maths.Pose([.1, -.1, .7])
    actor.get_functional_point = lambda index, ret: maths.Pose([.1, -.1, .7])
    return env, actor, maths


def test_original_place_actor_builds_finite_actions_with_shared_repair(monkeypatch):
    env, actor, maths = placement_environment(monkeypatch)
    arm, actions = env.place_actor(actor, 'left', maths.Pose([.2, .1, .8]), functional_point_id=0)
    assert str(arm) == 'left'
    assert env.plan_success is True
    assert all(np.isfinite(a.target_pose).all() for a in actions if a.action == 'move')


def test_zero_placement_projection_is_explicit_expert_failure(monkeypatch):
    env, actor, maths = placement_environment(monkeypatch)
    with pytest.raises(ExpertPlanningFailure) as error:
        env.get_place_pose(actor, 'left', maths.Pose([.2, .1, .8]), functional_point_id=0,
                           constrain='align', actor_axis=[0, 0, 1], actor_axis_type='world', align_axis=[1, 0, 0])
    assert error.value.diagnostics['kind'] == 'undefined_placement_direction'


def test_missing_functional_point_is_not_excluded_as_infeasible(monkeypatch):
    env, actor, maths = placement_environment(monkeypatch)
    actor.get_functional_point = lambda *args: None
    with pytest.raises(ValueError, match='Missing functional point'):
        env.place_actor(actor, 'left', maths.Pose(), functional_point_id=0)


def test_nonfinite_robot_pose_remains_an_execution_error(monkeypatch):
    env, actor, maths = placement_environment(monkeypatch)
    env.robot.get_left_ee_pose = lambda: [np.nan, 0, 0, 1, 0, 0, 0]
    with pytest.raises(ValueError, match='Invalid end effector pose'):
        env.place_actor(actor, 'left', maths.Pose(), functional_point_id=0)


@pytest.mark.parametrize("error", [FileNotFoundError("asset missing"),
                                  RuntimeError("CUDA out of memory"),
                                  AssertionError("unrelated assertion")])
def test_worker_preserves_unexpected_errors_and_disposes_environment(monkeypatch, error):
    from robotwin_phys import simulator
    disposed = []
    env = SimpleNamespace(initial_state=lambda: {}, dispose=lambda: disposed.append(True))

    def fail():
        raise error

    env.play_once = fail
    monkeypatch.setattr(simulator, "create_env", lambda *a, **kw: env)
    with pytest.raises(type(error), match=str(error)):
        worker.run({"mode": "expert", "episode": {"condition": {}}})
    assert disposed == [True]


def test_worker_rejects_known_planning_failure_with_diagnostics(monkeypatch):
    from robotwin_phys import simulator
    env, _ = environment(Adapted, ["Fail", "Fail"])
    disposed = []
    env.initial_state = lambda: {}
    env.dispose = lambda: disposed.append(True)
    env.play_once = lambda: env.grasp_actor(Actor(), "left")
    monkeypatch.setattr(simulator, "create_env", lambda *a, **kw: env)
    result = worker.run({"mode": "expert", "episode": {"condition": {}}})
    assert result["status"] == "rejected" and result["reason"] == "expert_failed"
    assert result["planning_failure"]["kind"] == "no_reachable_grasp"
    assert result["plan_success"] is False and disposed == [True]


def test_worker_error_result_retains_traceback(tmp_path, monkeypatch):
    request, result = tmp_path / "request.json", tmp_path / "result.json"
    request.write_text('{}')
    monkeypatch.setattr("sys.argv", ["worker", str(request), str(result)])

    def fail(request):
        raise FileNotFoundError("curobo_left.yml")

    monkeypatch.setattr(worker, "run", fail)
    with pytest.raises(FileNotFoundError):
        worker.main()
    saved = json.loads(result.read_text())
    assert saved["status"] == "error" and "FileNotFoundError" in saved["error"]


def test_worker_root_traceback_is_visible_in_task_error(tmp_path, monkeypatch):
    from robotwin_phys import cli

    def run(argv, stdout, **kwargs):
        stdout.write('Traceback: numerical root cause\nnumpy.linalg.LinAlgError: Eigenvalues did not converge\n')
        stdout.flush()
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(cli.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='Eigenvalues did not converge'):
        cli.worker({}, tmp_path, 'seed-100113', 30)


def test_freeze_continues_after_planning_failure_and_preserves_evidence(tmp_path, monkeypatch):
    from robotwin_phys import cli, provenance, simulator
    from robotwin_phys.manifest import load
    seeds, disposed = [], []

    def create(condition, **kwargs):
        seed = condition["episode_seed"]
        seeds.append(seed)
        env, _ = environment(Adapted, ["Fail", "Fail"] if seed == 100054 else ["Fail", "Success"])
        env.initial_state = lambda: {"entities": []}
        env.dispose = lambda: disposed.append(seed)
        env.check_success = lambda: True
        env.env_rng = SimpleNamespace(use=nullcontext)

        def play():
            env.grasp_actor(Actor(), "left")
            return {"info": {}}

        env.play_once = play
        return env

    instructions = ModuleType("generate_episode_instructions")
    instructions.generate_episode_descriptions = lambda *a: [{"unseen": ["Adjust the bottle."]}]
    monkeypatch.setitem(__import__("sys").modules, "generate_episode_instructions", instructions)
    monkeypatch.setattr(simulator, "create_env", create)
    monkeypatch.setattr(cli, "worker", lambda request, *a: worker.run(request))
    monkeypatch.setattr(cli, "task_names", lambda: ["adjust_bottle"])
    monkeypatch.setattr(provenance, "fingerprint", lambda: {"test": True})
    output = tmp_path / "adjust_bottle.json"
    args = SimpleNamespace(output=str(output), task="adjust_bottle", protocol="physical_random",
                           episodes=1, max_candidates=2, seed=100054, instruction_type="unseen",
                           only=None, without=None, resume=False, live=False, port=8765,
                           viewer=False, video=False, timeout=30)
    cli.freeze(args)
    manifest = load(output)
    assert seeds == disposed == [100054, 100055]
    assert manifest["episodes"][0]["seed"] == 100055
    assert manifest["rejected"][0]["seed"] == 100054
    assert manifest["rejected"][0]["planning_failure"]["kind"] == "no_reachable_grasp"


class BottleTask(official.OfficialBaseTask):
    """Pinned official put_bottles_dustbin.play_once, lines 89 and 96-98.

    A failed CuRobo path is recorded only as plan_success = False by the
    official move_to_pose helpers; the script then continues and indexes the
    empty action list returned by the next grasp_actor call.
    """

    def move(self, actions):
        if self.plan_success is False:
            return False
        arm_tag, steps = actions
        for step in steps:
            if step.action == "move" and self.robot.left_plan_path(step.target_pose)["status"] != "Success":
                self.plan_success = False
                return False
        return True

    def play_once(self):
        # Bottle on the left: the grasp search finds a candidate, the path fails.
        self.move(self.grasp_actor(Actor(), "left", pre_grasp_dis=0.1))
        # Bottle on the right: official lines 96-98.
        right_action = self.grasp_actor(Actor(), "right", pre_grasp_dis=0.1)
        right_action[1][0].target_pose[2] += 0.06
        right_action[1][1].target_pose[2] += 0.06
        return {"info": {}}


class AdaptedBottleTask(PlanningMixin, BottleTask):
    pass


def bottle_environment(cls, path_status):
    env, calls = environment(cls, ["Success", "Success", "Success", "Success"])
    env.robot.left_plan_path = env.robot.right_plan_path = lambda pose, **kw: {"status": path_status,
                                                                                 "position": [np.zeros(7)]}
    return env, calls


def test_official_task_crashes_after_silent_planner_failure():
    env, _ = bottle_environment(BottleTask, "Fail")
    with pytest.raises(IndexError, match="list index out of range"):
        env.play_once()
    assert env.plan_success is False


def test_crash_after_planner_failure_is_classified_as_expert_failure():
    env, _ = bottle_environment(AdaptedBottleTask, "Fail")
    with pytest.raises(ExpertPlanningFailure) as error:
        env.play_once()
    evidence = error.value.diagnostics
    assert evidence["kind"] == "exception_after_planning_failure"
    assert evidence["exception_type"] == "IndexError"
    assert evidence["exception_message"] == "list index out of range"
    assert "IndexError: list index out of range" in evidence["traceback"]
    assert "right_action[1][0].target_pose[2] += 0.06" in evidence["traceback"]
    assert evidence["actor"] is None and evidence["arm"] is None and evidence["contact_point_ids"] == []
    assert isinstance(error.value.__cause__, IndexError)
    assert env.plan_success is False
    json.dumps(evidence, allow_nan=False)


def test_successful_paths_keep_official_actions_after_wrapper():
    original, _ = bottle_environment(BottleTask, "Success")
    adapted, _ = bottle_environment(AdaptedBottleTask, "Success")
    assert original.play_once() == adapted.play_once() == {"info": {}}
    assert original.plan_success is adapted.plan_success is True


@pytest.mark.parametrize("error", [IndexError("list index out of range"), RuntimeError("CUDA error"),
                                   AssertionError("target_pose cannot be None"), KeyError("info")])
def test_exception_with_successful_planning_remains_a_runtime_error(error):
    class Task(official.OfficialBaseTask):
        def play_once(self):
            raise error

    env, _ = environment(type("AdaptedTask", (PlanningMixin, Task), {}), [])
    with pytest.raises(type(error)) as info:
        env.play_once()
    assert info.value is error and env.plan_success is True


def test_official_recovery_reset_restores_runtime_error_classification():
    class Task(official.OfficialBaseTask):
        def play_once(self):
            self.plan_success = False
            self.plan_success = True  # official "Try new way" (place_can_basket, open_microwave)
            raise IndexError("list index out of range")

    env, _ = environment(type("AdaptedTask", (PlanningMixin, Task), {}), [])
    with pytest.raises(IndexError):
        env.play_once()


def test_explicit_planning_failures_pass_through_the_wrapper_unchanged():
    env, calls = environment(AdaptedBottleTask, ["Fail", "Fail"])
    with pytest.raises(ExpertPlanningFailure) as error:
        env.play_once()
    assert error.value.diagnostics["kind"] == "no_reachable_grasp"
    assert error.value.__cause__ is None and calls == ["Fail", "Fail"]


def test_non_exception_interrupts_are_not_classified():
    class Task(official.OfficialBaseTask):
        def play_once(self):
            self.plan_success = False
            raise KeyboardInterrupt

    env, _ = environment(type("AdaptedTask", (PlanningMixin, Task), {}), [])
    with pytest.raises(KeyboardInterrupt):
        env.play_once()


def test_worker_records_crash_after_planner_failure_as_rejection(monkeypatch):
    from robotwin_phys import simulator
    env, _ = bottle_environment(AdaptedBottleTask, "Fail")
    disposed = []
    env.initial_state = lambda: {}
    env.dispose = lambda: disposed.append(True)
    monkeypatch.setattr(simulator, "create_env", lambda *a, **kw: env)
    result = worker.run({"mode": "expert", "episode": {"condition": {}}})
    assert result["status"] == "rejected" and result["reason"] == "expert_failed"
    assert result["planning_failure"]["kind"] == "exception_after_planning_failure"
    assert result["planning_failure"]["exception_type"] == "IndexError"
    assert result["plan_success"] is False and disposed == [True]
    json.dumps(result, allow_nan=False)


def test_worker_still_aborts_on_crash_without_planner_failure(monkeypatch):
    from robotwin_phys import simulator

    class Task(official.OfficialBaseTask):
        def play_once(self):
            raise IndexError("list index out of range")

    env, _ = environment(type("AdaptedTask", (PlanningMixin, Task), {}), [])
    env.initial_state, env.dispose = lambda: {}, lambda: None
    monkeypatch.setattr(simulator, "create_env", lambda *a, **kw: env)
    with pytest.raises(IndexError):
        worker.run({"mode": "expert", "episode": {"condition": {}}})

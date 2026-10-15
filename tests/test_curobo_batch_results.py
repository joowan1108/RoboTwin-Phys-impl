"""Pinned producer -> motion attempt -> clone/copy -> return, with real CPU tensors.

Only CUDA/optimization kernels and robot inputs are stand-ins. Control flow,
trajectory result production, best selection, and result merging are NVIDIA's
pinned method bodies, including the profiler decorator.
"""
import copy
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from robotwin_phys import curobo_compat as compat
from test_curobo_mask_compat import planner


class Trajectory:
    def __init__(self, position, velocity=None):
        self.position = position
        self.velocity = torch.ones_like(position) if velocity is None else velocity

    @property
    def shape(self):
        return self.position.shape

    def clone(self):
        return Trajectory(self.position.clone(), self.velocity.clone())

    def __getitem__(self, index):
        return Trajectory(self.position[index], self.velocity[index])

    def copy_at_index(self, other, index):
        self.position[index] = other.position
        self.velocity[index] = other.velocity

    def scale_by_dt(self, old_dt, new_dt):
        return Trajectory(self.position.clone(), self.velocity * (old_dt / new_dt))

    def repeat_seeds(self, count):
        assert count == 1
        return self

    @staticmethod
    def from_position(position):
        return Trajectory(position)


class Config(SimpleNamespace):
    def clone(self):
        return copy.copy(self)


class Pose:
    batch = 8

    def clone(self):
        return copy.copy(self)

    def repeat_seeds(self, count):
        assert count == 1
        return self


def trajopt_module():
    spec = importlib.util.spec_from_file_location('batch_result_trajopt',
            Path(__file__).parent / 'fixtures/curobo_trajopt.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_method(cls, name):
    function = inspect.unwrap(getattr(cls, name))
    function.__code__ = compat.repaired_function(function, name).__code__


def produce(module, flags, attempt=0, all_solutions=True, goalset=True):
    batch, horizon = 8, 5000
    solver = module.TrajOptSolver()
    solver.trim_steps = None
    solver.sync_cuda_time = False
    solver.evaluate_interpolated_trajectory = False
    solver.position_threshold = solver.rotation_threshold = solver.cspace_threshold = .01
    solver.solver_dt_tensor = torch.tensor(.02)
    solver.store_debug_in_result = False
    solver._col = torch.arange(batch)
    solver._velocity_bounds = None
    solver.traj_evaluator_config = SimpleNamespace(max_dt=1.)
    solver.solver = SimpleNamespace(rollout_fn=SimpleNamespace(traj_dt=.02,
        dynamics_model=SimpleNamespace(cspace_distance_weight=1.)))
    solver.traj_evaluator = SimpleNamespace(evaluate=lambda *a: (
        torch.ones(batch, dtype=torch.bool), torch.zeros(batch)))
    action = Trajectory(torch.full((batch, 4, 6), float(attempt)))
    interpolation = Trajectory(torch.full((batch, horizon, 6), float(attempt)))
    dt = torch.full((batch,), .02 + attempt * .01)
    lengths = [horizon - int(attempt) - i for i in range(batch)]
    solver.get_interpolated_trajectory = lambda action: (interpolation, lengths, dt, False)
    indices = (torch.arange(batch) + int(attempt) * 10).view(batch, 1).repeat(1, horizon)
    indices[:, -1] += 1  # Prove terminal selection, rather than arbitrarily taking the first step.
    metrics = SimpleNamespace(feasible=torch.tensor(flags).view(batch, 1).expand(batch, horizon),
        position_error=torch.zeros(batch, horizon), rotation_error=torch.zeros(batch, horizon),
        cspace_error=None, pose_error=torch.zeros(batch, horizon), cost=torch.zeros(batch, horizon),
        goalset_index=indices if goalset else None)
    wrap = SimpleNamespace(action=action, raw_action=action.position, metrics=metrics,
                           solve_time=.1, debug={})
    result = solver._get_result(wrap, all_solutions, SimpleNamespace(batch=batch), None, 1, batch_mode=True)
    return result, metrics


def full_planner(sequence, repaired=True, finetune=True):
    module, env, ik, solve, _, _, _, _ = planner(8, graph_mode='fail')
    trajopt = trajopt_module()
    module.JointState = Trajectory
    if repaired:
        for cls, names in ((module.MotionGen, {**compat.METHOD_HASHES, **compat.ATTEMPT_METHOD_HASHES}),
                           (module.MotionGenResult, compat.RESULT_METHOD_HASHES),
                           (trajopt.TrajOptSolver, compat.TRAJOPT_METHOD_HASHES)):
            for name in names:
                patch_method(cls, name)
    else:
        # Keep the earlier mask fix, to isolate the newly reported aggregation error.
        for name in compat.METHOD_HASHES:
            patch_method(module.MotionGen, name)
    solve.batch_env = False
    env.update_batch_size = lambda **kwargs: None
    env.trajopt_solver.interpolation_type = 'original'
    env.trajopt_solver.minimum_trajectory_dt = .001
    env.finetune_dt_scale = .9
    env.finetune_trajopt_solver = SimpleNamespace(update_solver_dt=lambda dt: None,
                                                 interpolation_type='original')
    config = Config(enable_graph=True, enable_opt=True, use_nn_ik_seed=False,
        partial_ik_opt=False, enable_finetune_trajopt=finetune, pose_cost_metric=None,
        enable_graph_attempt=None, disable_graph_attempt=None, max_attempts=len(sequence),
        timeout=1000, ik_fail_return=None, fail_on_invalid_query=True,
        success_ratio=1., time_dilation_factor=None)
    calls, results = [], []
    attempt = -1

    def optimize(*args, **kwargs):
        nonlocal attempt
        is_finetune = kwargs.get('trajopt_instance') is env.finetune_trajopt_solver
        if not is_finetune:
            attempt += 1
        all_solutions = kwargs.get('return_all_solutions', False)
        calls.append((attempt, is_finetune, all_solutions))
        value = float(attempt) + (.25 if is_finetune else 0.)
        result, _ = produce(trajopt, sequence[attempt], value, all_solutions)
        results.append(result)
        return result

    env._solve_trajopt_from_solve_state = optimize
    return module, env, solve, Trajectory(torch.zeros(8, 6)), Pose(), config, calls, results


@pytest.fixture(autouse=True)
def cpu_cuda_boundary(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda **kwargs: None)


@pytest.mark.parametrize('count', [2, 5])
def test_pinned_full_attempt_loop_reproduces_5000_column_mismatch(count):
    sequence = [[False] * 8, [i < count for i in range(8)]]
    module, env, solve, state, pose, config, calls, results = full_planner(sequence, repaired=False)
    with pytest.raises(RuntimeError, match=r'shape mismatch.*5000'):
        env._plan_batch_attempts(solve, state, pose, config)
    assert calls == [(0, False, True), (1, False, True), (1, True, False)]
    assert results[0].goalset_index.shape == (8, 5000)
    assert results[-1].goalset_index.shape == (8,)


@pytest.mark.parametrize('goalset', [True, False])
def test_producer_all_seed_and_best_select_metadata_agree(goalset):
    module = trajopt_module()
    wrapper = module.TrajOptSolver._get_result
    patch_method(module.TrajOptSolver, '_get_result')
    assert module.TrajOptSolver._get_result is wrapper
    all_result, metrics = produce(module, [True] * 8, all_solutions=True, goalset=goalset)
    selected, _ = produce(module, [True] * 8, all_solutions=False, goalset=goalset)
    if goalset:
        assert torch.equal(all_result.goalset_index, selected.goalset_index)
        assert all_result.goalset_index.shape == (8,)
        assert metrics.goalset_index.shape == (8, 5000)  # Raw diagnostics remain intact.
    else:
        assert all_result.goalset_index is selected.goalset_index is None


@pytest.mark.parametrize('finetune', [True, False])
@pytest.mark.parametrize('sequence', [
    [[False]*8, [True,True,False,False,False,False,False,False], [False,False,True,True,True,True,True,True]],
    [[True,False,True,False,True,False,True,False], [False,True,False,True,False,True,False,True]],
    [[True,False,False,False,False,False,False,False], [False]*8],
    [[False]*8, [False]*8],
    [[True]*8],
])
def test_full_loop_merges_correct_trajectory_dt_goal_and_length(sequence, finetune):
    module, env, solve, state, pose, config, calls, results = full_planner(sequence, finetune=finetune)
    final = env._plan_batch_attempts(solve, state, pose, config)
    assert final.attempts == len(sequence)
    assert torch.equal(final.success, torch.tensor(sequence).any(0))
    assert final.goalset_index.shape == (8,)
    for row in range(8):
        successes = [n for n, flags in enumerate(sequence) if flags[row]]
        attempt = successes[-1] if successes else 0
        # Clone of first attempt retains its unsuccessful rows, too.
        was_finetuned = finetune and any(sequence[attempt])
        value = attempt + (.25 if was_finetuned else 0.)
        assert torch.all(final.optimized_plan.position[row] == value)
        assert torch.all(final.interpolated_plan.position[row] == value)
        assert final.optimized_dt[row].item() == pytest.approx(.02 + value*.01)
        velocity = final.optimized_plan.velocity[row]
        assert torch.allclose(velocity, torch.full_like(velocity, .02 / (.02 + value*.01)))
        assert final.goalset_index[row].item() == row + attempt*10 + 1
        assert final.path_buffer_last_tstep[row] == 5000 - attempt - row
    assert final.path_buffer_last_tstep is not results[0].path_buffer_last_tstep
    assert env.trajopt_solver.interpolation_type == 'original'


def test_clone_owns_path_list_and_preserves_finetune_statistics():
    module, env, solve, state, pose, config, calls, results = full_planner([[True]*8])
    final = env._plan_batch_attempts(solve, state, pose, config)
    assert final.finetune_time == pytest.approx(.1)
    duplicate = final.clone()
    duplicate.path_buffer_last_tstep[0] = 1
    assert final.path_buffer_last_tstep[0] == 5000


def test_retiming_operates_on_returned_accumulation_after_last_attempt_failed(monkeypatch):
    module, env, solve, state, pose, config, calls, results = full_planner(
        [[True,False,False,False,False,False,False,False], [False]*8])
    config.time_dilation_factor = .5
    retimed = []
    monkeypatch.setattr(module.MotionGenResult, 'retime_trajectory',
                        lambda self, factor, **kwargs: retimed.append((self, factor)))
    final = env._plan_batch_attempts(solve, state, pose, config)
    assert len(retimed) == 1 and retimed[0][0] is final and retimed[0][1] == .5


def test_invalid_pose_metric_returns_normal_invalid_query_before_attempts():
    module, env, solve, state, pose, config, calls, results = full_planner([[True]*8])
    config.pose_cost_metric = object()
    env.update_pose_cost_metric = lambda *a: False
    final = env._plan_batch_attempts(solve, state, pose, config)
    assert final.valid_query is False and not final.success.any()
    assert final.success.shape == (8,) and calls == []


def test_actual_runtime_errors_are_not_reclassified_as_infeasible():
    module, env, solve, state, pose, config, calls, results = full_planner([[True]*8])
    def broken(*args, **kwargs):
        raise RuntimeError('CUDA kernel configuration error')
    env._solve_trajopt_from_solve_state = broken
    with pytest.raises(RuntimeError, match='CUDA kernel configuration error'):
        env._plan_batch_attempts(solve, state, pose, config)

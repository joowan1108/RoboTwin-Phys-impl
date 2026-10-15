"""Replay pinned CuRobo batch control flow with actual PyTorch tensors on CPU."""
import importlib.util
import ast
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from robotwin_phys import curobo_compat as compat


def fixture_module():
    spec = importlib.util.spec_from_file_location('official_motion_gen',
            Path(__file__).parent / 'fixtures/curobo_motion_gen.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class State:
    def __init__(self, batch):
        self.position = torch.zeros(batch, 6)

    def repeat_seeds(self, count):
        assert count == 1
        return self


def planner(successes, graph_mode='fail', valid_query=True, optimize=True, traj_success=True):
    module = fixture_module()
    env = module.MotionGen()
    batch = 8
    mask = torch.zeros(batch, 1, dtype=torch.bool)
    mask[:successes] = True
    solution = torch.arange(batch * 6, dtype=torch.float32).view(batch, 1, 6)
    ik = SimpleNamespace(success=mask, solution=solution, position_error=None, rotation_error=None, solve_time=0.)
    env._trajopt_goal_config = torch.empty(batch, 1, 6)
    env.get_retract_config = lambda: torch.zeros(6)
    env._dof = 6
    env.ik_solver = SimpleNamespace(dof=6)
    env.tensor_args = SimpleNamespace(device='cpu', dtype=torch.float32)
    env.store_debug_in_result = False
    env.graph_trajopt_iters = 1
    env._solve_ik_from_solve_state = lambda *args: ik
    graph_flags = torch.zeros(successes, dtype=torch.bool)
    if graph_mode == 'success':
        graph_flags[:] = True
    elif graph_mode == 'partial':
        graph_flags[::2] = True
    graph_count = int(graph_flags.sum())
    graph = SimpleNamespace(success=graph_flags, valid_query=valid_query, solve_time=0.,
                            interpolated_plan=SimpleNamespace(position=torch.zeros(graph_count, 4, 6)))
    env.graph_search = lambda *args: graph
    env.trajopt_solver = SimpleNamespace(action_horizon=4,
        get_seed_set=lambda *args, **kw: torch.zeros(batch, 4, 6))
    calls = []

    def trajopt(*args, **kwargs):
        calls.append(env._trajopt_goal_config.clone())
        flags = mask[:, 0].clone() if traj_success else torch.zeros(batch, dtype=torch.bool)
        return SimpleNamespace(success=flags, interpolated_solution=None, solve_time=0.,
            position_error=None, rotation_error=None, cspace_error=None, goalset_index=None,
            path_buffer_last_tstep=None, solution=None, optimized_dt=None)

    env._solve_trajopt_from_solve_state = trajopt
    solve = SimpleNamespace(num_trajopt_seeds=1, batch_size=batch, batch_mode=True)
    config = SimpleNamespace(enable_graph=True, enable_opt=optimize, use_nn_ik_seed=False,
                              partial_ik_opt=False, enable_finetune_trajopt=False)
    pose = SimpleNamespace(repeat_seeds=lambda count: None)
    return module, env, ik, solve, State(batch), pose, config, calls


def apply(module):
    for name in compat.METHOD_HASHES:
        fn = getattr(module.MotionGen, name)
        fn.__code__ = compat.repaired_function(fn, name).__code__


@pytest.mark.parametrize('successes', [3, 5])
def test_original_reproduces_reported_shape_mismatch(successes):
    module, env, ik, solve, state, pose, config, calls = planner(successes)
    with pytest.raises(RuntimeError, match='shape mismatch'):
        env._plan_from_solve_state_batch(solve, state, pose, config)
    assert not ik.success.any()  # Graph failure corrupted the earlier IK result.
    assert calls == []


@pytest.mark.parametrize('successes', [3, 5])
@pytest.mark.parametrize('traj_success', [True, False])
def test_graph_failure_keeps_ik_mask_and_allows_intended_trajopt_fallback(successes, traj_success):
    module, env, ik, solve, state, pose, config, calls = planner(successes, traj_success=traj_success)
    original_mask = ik.success.clone()
    apply(module)
    result = env._plan_from_solve_state_batch(solve, state, pose, config)
    assert torch.equal(ik.success, original_mask)
    assert len(calls) == 1
    assert torch.equal(calls[0][original_mask], ik.solution[original_mask])
    expected = original_mask[:, 0] if traj_success else torch.zeros(8, dtype=torch.bool)
    assert torch.equal(result.success, expected)
    if not traj_success:
        assert result.status == module.MotionGenStatus.TRAJOPT_FAIL


@pytest.mark.parametrize('optimize', [True, False])
def test_invalid_graph_query_returns_failure_without_corrupting_ik(optimize):
    module, env, ik, solve, state, pose, config, calls = planner(3, valid_query=False, optimize=optimize)
    mask = ik.success.clone()
    apply(module)
    result = env._plan_from_solve_state_batch(solve, state, pose, config)
    assert result.valid_query is False and not result.success.any()
    assert result.status == module.MotionGenStatus.GRAPH_FAIL and calls == []
    assert torch.equal(ik.success, mask)


def test_zero_ik_success_stays_a_normal_ik_failure():
    module, env, ik, solve, state, pose, config, calls = planner(0)
    apply(module)
    result = env._plan_from_solve_state_batch(solve, state, pose, config)
    assert result.status == module.MotionGenStatus.IK_FAIL and calls == []
    assert not result.success.any()


def test_graph_only_failure_does_not_corrupt_ik_or_start_optimization():
    module, env, ik, solve, state, pose, config, calls = planner(5, optimize=False)
    mask = ik.success.clone()
    apply(module)
    result = env._plan_from_solve_state_batch(solve, state, pose, config)
    assert not result.success.any() and calls == []
    assert torch.equal(mask, ik.success)


def test_single_plan_result_also_has_independent_success_storage():
    module = fixture_module()
    module.profiler = SimpleNamespace(record_function=lambda name: nullcontext())
    module.tensor_repeat_seeds = lambda value, seeds: value.repeat(seeds, 1)
    module.log_info = lambda message: None
    env = module.MotionGen()
    ik = SimpleNamespace(success=torch.ones(1, 1, dtype=torch.bool),
                         solution=torch.zeros(1, 1, 6), solve_time=0.)
    env._solve_ik_from_solve_state = lambda *args: ik
    env._trajopt_goal_config = torch.empty(1, 1, 6)
    env.ik_solver = SimpleNamespace(dof=6)
    env.trajopt_solver = SimpleNamespace(action_horizon=4)
    env.store_debug_in_result = False
    env.graph_search = lambda *args: SimpleNamespace(success=torch.zeros(1, dtype=torch.bool),
                                                     valid_query=False, solve_time=0.)
    config = SimpleNamespace(enable_graph=True, enable_opt=True, use_nn_ik_seed=False, partial_ik_opt=False)
    state = SimpleNamespace(position=torch.zeros(1, 6), shape=(1, 6))
    solve = SimpleNamespace(num_trajopt_seeds=1)
    pose = SimpleNamespace(shape=(1,))
    apply(module)
    result = env._plan_from_solve_state(solve, state, pose, config)
    assert not result.success.any() and result.valid_query is False
    assert ik.success.all()


def test_method_guard_is_independent_of_python_312_type_params():
    source = ast.parse('def example(self):\n    return True').body[0]
    expected = compat.method_checksum(source)
    if hasattr(source, 'type_params'):
        del source.type_params
    assert compat.method_checksum(source) == expected


@pytest.mark.parametrize('mode', ['success', 'partial'])
def test_other_graph_paths_keep_the_original_results(mode):
    original = planner(5, graph_mode=mode)
    fixed = planner(5, graph_mode=mode)
    old = original[1]._plan_from_solve_state_batch(*original[3:7])
    apply(fixed[0])
    new = fixed[1]._plan_from_solve_state_batch(*fixed[3:7])
    assert torch.equal(old.success, new.success)
    assert torch.equal(original[-1][0], fixed[-1][0])


def test_method_defaults_and_descriptor_identity_are_preserved():
    module = fixture_module()
    fn = module.MotionGen._plan_from_solve_state_batch
    defaults = fn.__defaults__
    apply(module)
    assert module.MotionGen._plan_from_solve_state_batch is fn
    assert fn.__defaults__ is defaults


def test_unreviewed_function_is_rejected():
    def different(self):
        return True

    with pytest.raises(RuntimeError, match='Unreviewed'):
        compat.repaired_function(different, '_plan_from_solve_state_batch')


def test_install_is_idempotent_and_checks_source_location(monkeypatch):
    module = fixture_module()
    spec = importlib.util.spec_from_file_location('official_trajopt',
            Path(__file__).parent / 'fixtures/curobo_trajopt.py')
    trajopt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trajopt)
    module.__file__ = str(compat.ROOT / 'vendor/curobo/src/curobo/wrap/reacher/motion_gen.py')
    trajopt.__file__ = str(compat.ROOT / 'vendor/curobo/src/curobo/wrap/reacher/trajopt.py')
    monkeypatch.setattr(compat, 'compatibility_metadata', lambda: {})
    monkeypatch.setattr(compat.importlib, 'import_module',
                        lambda name: trajopt if name.endswith('.trajopt') else module)
    compat.install()
    code = module.MotionGen._plan_from_solve_state_batch.__code__
    compat.install()
    assert module.MotionGen._plan_from_solve_state_batch.__code__ is code
    assert getattr(trajopt.TrajOptSolver._get_result.__wrapped__, '_robotwin_mask_fix') == compat.FIX
    module.__file__ = '/different/motion_gen.py'
    with pytest.raises(RuntimeError, match='outside'):
        compat.install()

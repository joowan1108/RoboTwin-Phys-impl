"""Exercise the original placement helper and its shared numerical repairs."""
import importlib.util
import ast
from pathlib import Path

import numpy as np
import pytest
import transforms3d as t3d
from types import SimpleNamespace

from robotwin_phys import numerics

ROOT = Path(__file__).parents[1]


def original():
    spec = importlib.util.spec_from_file_location('official_transforms',
             Path(__file__).parent / 'fixtures/robotwin_transforms.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixed():
    module = original()
    for name in ('get_align_matrix', 'cal_quat_dis', 'get_product_vector', 'rotate_cone'):
        setattr(module, name, getattr(numerics, name))
    return module


def test_float32_roundoff_reproduces_original_nan_and_is_repaired():
    rng = np.random.default_rng(8)
    pairs = []
    for _ in range(10000):
        first = rng.normal(size=3).astype(np.float32)
        second = (first + rng.normal(size=3) * 1e-4).astype(np.float32)
        a, b = first / np.linalg.norm(first), second / np.linalg.norm(second)
        if np.dot(a, b) > 1 and np.linalg.norm(np.cross(a, b)) > 1e-6:
            pairs.append((first, second))
            break
    assert pairs, 'Regression must exercise an out-of-domain arccos input'
    first, second = pairs[0]
    with np.errstate(invalid='ignore'):
        broken = original().get_align_matrix(first, second)
    assert not np.isfinite(broken).all()
    with pytest.raises(np.linalg.LinAlgError, match='Eigenvalues'):
        t3d.quaternions.mat2quat(broken)
    with np.errstate(all='raise'):
        repaired = numerics.get_align_matrix(first, second)
        quaternion = t3d.quaternions.mat2quat(repaired)
    assert np.isfinite(quaternion).all()
    np.testing.assert_allclose(repaired @ numerics.unit_vector(first), numerics.unit_vector(second), atol=1e-7)


@pytest.mark.parametrize('first,second', [([1, 0, 0], [1, 0, 0]),
                                       ([1, 0, 0], [-1, 0, 0]),
                                       ([0, 1, 0], [0, -1, 0]),
                                       ([0, 0, 1], [0, 0, -1]),
                                       ([1, 2, 3], [-1, -2, -3]),
                                       ([1, 0, 0], [0, 1, 0])])
def test_parallel_antiparallel_and_perpendicular_directions(first, second):
    with np.errstate(all='raise'):
        rotation = numerics.get_align_matrix(first, second)
    np.testing.assert_allclose(rotation @ numerics.unit_vector(first), numerics.unit_vector(second), atol=1e-7)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.)


def test_ordinary_rotations_preserve_upstream_geometry():
    rng = np.random.default_rng(4)
    upstream = original()
    for _ in range(300):
        first, second = rng.normal(size=(2, 3))
        np.testing.assert_allclose(numerics.get_align_matrix(first, second),
                                   upstream.get_align_matrix(first, second), atol=2e-14)


@pytest.mark.parametrize('axis', [[0, 0, 0], [1e-20, 0, 0]])
def test_undefined_finite_direction_is_explicit(axis):
    with pytest.raises(numerics.DegenerateDirectionError, match='Zero-length'):
        numerics.get_align_matrix(axis, [1, 0, 0])


@pytest.mark.parametrize('bad', [[np.nan, 0, 1], [np.inf, 0, 1]])
def test_invalid_input_remains_a_runtime_error(bad):
    with pytest.raises(ValueError, match='Non-finite'):
        numerics.get_align_matrix(bad, [1, 0, 0])


def test_quaternion_distance_clamps_roundoff_and_preserves_sign_invariance():
    rng = np.random.default_rng(13)
    upstream = original()
    for _ in range(200):
        a, b = rng.normal(size=(2, 4))
        assert numerics.cal_quat_dis(a, b) == pytest.approx(upstream.cal_quat_dis(a, b), abs=3e-14)
        with np.errstate(all='raise'):
            assert numerics.cal_quat_dis(a.astype(np.float32), -a.astype(np.float32)) == pytest.approx(0, abs=2e-8)
    with pytest.raises(ValueError, match='Invalid quaternion'):
        numerics.cal_quat_dis([0, 0, 0, 0], [1, 0, 0, 0])


def test_original_placement_function_uses_shared_fixed_helpers():
    module = fixed()
    rng = np.random.default_rng(15)
    for _ in range(100):
        actor = module.Pose([.1, -.1, .7], rng.normal(size=4))
        target = module.Pose([.2, .1, .8], rng.normal(size=4))
        with np.errstate(all='raise'):
            result = module.get_place_pose(actor, target, constrain='align', align_axis=[0, 1, 0],
                                            actor_axis=[.2, .4, .7], actor_axis_type='world', z_transform=False)
        assert np.isfinite(result).all()
        np.testing.assert_allclose(result[:3], target.p, atol=1e-7)
        assert np.linalg.norm(result[3:]) == pytest.approx(1.)


def test_collinear_view_up_vector_produces_a_valid_frame():
    with np.errstate(all='raise'):
        matrix = numerics.rotate_cone([0, 0, 0], [0, 0, 1])
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
    assert np.linalg.det(matrix) == pytest.approx(1.)


def test_patch_is_guarded_idempotent_and_replaces_all_shared_helpers():
    spec = importlib.util.spec_from_file_location('patch_robotwin', ROOT / 'tools/patch_robotwin.py')
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    upstream = (ROOT / 'vendor/RoboTwin/envs/utils/transforms.py').read_text()
    result = patch.patch_transform_text(upstream)
    assert patch.patch_transform_text(result) == result
    assert 'axis, angle = alignment_axis_angle(v1, v2)' in result
    assert 'get_align_matrix, cal_quat_dis, get_product_vector, rotate_cone' in result
    with pytest.raises(RuntimeError, match='implementation changed'):
        patch.patch_transform_text('version = 1')


def test_planner_runtime_exception_is_not_converted_into_fake_failures():
    spec = importlib.util.spec_from_file_location('patch_robotwin', ROOT / 'tools/patch_robotwin.py')
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    source = (ROOT / 'vendor/RoboTwin/envs/robot/planner.py').read_text()
    repaired = patch.patch_planner_text(source)
    assert patch.patch_planner_text(repaired) == repaired
    # Execute the exact original exception block and its repaired call, without
    # CUDA initialization, against a failing MotionGen implementation.
    tree = ast.parse(source)
    block = next(node for node in ast.walk(tree) if isinstance(node, ast.Try)
                 and any(isinstance(item, ast.Assign) and isinstance(item.value, ast.Call)
                         and isinstance(item.value.func, ast.Attribute) and item.value.func.attr == 'plan_batch'
                         for item in node.body))

    def raise_cuda(*args):
        raise RuntimeError('CUDA out of memory')

    mock = SimpleNamespace(motion_gen_batch=SimpleNamespace(plan_batch=raise_cuda))
    original_function = ast.parse('def original(self, start_joint_states, goal_pose_of_ee, plan_config):\n    pass')
    original_function.body[0].body = [block]
    original_function = ast.fix_missing_locations(original_function)
    scope = {}
    exec(compile(original_function, '<official-exception-block>', 'exec'), scope)
    assert scope['original'](mock, None, None, None) == {'status': ['Failure'] * 10}
    fixed_tree = ast.parse(repaired)
    assignment = next(node for node in ast.walk(fixed_tree) if isinstance(node, ast.Assign)
                      and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                      and node.value.func.attr == 'plan_batch')
    original_function.body[0].body = [assignment]
    exec(compile(ast.fix_missing_locations(original_function), '<fixed-exception-block>', 'exec'), scope)
    with pytest.raises(RuntimeError, match='CUDA out of memory'):
        scope['original'](mock, None, None, None)

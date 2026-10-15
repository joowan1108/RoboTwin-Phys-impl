"""Construction hooks called by the small, pinned RoboTwin factory patch."""
from contextlib import contextmanager
from contextvars import ContextVar
import copy
import numpy as np

_environment = ContextVar("robotwin_phys_construction", default=None)
# Official clutter distractors are built at their nominal size: the paper's object
# attributes describe the manipulated objects. Their poses still follow the tilted table.
_clutter = ContextVar("robotwin_phys_constructing_clutter", default=False)


@contextmanager
def construction(env, clutter=False):
    token, clutter_token = _environment.set(env), _clutter.set(clutter)
    try:
        yield
    finally:
        _clutter.reset(clutter_token)
        _environment.reset(token)


def _scale(env):
    return 1.0 if _clutter.get() else env.condition["geometry_scale"]


def geometry_size(size):
    env = _environment.get()
    if env is None or _clutter.get():
        return size
    value = np.asarray(size) * env.condition["geometry_scale"]
    return float(value) if value.ndim == 0 else value.tolist()


def geometry_model(scale, data):
    scale = geometry_size(scale)
    if data is not None:
        data = copy.deepcopy(data)
        data["scale"] = copy.deepcopy(scale)
    return scale, data


def object_pose(pose):
    env = _environment.get()
    if env is None:
        return pose
    import sapien
    import transforms3d as t3d
    center = np.array([*env.table_xy_bias, 0.74 + env.table_z_bias])
    p = np.array(pose.p, dtype=float)
    p[2] = center[2] + (p[2] - center[2]) * _scale(env)
    rotation = t3d.axangles.axangle2mat([1, 0, 0], np.deg2rad(env.condition["table_tilt_deg"]))
    # Azimuth 90 degrees: surface normal tilts toward +Y.
    rotation = rotation.T
    return sapien.Pose(center + rotation @ (p - center),
                       t3d.quaternions.mat2quat(rotation @ t3d.quaternions.quat2mat(pose.q)))

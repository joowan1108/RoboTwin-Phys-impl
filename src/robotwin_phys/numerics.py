"""Finite rotations for the shared, pinned RoboTwin transform helpers."""
import numpy as np
import transforms3d as t3d


class DegenerateDirectionError(ValueError):
    """Finite geometry has no direction to align (for example a zero projection)."""


def unit_vector(value, label="direction"):
    value = np.asarray(value, dtype=float).reshape(3)
    if not np.isfinite(value).all():
        raise ValueError(f"Non-finite {label}: {value.tolist()}")
    length = np.linalg.norm(value)
    if not np.isfinite(length):
        raise ValueError(f"Non-finite norm for {label}")
    if length < 1e-12:
        raise DegenerateDirectionError(f"Zero-length {label}: {value.tolist()}")
    return value / length


def alignment_axis_angle(v1, v2):
    first, second = unit_vector(v1), unit_vector(v2)
    axis = np.cross(first, second)
    cosine = float(np.clip(np.dot(first, second), -1., 1.))
    angle = float(np.arccos(cosine))
    if np.linalg.norm(axis) < 1e-6:
        if cosine >= 0:
            # Preserve the upstream tolerance for an already aligned direction.
            return np.array([1., 0., 0.]), 0.
        # Anti-parallel vectors need a 180 degree turn, not the original identity.
        basis = np.eye(3)[np.argmin(np.abs(first))]
        axis = np.cross(first, basis)
        angle = float(np.pi)
    return unit_vector(axis, "rotation axis"), angle


def get_align_matrix(v1, v2):
    axis, angle = alignment_axis_angle(v1, v2)
    return np.eye(3) if angle == 0 else t3d.axangles.axangle2mat(axis, angle)


def cal_quat_dis(quat1, quat2):
    quaternions = []
    for value in (quat1, quat2):
        value = np.asarray(value, dtype=float).reshape(4)
        length = np.linalg.norm(value)
        if not np.isfinite(value).all() or not np.isfinite(length) or length < 1e-12:
            raise ValueError("Invalid quaternion in rotation distance")
        quaternions.append(value / length)
    delta = t3d.quaternions.qmult(t3d.quaternions.qinverse(quaternions[0]), quaternions[1])
    cosine = np.clip(np.abs(delta[0] / np.linalg.norm(delta)), 0., 1.)
    return 2 * np.arccos(cosine) / np.pi


def get_product_vector(v1, v2):
    axis = unit_vector(v1, "projection axis")
    vector = np.asarray(v2, dtype=float).reshape(3)
    if not np.isfinite(vector).all():
        raise ValueError("Non-finite projected vector")
    return np.dot(axis, vector) * axis


def rotate_cone(new_pt, origin, z_dir=(0, 0, 1)):
    x = unit_vector(np.asarray(origin) - np.asarray(new_pt), "view direction")
    up = unit_vector(z_dir, "view up direction")
    z = up - np.dot(x, up) * x
    if np.linalg.norm(z) < 1e-12:
        basis = np.eye(3)[np.argmin(np.abs(x))]
        z = basis - np.dot(x, basis) * x
    z = unit_vector(z)
    return np.stack([x, np.cross(z, x), z], axis=1)

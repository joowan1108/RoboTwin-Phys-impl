# Extracted verbatim from RoboTwin-Platform/RoboTwin, MIT license.
# Revision: 13c3c47ff4312dd62484bcd51be034af55c062d1.
# Source: envs/utils/transforms.py. See ROBOTWIN_LICENSE.
from __future__ import annotations
import numpy as np
import transforms3d as t3d
from types import SimpleNamespace


class Pose:
    # Small CPU substitute for the pose container; float32 models SAPIEN output.
    def __init__(self, p=None, q=None):
        self.p = np.asarray([0, 0, 0] if p is None else p, dtype=np.float32)
        self.q = np.asarray([1, 0, 0, 0] if q is None else q, dtype=np.float32)
    def to_transformation_matrix(self):
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = t3d.quaternions.quat2mat(self.q)
        matrix[:3, 3] = self.p
        return matrix


sapien = SimpleNamespace(Pose=Pose)


def rotate_cone(new_pt: np.ndarray, origin: np.ndarray, z_dir: np.ndarray = [0, 0, 1]):
    x = origin - new_pt
    x = x / np.linalg.norm(x)
    bx_ = np.array(z_dir).reshape(3)
    z = bx_ - np.dot(x, bx_) * x
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)

def _tolist(pose: sapien.Pose | list | np.ndarray) -> list:
    if isinstance(pose, list):
        return pose
    elif isinstance(pose, sapien.Pose):
        return pose.p.tolist() + pose.q.tolist()
    else:
        return pose.tolist()

def _toPose(pose: sapien.Pose | list | np.ndarray) -> sapien.Pose:
    if isinstance(pose, list):
        assert len(pose) == 7 or len(pose) == 3
        if len(pose) == 3:
            return sapien.Pose(pose[:3], [1, 0, 0, 0])
        else:
            return sapien.Pose(pose[:3], pose[3:])
    elif isinstance(pose, np.ndarray):
        assert pose.shape == (7, ) or pose.shape == (3, )
        if pose.shape == (3, ):
            return sapien.Pose(pose[:3], [1, 0, 0, 0])
        else:
            return sapien.Pose(pose[:3], pose[3:])
    else:
        return pose

def rotate_along_axis(
    target_pose,
    center_pose,
    axis,
    theta: float = np.pi / 2,
    axis_type: Literal["center", "target", "world"] = "center",
    towards=None,
    camera_face=None,
) -> list:
    """
    以 center 为中心，沿着指定轴旋转指定角度。通过 towards 可指定旋转方向（方向为 center->target 向量与 towards 向量乘积为正的方向）

    target_pose: 目标点（比如在物体正上方的预抓取点）
    center_pose: 中心点（比如物体的位置）
    axis: 旋转轴
    theta: 旋转角度（单位：弧度）
    axis_type: 旋转轴的类型（'center'：相对于 center_pose，'target'：相对于 target_pose，'world'：世界坐标系），默认是 'center'
    towards: 旋转方向（可选），如果指定了这个参数，则会根据这个参数来决定旋转的方向
    camera_face: 相机朝向（可选），会限制相机向量与该向量点积为正；如果设置为 None，只旋转不考虑相机朝向
    返回值：列表，前3个元素是坐标，后4个元素是四元数
    """
    target_pose, center_pose = _toPose(target_pose), _toPose(center_pose)
    if theta == 0:
        return target_pose.p.tolist() + target_pose.q.tolist()
    rotate_mat = t3d.axangles.axangle2mat(axis, theta)

    target_mat = target_pose.to_transformation_matrix()
    center_mat = center_pose.to_transformation_matrix()
    if axis_type == "center":
        world_axis = (center_mat[:3, :3] @ np.array(axis).reshape(3, 1)).reshape(3)
    elif axis_type == "target":
        world_axis = (target_mat[:3, :3] @ np.array(axis).reshape(3, 1)).reshape(3)
    else:
        world_axis = np.array(axis).reshape(3)

    rotate_mat = t3d.axangles.axangle2mat(world_axis, theta)
    p = (rotate_mat @ (target_pose.p - center_pose.p).reshape(3, 1)).reshape(3) + center_pose.p
    if towards is not None:
        towards = np.dot(p - center_pose.p, np.array(towards).reshape(3))
        if towards < 0:
            rotate_mat = t3d.axangles.axangle2mat(world_axis, -theta)
            p = (rotate_mat @ (target_pose.p - center_pose.p).reshape(3, 1)).reshape(3) + center_pose.p

    if camera_face is None:
        q = t3d.quaternions.mat2quat(rotate_mat @ target_mat[:3, :3])
    else:
        q = t3d.quaternions.mat2quat(rotate_cone(p, center_pose.p, camera_face))
    return p.tolist() + q.tolist()

def rotate2rob(target_pose, rob_pose, box_pose, theta: float = 0.5) -> list:
    """
    向指定的 rob_pose 偏移
    """
    target_pose, rob_pose, box_pose = (
        _toPose(target_pose),
        _toPose(rob_pose),
        _toPose(box_pose),
    )

    target_mat = target_pose.to_transformation_matrix()
    v1 = (target_mat[:3, :3] @ np.array([[1, 0, 0]]).T).reshape(3)
    v2 = box_pose.p - rob_pose.p
    v2 = v2 / np.linalg.norm(v2)
    axis = np.cross(v1, v2)
    angle = np.arccos(np.dot(v1, v2))

    return rotate_along_axis(
        target_pose=target_pose,
        center_pose=box_pose,
        axis=axis,
        theta=angle * theta,
        axis_type="world",
        towards=-v2,
    )

def cal_quat_dis(quat1, quat2):
    qmult = t3d.quaternions.qmult
    qinv = t3d.quaternions.qinverse
    qnorm = t3d.quaternions.qnorm
    delta_quat = qmult(qinv(quat1), quat2)
    return 2 * np.arccos(np.fabs((delta_quat / qnorm(delta_quat))[0])) / np.pi

def get_align_matrix(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    获取从 v1 到 v2 的旋转矩阵
    """
    v1 = np.array(v1).reshape(3)
    v2 = np.array(v2).reshape(3)

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    axis = np.cross(v1, v2)
    angle = np.arccos(np.dot(v1, v2))

    if np.linalg.norm(axis) < 1e-6:
        return np.eye(3)
    else:
        return t3d.axangles.axangle2mat(axis, angle)

def get_product_vector(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    获取 v2 在 v1 上的投影向量
    """
    v1 = np.array(v1).reshape(3)
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.array(v2).reshape(3)
    return np.dot(v1, v2) * v1

def get_place_pose(
    actor_pose: np.ndarray | sapien.Pose | list,
    target_pose: np.ndarray | sapien.Pose | list,
    constrain: Literal["free", "align"] = "free",
    align_axis: list[np.ndarray] | np.ndarray | list = None,
    actor_axis: np.ndarray | list = [1, 0, 0],
    actor_axis_type: Literal["actor", "world"] = "actor",
    z_transform: bool = True,
) -> list:
    """
    获取物体应当被放置到的位置
    考虑因素：
        1. 三维坐标与给定坐标一致
        2. 物体的朝向合理
            - 物体 z 轴与给定坐标 z 轴一致
            - 满足在 xy 平面上的一定约束
                - 无约束（直接采用物体当前的 x,y 在 xOy 平面上的投影）
                - 物体的 x 轴对齐给定 x 轴
                - 选取物体的 x 轴与给定的世界轴单位向量集合中点积最小的方向

    actor_pose: 物体当前的 pose
    target_pose: 物体应当被放置到的位置
    constrain: 物体的约束类型
        - free: 无约束
        - align: 物体的 x 轴与给定的世界轴向量集合中点积最小的方向
    align_axis: 给定的世界轴向量集合，如果设置为 None，默认使用 target_pose 的 x 轴
    actor_axis: 计算点积的 actor 轴，默认使用 x 轴
    actor_axis_type: actor_axis 的类型，默认使用局部坐标系
        - actor: actor_pose 的局部坐标系
        - world: 世界坐标系
    """
    actor_pose_mat = _toPose(actor_pose).to_transformation_matrix()
    target_pose_mat = _toPose(target_pose).to_transformation_matrix()

    # 将物体的三维坐标与给定坐标对齐
    actor_pose_mat[:3, 3] = target_pose_mat[:3, 3]

    target_x = target_pose_mat[:3, 0]
    target_y = target_pose_mat[:3, 1]
    target_z = target_pose_mat[:3, 2]

    # 将物体的 z 轴与给定坐标的 z 轴对齐
    actor2world = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]).T
    if z_transform:
        z_align_matrix = get_align_matrix(actor_pose_mat[:3, :3] @ actor2world[:3, 2], target_z)
    else:
        z_align_matrix = get_align_matrix(actor_pose_mat[:3, 2], target_z)
    actor_pose_mat[:3, :3] = z_align_matrix @ actor_pose_mat[:3, :3]

    if constrain == "align":
        if align_axis is None:
            align_axis = np.array(target_pose_mat[:3, :3] @ np.array([[1, 0, 0]]).T)
        elif isinstance(align_axis, list):
            align_axis = np.array(align_axis).reshape((-1, 3)).T
        else:
            align_axis = np.array(align_axis).reshape((3, -1))
        align_axis = align_axis / np.linalg.norm(align_axis, axis=0)

        if actor_axis_type == "actor":
            actor_axis = actor_pose_mat[:3, :3] @ np.array(actor_axis).reshape(3, 1)
        elif actor_axis_type == "world":
            actor_axis = np.array(actor_axis)
        closest_axis_id = np.argmax(actor_axis.reshape(3) @ align_axis)
        align_axis = align_axis[:, closest_axis_id]

        actor_axis_xOy = get_product_vector(target_x, actor_axis) + get_product_vector(target_y, actor_axis)
        align_axis_xOy = get_product_vector(target_x, align_axis) + get_product_vector(target_y, align_axis)
        align_mat_xOy = get_align_matrix(actor_axis_xOy, align_axis_xOy)
        actor_pose_mat[:3, :3] = align_mat_xOy @ actor_pose_mat[:3, :3]

    return (actor_pose_mat[:3, 3].tolist() + t3d.quaternions.mat2quat(actor_pose_mat[:3, :3]).tolist())

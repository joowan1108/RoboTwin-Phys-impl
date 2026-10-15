# Extracted verbatim from RoboTwin-Platform/RoboTwin, MIT license.
# Revision: 13c3c47ff4312dd62484bcd51be034af55c062d1 (stable_2.0).
# Sources: envs/_base_task.py and envs/utils/action.py.
# These methods reproduce the upstream failure without CUDA/SAPIEN imports.
from __future__ import annotations
import numpy as np
import transforms3d as t3d
from copy import deepcopy


def _tolist(value):
    return np.asarray(value).tolist()


class ArmTag:
    def __init__(self, arm):
        self.arm = str(arm)
    def __str__(self):
        return self.arm
    def __eq__(self, other):
        return self.arm == str(other)


GRASP_DIRECTION_DIC = {
    name: [1., 0., 0., 0.] for name in
    ("top_down_little_left", "top_down_little_right", "side")
}


def cal_quat_dis(a, b):
    # The production orientation scorer is irrelevant to the all-failed path.
    return abs(1 - abs(np.dot(a, b)))


class Action:
    arm_tag: ArmTag
    action: Literal["move", "gripper"]
    target_pose: list = None
    target_gripper_pos: float = None

    def __init__(
        self,
        arm_tag: ArmTag | Literal["left", "right"],
        action: Literal["move", "open", "close", "gripper"],
        target_pose: sapien.Pose | list | np.ndarray = None,
        target_gripper_pos: float = None,
        **args,
    ):
        self.arm_tag = ArmTag(arm_tag)
        if action != "move":
            if action == "open":
                self.action = "gripper"
                self.target_gripper_pos = (target_gripper_pos if target_gripper_pos is not None else 1.0)
            elif action == "close":
                self.action = "gripper"
                self.target_gripper_pos = (target_gripper_pos if target_gripper_pos is not None else 0.0)
            elif action == "gripper":
                self.action = "gripper"
            else:
                raise ValueError(f"Invalid action: {action}. Must be 'open' or 'close'.")
            assert (self.target_gripper_pos is not None), "target_gripper_pos cannot be None for gripper action."
        else:
            self.action = "move"
            assert (target_pose is not None), "target_pose cannot be None for move action."
            self.target_pose = _tolist(target_pose)
        self.args = args

    def __str__(self):
        result = f"{self.arm_tag}: {self.action}"
        if self.action == "move":
            result += f"({self.target_pose})"
        else:
            result += f"({self.target_gripper_pos})"
        if self.args:
            result += f"    {self.args}"
        return result


class OfficialBaseTask:
    def choose_best_pose(self, res_pose, center_pose, arm_tag: ArmTag = None):
        """
        Choose the best pose from the list of target poses.
        - target_lst: List of target poses.
        """
        if not self.plan_success:
            return [-1, -1, -1, -1, -1, -1, -1]
        if arm_tag == "left":
            plan_multi_pose = self.robot.left_plan_multi_path
        elif arm_tag == "right":
            plan_multi_pose = self.robot.right_plan_multi_path
        target_lst = self.robot.create_target_pose_list(res_pose, center_pose, arm_tag)
        pose_num = len(target_lst)
        traj_lst = plan_multi_pose(target_lst)
        now_pose = None
        now_step = -1
        for i in range(pose_num):
            if traj_lst["status"][i] != "Success":
                continue
            if now_pose is None or len(traj_lst["position"][i]) < now_step:
                now_pose = target_lst[i]
        return now_pose

    def get_grasp_pose(
        self,
        actor: Actor,
        arm_tag: ArmTag,
        contact_point_id: int = 0,
        pre_dis: float = 0.0,
    ) -> list:
        """
        Obtain the grasp pose through the marked grasp point.
        - actor: The instance of the object to be grasped.
        - arm_tag: The arm to be used, either "left" or "right".
        - pre_dis: The distance in front of the grasp point.
        - contact_point_id: The index of the grasp point.
        """
        if not self.plan_success:
            return [-1, -1, -1, -1, -1, -1, -1]

        contact_matrix = actor.get_contact_point(contact_point_id, "matrix")
        if contact_matrix is None:
            return None
        global_contact_pose_matrix = contact_matrix @ np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0],
                                                                [0, 0, 0, 1]])
        global_contact_pose_matrix_q = global_contact_pose_matrix[:3, :3]
        global_grasp_pose_p = (global_contact_pose_matrix[:3, 3] +
                               global_contact_pose_matrix_q @ np.array([-0.12 - pre_dis, 0, 0]).T)
        global_grasp_pose_q = t3d.quaternions.mat2quat(global_contact_pose_matrix_q)
        res_pose = list(global_grasp_pose_p) + list(global_grasp_pose_q)
        res_pose = self.choose_best_pose(res_pose, actor.get_contact_point(contact_point_id, "list"), arm_tag)
        return res_pose

    def choose_grasp_pose(
        self,
        actor: Actor,
        arm_tag: ArmTag,
        pre_dis=0.1,
        target_dis=0,
        contact_point_id: list | float = None,
    ) -> list:
        """
        Test the grasp pose function.
        - actor: The actor to be grasped.
        - arm_tag: The arm to be used for grasping, either "left" or "right".
        - pre_dis: The distance in front of the grasp point, default is 0.1.
        """
        if not self.plan_success:
            return
        res_pre_top_down_pose = None
        res_top_down_pose = None
        dis_top_down = 1e9
        res_pre_side_pose = None
        res_side_pose = None
        dis_side = 1e9
        res_pre_pose = None
        res_pose = None
        dis = 1e9

        pref_direction = self.robot.get_grasp_perfect_direction(arm_tag)

        def get_grasp_pose(pre_grasp_pose, pre_grasp_dis):
            grasp_pose = deepcopy(pre_grasp_pose)
            grasp_pose = np.array(grasp_pose)
            direction_mat = t3d.quaternions.quat2mat(grasp_pose[-4:])
            grasp_pose[:3] += [pre_grasp_dis, 0, 0] @ np.linalg.inv(direction_mat)
            grasp_pose = grasp_pose.tolist()
            return grasp_pose

        def check_pose(pre_pose, pose, arm_tag):
            if arm_tag == "left":
                plan_func = self.robot.left_plan_path
            else:
                plan_func = self.robot.right_plan_path
            pre_path = plan_func(pre_pose)
            if pre_path["status"] != "Success":
                return False
            pre_qpos = pre_path["position"][-1]
            return plan_func(pose)["status"] == "Success"

        if contact_point_id is not None:
            if type(contact_point_id) != list:
                contact_point_id = [contact_point_id]
            contact_point_id = [(i, None) for i in contact_point_id]
        else:
            contact_point_id = actor.iter_contact_points()

        for i, _ in contact_point_id:
            pre_pose = self.get_grasp_pose(actor, arm_tag, contact_point_id=i, pre_dis=pre_dis)
            if pre_pose is None:
                continue
            pose = get_grasp_pose(pre_pose, pre_dis - target_dis)
            now_dis_top_down = cal_quat_dis(
                pose[-4:],
                GRASP_DIRECTION_DIC[("top_down_little_left" if arm_tag == "right" else "top_down_little_right")],
            )
            now_dis_side = cal_quat_dis(pose[-4:], GRASP_DIRECTION_DIC[pref_direction])

            if res_pre_top_down_pose is None or now_dis_top_down < dis_top_down:
                res_pre_top_down_pose = pre_pose
                res_top_down_pose = pose
                dis_top_down = now_dis_top_down

            if res_pre_side_pose is None or now_dis_side < dis_side:
                res_pre_side_pose = pre_pose
                res_side_pose = pose
                dis_side = now_dis_side

            now_dis = 0.7 * now_dis_top_down + 0.3 * now_dis_side
            if res_pre_pose is None or now_dis < dis:
                res_pre_pose = pre_pose
                res_pose = pose
                dis = now_dis

        if dis_top_down < 0.15:
            return res_pre_top_down_pose, res_top_down_pose
        if dis_side < 0.15:
            return res_pre_side_pose, res_side_pose
        return res_pre_pose, res_pose

    def grasp_actor(
        self,
        actor: Actor,
        arm_tag: ArmTag,
        pre_grasp_dis=0.1,
        grasp_dis=0,
        gripper_pos=0.0,
        contact_point_id: list | float = None,
    ):
        if not self.plan_success:
            return None, []
        if self.need_plan == False:
            if pre_grasp_dis == grasp_dis:
                return arm_tag, [
                    Action(arm_tag, "move", target_pose=[0, 0, 0, 0, 0, 0, 0]),
                    Action(arm_tag, "close", target_gripper_pos=gripper_pos),
                ]
            else:
                return arm_tag, [
                    Action(arm_tag, "move", target_pose=[0, 0, 0, 0, 0, 0, 0]),
                    Action(
                        arm_tag,
                        "move",
                        target_pose=[0, 0, 0, 0, 0, 0, 0],
                        constraint_pose=[1, 1, 1, 0, 0, 0],
                    ),
                    Action(arm_tag, "close", target_gripper_pos=gripper_pos),
                ]

        pre_grasp_pose, grasp_pose = self.choose_grasp_pose(
            actor,
            arm_tag=arm_tag,
            pre_dis=pre_grasp_dis,
            target_dis=grasp_dis,
            contact_point_id=contact_point_id,
        )
        if pre_grasp_pose == grasp_pose:
            return arm_tag, [
                Action(arm_tag, "move", target_pose=pre_grasp_pose),
                Action(arm_tag, "close", target_gripper_pos=gripper_pos),
            ]
        else:
            return arm_tag, [
                Action(arm_tag, "move", target_pose=pre_grasp_pose),
                Action(
                    arm_tag,
                    "move",
                    target_pose=grasp_pose,
                    constraint_pose=[1, 1, 1, 0, 0, 0],
                ),
                Action(arm_tag, "close", target_gripper_pos=gripper_pos),
            ]

    def get_place_pose(
        self,
        actor: Actor,
        arm_tag: ArmTag,
        target_pose: list | np.ndarray,
        constrain: Literal["free", "align", "auto"] = "auto",
        align_axis: list[np.ndarray] | np.ndarray | list = None,
        actor_axis: np.ndarray | list = [1, 0, 0],
        actor_axis_type: Literal["actor", "world"] = "actor",
        functional_point_id: int = None,
        pre_dis: float = 0.1,
        pre_dis_axis: Literal["grasp", "fp"] | np.ndarray | list = "grasp",
    ):

        if not self.plan_success:
            return [-1, -1, -1, -1, -1, -1, -1]

        actor_matrix = actor.get_pose().to_transformation_matrix()
        if functional_point_id is not None:
            place_start_pose = actor.get_functional_point(functional_point_id, "pose")
            z_transform = False
        else:
            place_start_pose = actor.get_pose()
            z_transform = True

        end_effector_pose = (self.robot.get_left_ee_pose() if arm_tag == "left" else self.robot.get_right_ee_pose())

        if constrain == "auto":
            grasp_direct_vec = place_start_pose.p - end_effector_pose[:3]
            if np.abs(np.dot(grasp_direct_vec, [0, 0, 1])) <= 0.1:
                place_pose = get_place_pose(
                    place_start_pose,
                    target_pose,
                    constrain="align",
                    actor_axis=grasp_direct_vec,
                    actor_axis_type="world",
                    align_axis=[1, 1, 0] if arm_tag == "left" else [-1, 1, 0],
                    z_transform=z_transform,
                )
            else:
                camera_vec = transforms._toPose(end_effector_pose).to_transformation_matrix()[:3, 2]
                place_pose = get_place_pose(
                    place_start_pose,
                    target_pose,
                    constrain="align",
                    actor_axis=camera_vec,
                    actor_axis_type="world",
                    align_axis=[0, 1, 0],
                    z_transform=z_transform,
                )
        else:
            place_pose = get_place_pose(
                place_start_pose,
                target_pose,
                constrain=constrain,
                actor_axis=actor_axis,
                actor_axis_type=actor_axis_type,
                align_axis=align_axis,
                z_transform=z_transform,
            )
        start2target = (transforms._toPose(place_pose).to_transformation_matrix()[:3, :3]
                        @ place_start_pose.to_transformation_matrix()[:3, :3].T)
        target_point = (start2target @ (actor_matrix[:3, 3] - place_start_pose.p).reshape(3, 1)).reshape(3) + np.array(
            place_pose[:3])

        ee_pose_matrix = t3d.quaternions.quat2mat(end_effector_pose[-4:])
        target_grasp_matrix = start2target @ ee_pose_matrix

        res_matrix = np.eye(4)
        res_matrix[:3, 3] = actor_matrix[:3, 3] - end_effector_pose[:3]
        res_matrix[:3, 3] = np.linalg.inv(ee_pose_matrix) @ res_matrix[:3, 3]
        target_grasp_qpose = t3d.quaternions.mat2quat(target_grasp_matrix)

        grasp_bias = target_grasp_matrix @ res_matrix[:3, 3]
        if pre_dis_axis == "grasp":
            target_dis_vec = target_grasp_matrix @ res_matrix[:3, 3]
            target_dis_vec /= np.linalg.norm(target_dis_vec)
        else:
            target_pose_mat = transforms._toPose(target_pose).to_transformation_matrix()
            if pre_dis_axis == "fp":
                pre_dis_axis = [0.0, 0.0, 1.0]
            pre_dis_axis = np.array(pre_dis_axis)
            pre_dis_axis /= np.linalg.norm(pre_dis_axis)
            target_dis_vec = (target_pose_mat[:3, :3] @ np.array(pre_dis_axis).reshape(3, 1)).reshape(3)
            target_dis_vec /= np.linalg.norm(target_dis_vec)
        res_pose = (target_point - grasp_bias - pre_dis * target_dis_vec).tolist() + target_grasp_qpose.tolist()
        return res_pose

    def place_actor(
        self,
        actor: Actor,
        arm_tag: ArmTag,
        target_pose: list | np.ndarray,
        functional_point_id: int = None,
        pre_dis: float = 0.1,
        dis: float = 0.02,
        is_open: bool = True,
        **args,
    ):
        if not self.plan_success:
            return None, []
        if self.need_plan:
            place_pre_pose = self.get_place_pose(
                actor,
                arm_tag,
                target_pose,
                functional_point_id=functional_point_id,
                pre_dis=pre_dis,
                **args,
            )
            place_pose = self.get_place_pose(
                actor,
                arm_tag,
                target_pose,
                functional_point_id=functional_point_id,
                pre_dis=dis,
                **args,
            )
        else:
            place_pre_pose = [0, 0, 0, 0, 0, 0, 0]
            place_pose = [0, 0, 0, 0, 0, 0, 0]

        actions = [
            Action(arm_tag, "move", target_pose=place_pre_pose),
            Action(arm_tag, "move", target_pose=place_pose),
        ]
        if is_open:
            actions.append(Action(arm_tag, "open", target_gripper_pos=1.0))
        return arm_tag, actions


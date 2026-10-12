"""RoboTwin 2.0 integration. Conditions applied before stability/expert checks."""
from __future__ import annotations

import copy
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import yaml

from .conditions import ROOT, stable_seed
from .hooks import construction
from .rng import EnvironmentRNG


def prepare_imports():
    repo = ROOT / "vendor/RoboTwin"
    os.chdir(repo)  # Upstream uses relative asset paths throughout.
    for path in (repo, repo / "policy", repo / "description/utils"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def environment_config(task, protocol, viewer=False):
    repo = ROOT / "vendor/RoboTwin"
    randomized = protocol in ("official_random", "physical_random")
    config = "demo_randomized" if randomized else "demo_clean"
    args = yaml.safe_load((repo / f"task_config/{config}.yml").read_text())
    embodiment_path = repo / "assets/embodiments/aloha-agilex"
    embodiment = yaml.safe_load((embodiment_path / "config.yml").read_text())
    camera = yaml.safe_load((repo / "task_config/_camera_config.yml").read_text())[args["camera"]["head_camera_type"]]
    args.update(task_name=task, task_config=config, eval_mode=True, render_freq=1 if viewer else 0,
                save_data=False, collect_data=False, eval_video_log=False, eval_video_save_dir=None,
                left_robot_file=str(embodiment_path), right_robot_file=str(embodiment_path),
                left_embodiment_config=copy.deepcopy(embodiment), right_embodiment_config=copy.deepcopy(embodiment),
                dual_arm_embodied=True, head_camera_h=camera["h"], head_camera_w=camera["w"])
    return args


def _pose(pose):
    return {"p": np.asarray(pose.p).tolist(), "q": np.asarray(pose.q).tolist()}


class PhysicsMixin:
    def configure(self, condition, live=None, recorder=None):
        self.condition = copy.deepcopy(condition)
        self.env_rng = EnvironmentRNG(condition["episode_seed"])
        self.live, self.recorder = live, recorder
        self.phase = "initialization"
        self.phys_bodies = []
        self.applied = {}
        self._physics_applied = False
        self._last_frame_step = None
        self._simulation_step = 0
        self._next_capture = 0

    def setup_demo(self, **kwargs):
        import torch
        # Upstream calls torch.manual_seed in setup. Isolate that from model state.
        with self.env_rng.use(), torch.random.fork_rng():
            return super().setup_demo(**copy.deepcopy(kwargs))

    def setup_scene(self, **kwargs):
        super().setup_scene(**kwargs)
        self._nominal_ambient = np.asarray(self.scene.ambient_light).copy()
        self._original_scene_step = self.scene.step
        self.scene.step = self._step_with_force

    def _update_render(self):
        # Original crazy lighting consumes global RNG on every observation/render
        # request. Index it by simulated time so policy observation count cannot
        # change conditions. Restore the caller RNG after the upstream update.
        state = np.random.get_state()
        try:
            np.random.seed(stable_seed("visual-light-v1", self.condition["episode_seed"], self._simulation_step))
            if self.crazy_random_light:
                self.scene.set_ambient_light(self._nominal_ambient)
            super()._update_render()
        finally:
            np.random.set_state(state)

    def _step_with_force(self):
        if self.phase == "policy":
            for body in self.phys_bodies:
                body.add_force_torque(np.array([self.condition["force_mg"] * body.mass * 9.81, 0, 0]), np.zeros(3))
        self._original_scene_step()
        self._simulation_step += 1
        # Capture at 30 simulated Hz during rollouts, including the expert demo.
        # No _update_render call here: it advances upstream random lighting.
        if self.phase in ("expert", "policy") and (self.live or self.recorder):
            if self._simulation_step >= self._next_capture:
                if self._next_capture == 0:
                    self._next_capture = self._simulation_step
                self._next_capture += 250 / 30
                self.capture()

    def create_table_and_wall(self, *args, **kwargs):
        height = self.condition["table_height_offset_m"]
        if height is not None:
            self.table_z_bias = height
        super().create_table_and_wall(*args, **kwargs)
        import sapien
        import transforms3d as t3d
        angle = np.deg2rad(self.condition["table_tilt_deg"])
        self.table.set_pose(sapien.Pose(self.table.get_pose().p, t3d.quaternions.axangle2quat([1, 0, 0], -angle)))

    def load_robot(self, **kwargs):
        for side in ("left", "right"):
            c = kwargs[f"{side}_embodiment_config"] = copy.deepcopy(kwargs[f"{side}_embodiment_config"])
            c["joint_damping"] = c.get("joint_damping", 200) * self.condition["arm_damping_scale"]
            c["gripper_damping"] = c.get("gripper_damping", 200) * self.condition["gripper_damping_scale"]
        super().load_robot(**kwargs)

    def load_actors(self):
        with construction(self):
            return super().load_actors()

    def get_cluttered_table(self, *args, **kwargs):
        with construction(self):
            return super().get_cluttered_table(*args, **kwargs)

    def load_camera(self, **kwargs):
        super().load_camera(**kwargs)
        import sapien
        import transforms3d as t3d
        params = self.condition["camera"]
        for camera, name in zip(self.cameras.static_camera_list, self.cameras.static_camera_name):
            if name != "head_camera":
                continue
            pose = camera.get_pose()
            p = np.array(pose.p, dtype=float)
            rot = t3d.quaternions.quat2mat(pose.q)
            direction = np.array(params["direction"])
            if params["mode"] == "translation":
                p += params["translation_m"] * direction
            elif params["mode"] == "rotation":
                rot = t3d.axangles.axangle2mat(direction, np.deg2rad(params["rotation_deg"])) @ rot
            elif params["mode"] == "orbit":
                # Fixed gaze point: nominal optical ray's intersection with tabletop.
                distance = (0.74 + self.table_z_bias - p[2]) / rot[2, 0]
                target = p + rot[:, 0] * distance
                radius = np.linalg.norm(p-target)
                axis = np.cross(p-target, direction)
                axis /= np.linalg.norm(axis)
                angle = 2 * np.arcsin(np.clip(params["orbit_m"] / (2*radius), 0, 1))
                p = target + t3d.axangles.axangle2mat(axis, angle) @ (p-target)
                forward = target - p
                forward /= np.linalg.norm(forward)
                left = np.cross([0., 0., 1.], forward)
                left /= np.linalg.norm(left)
                rot = np.column_stack((forward, left, np.cross(forward, left)))
            camera.set_pose(sapien.Pose(p, t3d.quaternions.mat2quat(rot)))

    def check_stable(self):
        self._apply_physics()
        return super().check_stable()

    def _apply_physics(self):
        if self._physics_applied:
            raise RuntimeError("Physics must be applied exactly once per scene")
        import sapien
        p = sapien.physx
        condition = self.condition
        # pybind may return a new Python wrapper for the same native component.
        # Python id() is therefore not a native simulator identity.
        robot_links = {link.entity.per_scene_id for a in (self.robot.left_entity, self.robot.right_entity) for link in a.get_links()}
        self.applied = {"bodies": [], "object_joints": [], "robot_joints": [], "robot_bodies": []}
        for entity in self.scene.entities:
            for body in entity.components:
                if not isinstance(body, p.PhysxRigidBaseComponent):
                    continue
                is_robot = entity.per_scene_id in robot_links
                dynamic = isinstance(body, p.PhysxRigidBodyComponent)
                is_object = not is_robot and entity.name not in ("wall", "table", "ground")
                if is_object and dynamic:
                    old_mass, old_inertia = body.mass, np.asarray(body.inertia).copy()
                    body.mass = old_mass * condition["mass_scale"]
                    body.inertia = old_inertia * condition["mass_scale"]
                    com = body.cmass_local_pose
                    body.cmass_local_pose = sapien.Pose(np.asarray(com.p) + condition["com_offset_m"], com.q)
                    if not getattr(body, "kinematic", False):
                        self.phys_bodies.append(body)
                if is_robot and dynamic:
                    self.applied["robot_bodies"].append({"name": entity.name, "mass": body.mass,
                        "inertia": np.asarray(body.inertia).tolist(), "com": _pose(body.cmass_local_pose)})
                # Contact material also covers table and grippers. Preserve unspecified fields.
                for shape in body.collision_shapes:
                    material = shape.physical_material
                    mu, rest = condition["friction"], condition["restitution"]
                    if mu is not None or rest is not None:
                        shape.physical_material = self.scene.create_physical_material(
                            material.static_friction if mu is None else mu,
                            material.dynamic_friction if mu is None else mu,
                            material.restitution if rest is None else rest)
                if is_object and dynamic:
                    self.applied["bodies"].append({"name": entity.name, "mass": body.mass,
                        "inertia": np.asarray(body.inertia).tolist(), "com": _pose(body.cmass_local_pose),
                        "materials": [{"static_friction": s.physical_material.static_friction,
                                       "dynamic_friction": s.physical_material.dynamic_friction,
                                       "restitution": s.physical_material.restitution} for s in body.collision_shapes]})
        seen = set()
        for articulation in self.scene.get_all_articulations():
            is_robot = any(link.entity.per_scene_id in robot_links for link in articulation.get_links())
            for joint in articulation.get_active_joints():
                key = (articulation.get_links()[0].entity.per_scene_id, joint.name)
                if key in seen:
                    continue
                seen.add(key)
                if not is_robot and condition["object_joint_damping"] is not None:
                    joint.set_drive_property(joint.stiffness, condition["object_joint_damping"], joint.force_limit, joint.drive_mode)
                self.applied["robot_joints" if is_robot else "object_joints"].append(
                    {"name": joint.name, "stiffness": joint.stiffness, "damping": joint.damping})
        self._physics_applied = True

    def initial_state(self):
        state = {"entities": [], "articulations": [], "cameras": [], "physical": self.applied,
                 "table_height_m": 0.74 + self.table_z_bias,
                 "textures": {"wall": self.wall_texture, "table": self.table_texture},
                 "lighting": {"ambient": np.asarray(self.scene.ambient_light).tolist(),
                    "points": [np.asarray(x.color).tolist() for x in self.point_light_lst],
                    "directional": [np.asarray(x.color).tolist() for x in self.direction_light_lst]}}
        for entity in self.scene.get_all_actors():
            import sapien
            body = entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            state["entities"].append({"name": entity.name, "pose": _pose(entity.pose),
                "linear_velocity": np.asarray(body.linear_velocity).tolist() if body is not None else None,
                "angular_velocity": np.asarray(body.angular_velocity).tolist() if body is not None else None})
        for articulation in self.scene.get_all_articulations():
            state["articulations"].append({"name": articulation.name, "pose": _pose(articulation.pose),
                "qpos": np.asarray(articulation.qpos).tolist(), "qvel": np.asarray(articulation.qvel).tolist()})
        for camera, name in zip(self.cameras.static_camera_list, self.cameras.static_camera_name):
            state["cameras"].append({"name": name, "pose": _pose(camera.get_pose())})
        return state

    def capture(self):
        self.cameras.update_wrist_camera(self.robot.left_camera.get_pose(), self.robot.right_camera.get_pose())
        self.scene.update_render()
        camera = self.cameras.static_camera_list[self.cameras.head_camera_id]
        camera.take_picture()
        rgb = (camera.get_picture("Color")[..., :3] * 255).clip(0, 255).astype(np.uint8)
        if self.recorder:
            self.recorder.write(rgb)
        if self.live:
            self.live.update(rgb, {"task": self.condition["task"], "seed": self.condition["episode_seed"],
                "protocol": self.condition["protocol"], "stage": self.phase, "action_step": self.take_action_cnt,
                "simulated_seconds": round(self._simulation_step / 250, 3)})
        return rgb

    def get_obs(self):
        with self.env_rng.use():
            return super().get_obs()

    def take_action(self, action, action_type="qpos"):
        with self.env_rng.use():
            return super().take_action(action, action_type)

    def play_once(self):
        import torch
        with self.env_rng.use(), torch.random.fork_rng():
            torch.manual_seed(self.condition["episode_seed"])
            return super().play_once()

    def dispose(self):
        self.phase = "closed"
        if hasattr(self, "viewer"):
            self.viewer.close()
        if hasattr(self, "scene") and hasattr(self, "_original_scene_step"):
            self.scene.step = self._original_scene_step
        if hasattr(self, "robot"):
            for name in ("left_proc", "right_proc"):
                proc = getattr(self.robot, name, None)
                if proc is not None:
                    proc.terminate()
                    proc.join(timeout=5)


def create_env(condition, viewer=False, live=None, recorder=None):
    prepare_imports()
    cls = getattr(importlib.import_module("envs." + condition["task"]), condition["task"])
    env = type("Physical_" + condition["task"], (PhysicsMixin, cls), {})()
    env.configure(condition, live=live, recorder=recorder)
    try:
        env.setup_demo(seed=condition["episode_seed"], now_ep_num=0, is_test=True,
                       **environment_config(condition["task"], condition["protocol"], viewer))
    except BaseException:
        env.dispose()
        raise
    return env


def assert_same_state(expected, actual, path="state"):
    """Floating tolerances admit serialization noise, not different episodes."""
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise ValueError(f"Initial scene keys differ: {path}")
        for k in expected:
            assert_same_state(expected[k], actual[k], path + "." + k)
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            raise ValueError(f"Initial scene length differs: {path}")
        for i, (a, b) in enumerate(zip(expected, actual)):
            assert_same_state(a, b, f"{path}[{i}]")
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if not np.isclose(expected, actual, atol=2e-6, rtol=1e-6):
            raise ValueError(f"Initial scene differs at {path}: {expected} != {actual}")
    elif expected != actual:
        raise ValueError(f"Initial scene differs at {path}: {expected} != {actual}")

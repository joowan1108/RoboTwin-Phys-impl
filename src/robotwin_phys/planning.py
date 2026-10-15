"""Recognize an exhausted official grasp search without hiding runtime errors."""
import traceback

import numpy as np

from .numerics import DegenerateDirectionError


class ExpertPlanningFailure(RuntimeError):
    """Expert geometry or a valid grasp search cannot produce a usable action."""

    def __init__(self, actor, arm_tag, contact_point_ids, kind="no_reachable_grasp", details=None):
        self.diagnostics = {"kind": kind, "actor": None if actor is None else actor.get_name(),
                            "arm": None if arm_tag is None else str(arm_tag),
                            "contact_point_ids": np.asarray(contact_point_ids).tolist()}
        if actor is not None:
            pose = actor.get_pose()
            self.diagnostics["actor_pose"] = {"p": np.asarray(pose.p).tolist(),
                                              "q": np.asarray(pose.q).tolist()}
        super().__init__(details or f"No reachable grasp for {self.diagnostics['actor']} "
                         f"with {arm_tag} arm after checking contacts {contact_point_ids}")


class PlanningMixin:
    def play_once(self):
        try:
            return super().play_once()
        except ExpertPlanningFailure:
            raise
        except Exception as exc:
            # Official move_to_pose records a failed CuRobo path only as
            # plan_success = False and the expert script keeps running. Later
            # helpers then return (None, []) and some official tasks crash on
            # that value (put_bottles_dustbin indexes grasp_actor(...)[1][0]).
            # The official evaluation loop skips such seeds. Report the crash
            # as the planning failure it follows; with plan_success still True
            # the exception is a runtime error and propagates unchanged.
            if self.plan_success is not False:
                raise
            failure = ExpertPlanningFailure(
                None, None, [], kind="exception_after_planning_failure",
                details=f"{type(exc).__name__} after the official planner failed: {exc}")
            failure.diagnostics["exception_type"] = type(exc).__name__
            failure.diagnostics["exception_message"] = str(exc)
            failure.diagnostics["traceback"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            raise failure from exc

    def get_place_pose(self, actor, arm_tag, target_pose, *args, **kwargs):
        # Missing functional metadata must remain an asset/configuration error.
        if self.plan_success:
            from envs.utils.transforms import _toPose
            for label, point in (("actor", actor.get_pose()), ("target", _toPose(target_pose))):
                if not np.isfinite(np.r_[point.p, point.q]).all() or np.linalg.norm(point.q) < 1e-12:
                    raise ValueError(f"Invalid {label} pose for {actor.get_name()}")
            end_effector = (self.robot.get_left_ee_pose() if str(arm_tag) == "left" else
                            self.robot.get_right_ee_pose())
            if (np.asarray(end_effector).shape != (7,) or not np.isfinite(end_effector).all() or
                    np.linalg.norm(end_effector[3:]) < 1e-12):
                raise ValueError(f"Invalid end effector pose for {arm_tag} arm")
            functional_id = kwargs.get("functional_point_id", args[4] if len(args) > 4 else None)
            if functional_id is not None:
                point = actor.get_functional_point(functional_id, "pose")
                if point is None:
                    raise ValueError(f"Missing functional point {functional_id} for {actor.get_name()}")
                if not np.isfinite(np.r_[point.p, point.q]).all() or np.linalg.norm(point.q) < 1e-12:
                    raise ValueError(f"Invalid functional point {functional_id} for {actor.get_name()}")
        try:
            with np.errstate(invalid="raise", divide="raise", over="raise"):
                pose = super().get_place_pose(actor, arm_tag, target_pose, *args, **kwargs)
        except (DegenerateDirectionError, FloatingPointError) as exc:
            self.plan_success = False
            raise ExpertPlanningFailure(actor, arm_tag, [], kind="undefined_placement_direction",
                                        details=str(exc)) from exc
        if pose is None or np.asarray(pose).shape != (7,) or not np.isfinite(pose).all():
            raise ValueError(f"Invalid placement pose for {actor.get_name()}")
        return pose

    @staticmethod
    def _validate_grasp_contact(actor, index, arm_tag=None):
        matrix = actor.get_contact_point(index, "matrix")
        if matrix is None:
            # Official tasks sometimes select an index unavailable in a valid
            # asset variant (e.g. left contact 2 on two-contact cup variants).
            # Preserve that expert choice and reject the candidate, without
            # inventing a contact or masking missing/corrupt metadata.
            config = getattr(actor, "config", None)
            key = getattr(actor, "POINTS", {}).get("contact")
            points = config.get(key) if isinstance(config, dict) else None
            if (isinstance(points, list) and isinstance(index, (int, np.integer))
                    and not isinstance(index, bool) and index >= len(points)):
                for point in points:
                    data = np.asarray(point, dtype=float)
                    if data.shape != (4, 4) or not np.isfinite(data).all():
                        raise ValueError(f"Invalid grasp annotation list for {actor.get_name()}")
                failure = ExpertPlanningFailure(
                    actor, arm_tag, [index], kind="unavailable_grasp_annotation",
                    details=f"Expert requested grasp contact {index} for {actor.get_name()}, "
                            f"but the asset has {len(points)} annotated contacts")
                failure.diagnostics["available_contact_point_ids"] = list(range(len(points)))
                raise failure
            raise ValueError(f"Missing grasp contact point {index} for {actor.get_name()}")
        matrix = np.asarray(matrix)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid grasp contact point {index} for {actor.get_name()}")

    def get_grasp_pose(self, actor, arm_tag, contact_point_id=0, pre_dis=0.0):
        if self.plan_success:
            self._validate_grasp_contact(actor, contact_point_id, arm_tag)
        pose = super().get_grasp_pose(actor, arm_tag, contact_point_id, pre_dis)
        # Some official tasks query one contact directly. During a multi-contact
        # search, None must remain available so the next contact can be tried.
        if pose is None and not getattr(self, "_grasp_search_depth", 0):
            self.plan_success = False
            raise ExpertPlanningFailure(actor, arm_tag, [contact_point_id])
        return pose

    def choose_grasp_pose(self, actor, arm_tag, pre_dis=0.1, target_dis=0,
                          contact_point_id=None):
        # Preserve upstream's short circuit after an earlier failed action.
        if not self.plan_success:
            return super().choose_grasp_pose(actor, arm_tag, pre_dis, target_dis, contact_point_id)
        if contact_point_id is None:
            contacts = [i for i, _ in actor.iter_contact_points()]
        else:
            contacts = contact_point_id if isinstance(contact_point_id, list) else [contact_point_id]
        if not contacts:
            # Official rigid/articulation assets can explicitly have no grasp
            # annotations (all five 086_woodenblock variants do). That gives
            # the scripted expert zero candidates; it is not a missing file.
            config = getattr(actor, "config", None)
            key = getattr(actor, "POINTS", {}).get("contact")
            if (contact_point_id is None and isinstance(config, dict) and key in config
                    and isinstance(config[key], list) and len(config[key]) == 0):
                self.plan_success = False
                raise ExpertPlanningFailure(actor, arm_tag, [], kind="no_grasp_annotations",
                                            details=f"No annotated grasp candidates for {actor.get_name()}")
            raise ValueError(f"Missing grasp contact points for {actor.get_name()}")
        # Official Actor.get_point can silently return None for missing metadata.
        # Only classify an exhausted search as infeasible when its inputs exist.
        for index in contacts:
            self._validate_grasp_contact(actor, index, arm_tag)
        # Keep the official search and its candidate selection unchanged.
        depth = getattr(self, "_grasp_search_depth", 0)
        self._grasp_search_depth = depth + 1
        try:
            result = super().choose_grasp_pose(actor, arm_tag, pre_dis, target_dis, contact_point_id)
        finally:
            self._grasp_search_depth = depth
        if result[0] is None and result[1] is None:
            self.plan_success = False
            raise ExpertPlanningFailure(actor, arm_tag, contacts)
        if result[0] is None or result[1] is None:
            raise RuntimeError("Grasp search returned an incomplete pose pair")
        return result

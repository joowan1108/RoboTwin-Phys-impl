"""Attributes that an official check_success reads but only the expert's play_once assigns.

The official evaluation loop keeps one environment object alive: it runs the
scripted expert (play_once) as its feasibility check and then evaluates the
policy on the same object, so attributes assigned in play_once still exist when
check_success runs during the policy rollout. Every episode here starts a fresh
process without the expert, so the three tasks whose check_success depends on
such attributes get them from the initial scene, using the official play_once
expressions verbatim. The expert's recorded info ("{a}" = arm) is used as a
cross-check where the task records it.

Scan of the pinned official tasks (attributes read in check_success but
assigned only in play_once): open_laptop (arm_tag), place_object_scale
(arm_tag), put_object_cabinet (arm_tag, origin_z). No other task is affected.
"""
TASKS = ("open_laptop", "place_object_scale", "put_object_cabinet")


def prepare(env, task, expert_info=None):
    """Set the attributes for `task` on `env` and return what was set."""
    if task not in TASKS:
        return {}
    from envs.utils import ArmTag
    context = {}
    if task == "open_laptop":
        from envs.utils import get_face_prod
        face_prod = get_face_prod(env.laptop.get_pose().q, [1, 0, 0], [1, 0, 0])
        env.arm_tag = ArmTag("left" if face_prod > 0 else "right")
    else:
        env.arm_tag = ArmTag("right" if env.object.get_pose().p[0] > 0 else "left")
    context["arm_tag"] = str(env.arm_tag)
    if task == "put_object_cabinet":
        env.origin_z = float(env.object.get_pose().p[2])
        context["origin_z"] = env.origin_z
    recorded = ((expert_info or {}).get("info") or {}).get("{a}")
    if recorded is not None and str(recorded) != context["arm_tag"]:
        raise ValueError(f"Success context differs from the expert record for {task}: "
                         f"derived {context['arm_tag']}, expert used {recorded}")
    return context

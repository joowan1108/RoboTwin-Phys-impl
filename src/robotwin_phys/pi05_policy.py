"""Motus pi0.5 checkpoint's documented transforms, not RoboTwin's stock config.

Run in the Python 3.11 OpenPI environment. The HF card specifies adapt_to_pi=True,
delta joints with absolute grippers, action horizon 32, and the supplied stats.
"""
from collections import deque
import numpy as np


def get_model(args):
    from openpi.training import config as c
    from openpi.models import pi0_config
    from openpi.policies.policy_config import create_trained_policy
    from openpi.shared import normalize
    model_config = pi0_config.Pi0Config(pi05=True, action_horizon=32)
    data = c.LeRobotAlohaDataConfig(
        repo_id="robotwin_clean_randomized_joint_training",
        adapt_to_pi=True, use_delta_joint_actions=True,
        base_config=c.DataConfig(prompt_from_task=True))
    train = c.TrainConfig(name="motus_pi05_robotwin2_reproduction", model=model_config, data=data)
    stats = normalize.load(args["norm_stats_dir"])
    policy = create_trained_policy(train, args["checkpoint_dir"], norm_stats=stats,
                                    pytorch_device=args.get("device", "cuda"),
                                    sample_kwargs={"num_steps": args.get("num_inference_steps", 10)})
    return {"policy": policy, "queue": deque(), "replan_steps": args.get("replan_steps", 32)}


def reset_model(model):
    model["queue"].clear()


def eval(env, model, observation):
    if not model["queue"]:
        cameras = observation["observation"]
        obs = {"state": np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
               "images": {"cam_high": np.moveaxis(cameras["head_camera"]["rgb"], -1, 0),
                          "cam_left_wrist": np.moveaxis(cameras["left_camera"]["rgb"], -1, 0),
                          "cam_right_wrist": np.moveaxis(cameras["right_camera"]["rgb"], -1, 0)},
               "prompt": env.get_instruction()}
        actions = np.asarray(model["policy"].infer(obs)["actions"])
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError(f"Unexpected pi0.5 action shape: {actions.shape}")
        model["queue"].extend(actions[:model["replan_steps"]])
    env.take_action(model["queue"].popleft(), action_type="qpos")

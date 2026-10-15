"""One fresh process per episode prevents planner/renderer state leaking."""
import argparse
import json
from pathlib import Path
import random
import time
import traceback

from .conditions import digest, stable_seed
from .manifest import atomic_json
from .planning import ExpertPlanningFailure


def run(request):
    from .simulator import create_env, assert_same_state
    from .rendering import LiveView, Recorder
    env = recorder = live = client = None
    start = time.monotonic()
    try:
        if request.get("live"):
            live = LiveView(request.get("port", 8765))
        if request.get("video"):
            recorder = Recorder(request["video"])
        episode = request["episode"]
        try:
            env = create_env(episode["condition"], viewer=request.get("viewer", False), live=live, recorder=recorder)
        except Exception as exc:
            from envs.utils.create_actor import UnStableError
            if isinstance(exc, UnStableError) and request["mode"] in ("expert", "inspect"):
                return {"status": "rejected", "reason": "unstable_initialization", "details": str(exc)}
            raise
        initial = env.initial_state()
        if request["mode"] == "inspect":
            # A runtime audit, never an expert-verified benchmark episode.
            camera = env.cameras.static_camera_list[env.cameras.head_camera_id]
            a = camera.get_pose()
            env.crazy_random_light = True
            env._update_render()
            lights_a = [x.color.copy() for x in env.point_light_lst]
            env.get_obs(); env.get_obs()
            import numpy as np
            for expected, lamp in zip(lights_a, env.point_light_lst):
                np.testing.assert_array_equal(expected, lamp.color)
            env.capture()
            return {"status": "inspected", "initial_state": initial,
                    "extra_observations_preserve_lights": True,
                    "render_device": str(env.scene.render_system.device)}
        if request["mode"] == "expert":
            env.phase = "expert"
            if live or recorder:
                env.capture()
            try:
                info = env.play_once()
            except ExpertPlanningFailure as exc:
                return {"status": "rejected", "reason": "expert_failed", "plan_success": False,
                        "details": str(exc), "planning_failure": exc.diagnostics}
            success = bool(env.plan_success and env.check_success())
            if not success:
                return {"status": "rejected", "reason": "expert_failed", "plan_success": bool(env.plan_success)}
            from generate_episode_instructions import generate_episode_descriptions
            with env.env_rng.use():
                random.seed(stable_seed("language-v1", episode["condition"]["task"], episode["seed"]))
                descriptions = generate_episode_descriptions(episode["condition"]["task"], [info["info"]], 1)
                choices = descriptions[0][request.get("instruction_type", "unseen")]
                if not choices:
                    raise ValueError("No matching instruction for this episode")
                instruction = random.choice(choices)
            return {"status": "verified", "episode": {**episode, "expert_success": True,
                "initial_state": initial, "initial_state_sha256": digest(initial), "instruction": instruction,
                "expert_info": info, "expert_seconds": time.monotonic()-start,
                "force_during_expert": False}}
        assert_same_state(episode["initial_state"], initial)
        env.set_instruction(episode["instruction"])
        env.current_seed = env.episode_seed = episode["seed"]
        env.test_num = env.ep_num = episode["episode_index"]
        # Official check_success for three tasks reads attributes that only the
        # expert's play_once assigns; derive them from the verified initial scene.
        from .success_context import prepare
        success_context = prepare(env, episode["condition"]["task"], episode.get("expert_info"))
        if request["mode"] == "replay":
            if live or recorder:
                env.capture()
            return {"status": "matched", "initial_state_sha256": digest(initial)}
        from .rpc import PolicyClient
        client = PolicyClient(request.get("host", "127.0.0.1"), request.get("policy_port", 9000))
        if request.get("policy_identity") and client.identity != request["policy_identity"]:
            raise ValueError("Policy server identity changed during evaluation")
        client.request("reset", seed=stable_seed("policy-v1", request.get("policy_seed", 0), episode["seed"]))
        env.phase = "policy"
        if live or recorder:
            env.capture()
        start_step = env.take_action_cnt
        limit = request.get("smoke_max_actions") or env.step_lim
        while env.take_action_cnt < min(env.step_lim, limit) and not env.eval_success:
            old = env.take_action_cnt
            observation = env.get_obs()
            client.request("step", env=env, observation=observation)
            if env.take_action_cnt <= old:
                raise RuntimeError("Policy returned without taking an action")
        if live or recorder:
            env.capture()
        return {"status": "completed", "success": bool(env.eval_success), "actions": env.take_action_cnt-start_step,
                "success_context": success_context,
                "seed": episode["seed"], "episode_index": episode["episode_index"],
                "condition_sha256": episode["condition_sha256"], "initial_state_sha256": digest(initial),
                "policy": client.identity, "seconds": time.monotonic()-start,
                "benchmark_eligible": not bool(request.get("smoke_max_actions")) and client.identity["name"] != "noop_infrastructure_test"}
    finally:
        if recorder:
            recorder.close()
        if client:
            client.close()
        if env:
            env.dispose()
        if live:
            live.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("request")
    p.add_argument("result")
    args = p.parse_args()
    result_path = Path(args.result).resolve()
    request = json.loads(Path(args.request).resolve().read_text())
    try:
        result = run(request)
    except Exception:
        result = {"status": "error", "error": traceback.format_exc()}
        atomic_json(result_path, result)
        raise
    atomic_json(result_path, result)


if __name__ == "__main__":
    main()

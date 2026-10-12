from collections import deque
from pathlib import Path
import numpy as np
import yaml

from robotwin_phys import pi05_policy
from robotwin_phys.conditions import ROOT


def test_pi05_matches_documented_camera_order_and_action_queue():
    seen = []
    class FakePolicy:
        def infer(self, data):
            seen.append(data)
            assert data["images"]["cam_high"].shape == (3, 240, 320)
            assert data["images"]["cam_left_wrist"][0, 0, 0] == 2
            assert data["images"]["cam_right_wrist"][0, 0, 0] == 3
            return {"actions": np.arange(32*14).reshape(32,14)}
    class Env:
        def __init__(self): self.actions=[]
        def get_instruction(self): return "Pick the bottle"
        def take_action(self, action, action_type): self.actions.append(action)
    model={"policy":FakePolicy(),"queue":deque(),"replan_steps":2}
    obs={"joint_action":{"vector":np.zeros(14)},"observation":{
        name:{"rgb":np.full((240,320,3),i,np.uint8)} for i,name in
        enumerate(["head_camera","left_camera","right_camera"],1)}}
    env=Env()
    pi05_policy.eval(env,model,obs); pi05_policy.eval(env,model,obs)
    assert len(seen)==1 and len(env.actions)==2
    np.testing.assert_array_equal(env.actions[1], np.arange(14,28))
    pi05_policy.reset_model(model)
    assert not model["queue"]


def test_all_five_configs_reference_real_official_adapters():
    for name in ("g05","pi05","fact","motus","fastwam"):
        config=yaml.safe_load((ROOT/f"configs/policies/{name}.yaml").read_text())
        assert (ROOT/config["adapter"]).exists()
        assert len(config["checkpoint"]["revision"])==40

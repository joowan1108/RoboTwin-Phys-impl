"""Actual CPU PhysX dynamics test; exercises the production force-step hook."""
from types import SimpleNamespace
import numpy as np
import pytest

from robotwin_phys.simulator import PhysicsMixin


def test_force_is_policy_only_and_scales_as_mg():
    import sapien
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
    scene.set_timestep(1/250)
    bodies=[]
    for i, mass in enumerate((1., 3.)):
        entity=sapien.Entity()
        body=sapien.physx.PhysxRigidDynamicComponent()
        body.attach(sapien.physx.PhysxCollisionShapeBox([.01,.01,.01],sapien.physx.PhysxMaterial(.5,.5,0)))
        entity.add_component(body)
        entity.set_pose(sapien.Pose([i,0,2]))
        scene.add_entity(entity)
        body.mass=mass
        bodies.append(body)
    env=SimpleNamespace(phase="expert", phys_bodies=bodies, condition={"force_mg":.1},
        _original_scene_step=scene.step, _simulation_step=0, live=None,recorder=None)
    for _ in range(10): PhysicsMixin._step_with_force(env)
    assert all(abs(b.linear_velocity[0])<1e-7 for b in bodies)
    env.phase="policy"
    for _ in range(10): PhysicsMixin._step_with_force(env)
    for b in bodies:
        assert b.linear_velocity[0] == pytest.approx(.1*9.81*10/250, rel=1e-4)
    # Same acceleration for different masses verifies the mg normalization.
    assert bodies[0].linear_velocity[0] == pytest.approx(bodies[1].linear_velocity[0], abs=1e-7)

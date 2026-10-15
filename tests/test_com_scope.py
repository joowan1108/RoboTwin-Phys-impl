"""CoM-offset scope: manipulated objects shift, official clutter keeps its nominal centre of mass.

The production _apply_physics runs against a minimal stand-in for the SAPIEN
component types, so the scope rule is checked on machines without SAPIEN.
"""
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from robotwin_phys.simulator import PhysicsMixin, _clutter_entity_ids


class Pose:
    def __init__(self, p=(0.0, 0.0, 0.0), q=(1.0, 0.0, 0.0, 0.0)):
        self.p, self.q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)


class RigidBase:
    """Static body such as the table: a collision component without mass or CoM."""
    def __init__(self):
        self.collision_shapes = [SimpleNamespace(physical_material=SimpleNamespace(
            static_friction=0.9, dynamic_friction=0.8, restitution=0.0))]


class RigidBody(RigidBase):
    def __init__(self, mass, com, kinematic=False):
        super().__init__()
        self.mass, self.inertia = mass, np.array([1.0, 2.0, 3.0]) * mass
        self.cmass_local_pose, self.kinematic = Pose(com), kinematic
        self.collision_shapes = [SimpleNamespace(physical_material=SimpleNamespace(
            static_friction=0.5, dynamic_friction=0.4, restitution=0.1))]


class Entity:
    def __init__(self, name, scene_id, body):
        self.name, self.per_scene_id, self.components = name, scene_id, [body]


@pytest.fixture
def fake_sapien(monkeypatch):
    sapien, physx = types.ModuleType("sapien"), types.ModuleType("sapien.physx")
    physx.PhysxRigidBaseComponent, physx.PhysxRigidBodyComponent = RigidBase, RigidBody
    sapien.physx, sapien.Pose = physx, Pose
    monkeypatch.setitem(sys.modules, "sapien", sapien)
    monkeypatch.setitem(sys.modules, "sapien.physx", physx)
    return sapien


def make_env(condition, task_objects, clutter_objects, articulated_clutter=()):
    links = [SimpleNamespace(entity=SimpleNamespace(per_scene_id=900 + i)) for i in range(2)]
    robot = SimpleNamespace(left_entity=SimpleNamespace(get_links=lambda: links[:1]),
                            right_entity=SimpleNamespace(get_links=lambda: links[1:]))
    entities = [*task_objects, *clutter_objects, Entity("table", 1, RigidBase()),
                Entity("robot_link", 900, RigidBody(2.0, (0.0, 0.0, 0.0)))]
    for articulation in articulated_clutter:
        entities.extend(link.entity for link in articulation.get_links())

    def material(static, dynamic, restitution):
        return SimpleNamespace(static_friction=static, dynamic_friction=dynamic, restitution=restitution)

    scene = SimpleNamespace(entities=entities, create_physical_material=material, get_all_articulations=lambda: [])
    cluttered = [SimpleNamespace(actor=e) for e in clutter_objects] + \
                [SimpleNamespace(actor=a) for a in articulated_clutter]
    return SimpleNamespace(_physics_applied=False, condition=condition, scene=scene, robot=robot,
                           phys_bodies=[], applied={}, cluttered_objs=cluttered)


CONDITION = {"mass_scale": 2.0, "com_offset_m": [0.012, -0.009, 0.014], "friction": 0.3,
             "restitution": 0.25, "object_joint_damping": None}


def test_task_objects_shift_and_clutter_keeps_nominal_com(fake_sapien):
    bottle = Entity("001_bottle", 10, RigidBody(0.5, (0.001, 0.002, 0.03)))
    marker = Entity("marker", 11, RigidBody(0.05, (0.0, 0.0, 0.01)))
    env = make_env(CONDITION, [bottle], [marker])
    PhysicsMixin._apply_physics(env)
    assert bottle.components[0].cmass_local_pose.p == pytest.approx([0.013, -0.007, 0.044])
    assert marker.components[0].cmass_local_pose.p == pytest.approx([0.0, 0.0, 0.01])
    # Mass scaling and the policy-stage force reach task objects only; contact materials reach everything.
    assert bottle.components[0].mass == pytest.approx(1.0) and marker.components[0].mass == pytest.approx(0.05)
    assert marker.components[0].inertia == pytest.approx(np.array([1.0, 2.0, 3.0]) * 0.05)
    for entity in (bottle, marker):
        shape = entity.components[0].collision_shapes[0]
        assert (shape.physical_material.static_friction, shape.physical_material.restitution) == (0.3, 0.25)
    assert env.phys_bodies == [bottle.components[0]]
    # The table material is replaced too, as before (contact material covers table and grippers).
    table = next(e for e in env.scene.entities if e.name == "table").components[0]
    assert table.collision_shapes[0].physical_material.static_friction == 0.3
    record = {b["name"]: b for b in env.applied["bodies"]}
    assert record["001_bottle"]["clutter"] is False and record["001_bottle"]["com_offset_m"] == [0.012, -0.009, 0.014]
    assert record["marker"]["clutter"] is True and record["marker"]["com_offset_m"] == [0.0, 0.0, 0.0]
    assert record["marker"]["com"]["p"] == pytest.approx([0.0, 0.0, 0.01])
    assert record["marker"]["mass_scale"] == 1.0 and record["marker"]["external_force"] is False
    assert record["001_bottle"]["mass_scale"] == 2.0 and record["001_bottle"]["external_force"] is True


def test_articulated_clutter_links_are_recognised(fake_sapien):
    links = [SimpleNamespace(entity=Entity("drinkbox_link", 20 + i, RigidBody(0.2, (0.0, 0.0, 0.0)))) for i in range(2)]
    articulation = SimpleNamespace(get_links=lambda: links)
    cup = Entity("021_cup", 10, RigidBody(0.3, (0.0, 0.0, 0.0)))
    env = make_env(CONDITION, [cup], [], articulated_clutter=[articulation])
    PhysicsMixin._apply_physics(env)
    assert cup.components[0].cmass_local_pose.p == pytest.approx(CONDITION["com_offset_m"])
    for link in links:
        assert link.entity.components[0].cmass_local_pose.p == pytest.approx([0.0, 0.0, 0.0])
        assert link.entity.components[0].mass == pytest.approx(0.2)
    assert env.phys_bodies == [cup.components[0]]
    assert _clutter_entity_ids(env.cluttered_objs) == {20, 21}


def test_without_clutter_every_object_shifts_as_before(fake_sapien):
    hammer = Entity("020_hammer", 10, RigidBody(0.4, (0.0, 0.0, 0.0)))
    env = make_env(CONDITION, [hammer], [])
    PhysicsMixin._apply_physics(env)
    assert hammer.components[0].cmass_local_pose.p == pytest.approx(CONDITION["com_offset_m"])
    assert [b["clutter"] for b in env.applied["bodies"]] == [False]


def test_clutter_ids_accept_bare_entities_and_wrapped_actors():
    bare = SimpleNamespace(per_scene_id=5)
    wrapped = SimpleNamespace(actor=SimpleNamespace(per_scene_id=6))
    assert _clutter_entity_ids([bare, wrapped]) == {5, 6}
    assert _clutter_entity_ids([]) == set()


class OfficialLike:
    """Mimics the official get_cluttered_table: adds entities, then clears cluttered_objs."""
    def get_cluttered_table(self, count=2):
        self.cluttered_objs = []
        for i in range(count):
            entity = Entity(f"clutter{i}", 500 + i, RigidBody(0.05, (0.0, 0.0, 0.0)))
            self.scene.entities.append(entity)
            self.cluttered_objs.append(SimpleNamespace(actor=entity))
        self.cluttered_objs = []  # official code resets the list before returning


class EnvLike(PhysicsMixin, OfficialLike):
    pass


def test_override_records_clutter_ids_although_official_code_clears_the_list(fake_sapien):
    bottle = Entity("001_bottle", 10, RigidBody(0.5, (0.0, 0.0, 0.0)))
    env = make_env(CONDITION, [bottle], [])
    real = EnvLike.__new__(EnvLike)
    real.__dict__.update(env.__dict__)
    real._clutter_ids = set()
    real.get_cluttered_table()
    assert real.cluttered_objs == [] and real._clutter_ids == {500, 501}
    PhysicsMixin._apply_physics(real)
    record = {b["name"]: b for b in real.applied["bodies"]}
    assert record["001_bottle"]["clutter"] is False
    assert record["clutter0"]["clutter"] is True and record["clutter1"]["clutter"] is True
    assert record["clutter0"]["com_offset_m"] == [0.0, 0.0, 0.0]
    assert bottle.components[0].cmass_local_pose.p == pytest.approx(CONDITION["com_offset_m"])


def test_construction_hook_scales_task_objects_but_not_clutter():
    from robotwin_phys import hooks
    env = SimpleNamespace(condition={"geometry_scale": 0.9, "table_tilt_deg": 0.0}, table_xy_bias=[0.0, 0.0], table_z_bias=0.0)
    assert hooks.geometry_size(1.0) == 1.0  # outside construction: untouched
    with hooks.construction(env):
        assert hooks.geometry_size(1.0) == pytest.approx(0.9)
        assert hooks.geometry_size([2.0, 2.0, 2.0]) == pytest.approx([1.8, 1.8, 1.8])
        with hooks.construction(env, clutter=True):
            assert hooks.geometry_size(1.0) == 1.0
            assert hooks.geometry_size([2.0, 2.0, 2.0]) == [2.0, 2.0, 2.0]
        assert hooks.geometry_size(1.0) == pytest.approx(0.9)
    assert hooks.geometry_size(1.0) == 1.0


def test_clutter_poses_still_follow_the_tilted_table(fake_sapien):
    from robotwin_phys import hooks
    env = SimpleNamespace(condition={"geometry_scale": 0.9, "table_tilt_deg": 2.0}, table_xy_bias=[0.0, 0.0], table_z_bias=0.0)
    pose = Pose([0.1, 0.2, 0.80], [1.0, 0.0, 0.0, 0.0])
    with hooks.construction(env):
        task = hooks.object_pose(pose)
    with hooks.construction(env, clutter=True):
        clutter = hooks.object_pose(pose)
    # Same tilt rotation for both; only the height above the table is scaled for task objects.
    assert clutter.q == pytest.approx(task.q)
    assert clutter.p[2] > task.p[2]
    nominal = hooks.object_pose(pose)  # no construction context: identity
    assert nominal.p == pytest.approx(pose.p) and nominal.q == pytest.approx(pose.q)

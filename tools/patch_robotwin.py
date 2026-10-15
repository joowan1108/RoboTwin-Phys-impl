#!/usr/bin/env python3
"""Idempotent, guarded changes to the pinned upstream, preserved as a diff."""
from pathlib import Path
import ast
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "vendor/RoboTwin"
PIN = "13c3c47ff4312dd62484bcd51be034af55c062d1"


def patch_transform_text(text):
    marker = "# Reviewed finite-rotation helpers for RoboTwin-Phys."
    if marker in text:
        return text
    # Preserve pose placement and candidate generation; replace the shared
    # numeric operations and guard the remaining axis-angle entry points.
    old = "    v2 = v2 / np.linalg.norm(v2)\n    axis = np.cross(v1, v2)\n    angle = np.arccos(np.dot(v1, v2))"
    node = next((n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'rotate2rob'), None)
    part = ast.get_source_segment(text, node) if node is not None else ''
    if part.count(old) != 1:
        raise RuntimeError("Pinned rotate2rob implementation changed; review before patching")
    text = text.replace(part, part.replace(old, "    v2 = unit_vector(v2)\n    axis, angle = alignment_axis_angle(v1, v2)"), 1)
    anchor = "    rotate_mat = t3d.axangles.axangle2mat(axis, theta)"
    if text.count(anchor) != 1:
        raise RuntimeError("Pinned rotate_along_axis implementation changed")
    text = text.replace(anchor, "    if not np.isfinite(theta):\n"
                       "        raise ValueError('Non-finite rotation angle')\n"
                       "    axis = unit_vector(axis, 'rotation axis')\n" + anchor)
    text += "\n\n" + marker + "\nfrom robotwin_phys.numerics import (\n" \
            "    get_align_matrix, cal_quat_dis, get_product_vector, rotate_cone,\n" \
            "    alignment_axis_angle, unit_vector,\n)\n"
    return text


def patch_planner_text(text):
    marker = '# Preserve CuRobo runtime errors; a failed plan is reported by result.success.'
    if marker in text:
        return text
    old = ('            try:\n'
           '                result = self.motion_gen_batch.plan_batch(start_joint_states, goal_pose_of_ee, plan_config)\n'
           '            except Exception as e:\n'
           '                return {"status": ["Failure" for i in range(10)]}')
    if text.count(old) != 1:
        raise RuntimeError('Pinned planner exception handling changed; review before patching')
    return text.replace(old, '            ' + marker + '\n'
                        '            result = self.motion_gen_batch.plan_batch(start_joint_states, goal_pose_of_ee, plan_config)')


def main():
    sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    if sha != PIN:
        raise RuntimeError(f"RoboTwin revision changed: {sha}; review hooks before patching")
    path = REPO / "envs/utils/create_actor.py"
    text = path.read_text()
    if "from robotwin_phys.hooks import" not in text:
        text = text.replace("from .actor_utils import Actor, ArticulationActor", "from .actor_utils import Actor, ArticulationActor\nfrom robotwin_phys.hooks import geometry_size, geometry_model, object_pose")
        text = text.replace("return scene, pose\n", "return scene, object_pose(pose)\n")
        text = text.replace("return scene.scene, sapien.Pose([pose.p[0], pose.p[1], pose.p[2] + scene.table_z_bias], pose.q)",
                            "return scene.scene, object_pose(sapien.Pose([pose.p[0], pose.p[1], pose.p[2] + scene.table_z_bias], pose.q))")
        lines = text.splitlines(keepends=True)
        for node in reversed(ast.parse(text).body):
            if not isinstance(node, ast.FunctionDef):
                continue
            part = "".join(lines[node.lineno-1:node.end_lineno])
            if node.name in ("create_obj", "create_glb", "create_actor"):
                part = part.replace("    builder = scene.create_actor_builder()", "    scale, model_data = geometry_model(scale, model_data)\n    builder = scene.create_actor_builder()")
            elif node.name == "create_urdf_obj":
                part = part.replace("    loader.fix_root_link", "    scale, model_data = geometry_model(loader.scale, model_data)\n    loader.scale = scale\n    loader.fix_root_link", 1)
            elif node.name == "create_sapien_urdf_obj":
                part = part.replace("    loader.scale = scale", "    scale, model_data = geometry_model(scale, model_data)\n    loader.scale = scale", 1)
            elif node.name == "create_box":
                part = part.replace("    entity = create_entity_box(", "    half_size = geometry_size(half_size)\n    entity = create_entity_box(", 1)
            elif node.name in ("create_sphere", "create_cylinder", "create_visual_box"):
                sizes = {"create_sphere": ["radius"], "create_cylinder": ["radius", "half_length"], "create_visual_box": ["half_size"]}[node.name]
                inserted = "".join(f"    {key} = geometry_size({key})\n" for key in sizes)
                part = part.replace("    scene, pose = preprocess(scene, pose)", inserted + "    scene, pose = preprocess(scene, pose)", 1)
            lines[node.lineno-1:node.end_lineno] = [part]
        path.write_text("".join(lines))
    # Upstream has a mutable default that accumulates table bias over episodes.
    path = REPO / "envs/_base_task.py"
    text = path.read_text()
    anchor = "        self.record_cluttered_objects = []  # record cluttered objects"
    if "xlim, ylim, zlim = list(xlim)" not in text:
        text = text.replace(anchor, "        xlim, ylim, zlim = list(xlim), list(ylim), list(zlim)\n" + anchor)
    text = text.replace("        self.scene = self.engine.create_scene(scene_config)",
                        "        sapien.physx.set_scene_config(scene_config)\n"
                        "        self.scene = sapien.Scene([sapien.physx.PhysxCpuSystem(),\n"
                        "            sapien.render.RenderSystem(os.environ.get('SAPIEN_RENDER_DEVICE', 'cuda:0'))])")
    path.write_text(text)
    path = REPO / "envs/utils/transforms.py"
    path.write_text(patch_transform_text(path.read_text()))
    path = REPO / "envs/robot/planner.py"
    path.write_text(patch_planner_text(path.read_text()))
    diff = subprocess.check_output(["git", "-C", str(REPO), "diff"], text=True)
    (ROOT / "references/robotwin-local.patch").write_text(diff)
    # Official RoboTwin installation fixes, constrained to this isolated environment.
    import importlib.util
    mplib = Path(importlib.util.find_spec("mplib").origin).parent / "planner.py"
    text = mplib.read_text()
    text = text.replace("if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:",
                        "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:")
    mplib.write_text(text)
    sapien = Path(importlib.util.find_spec("sapien").origin).parent / "wrapper/urdf_loader.py"
    text = sapien.read_text().replace('urdf_file[:-4] + "srdf"', 'urdf_file[:-4] + ".srdf"')
    text = text.replace('open(urdf_file, "r")', 'open(urdf_file, "r", encoding="utf-8")')
    text = text.replace('open(srdf_file, "r")', 'open(srdf_file, "r", encoding="utf-8")')
    sapien.write_text(text)
    print("Applied reviewed factory hooks and upstream installation fixes")


if __name__ == "__main__":
    main()

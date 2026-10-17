#!/usr/bin/env python3
"""Package the success-context fix for the running v6 pool as a new immutable snapshot.

Baseline: the delivered robotwin-fresh-pool-v2.zip. The only source differences are
src/robotwin_phys/worker.py (evaluation path) and the new success_context.py; tools
gain the provenance-preserving continuation. The launcher builds/continues the pool
if needed and then evaluates FastWAM on both GPUs, exactly like the v6 "all" script.
"""
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NAME = 'robotwin-success-context-fix-v7'
PREVIOUS_DELIVERY = 'robotwin-success-context-fix-v6'
OIDN_VERSION = '2.3.3'
OIDN_URL = 'https://github.com/RenderKit/oidn/releases/download/v2.3.3/oidn-2.3.3.x86_64.linux.tar.gz'
OIDN_SHA256 = '3c385230d9e6f63527ba72f2229594dbac5051674219d72e0044b5d0b841796f'
OIDN_LOADER = 'docker/oidn/_oidn_tricks.py'
SAPIEN_PKG = '/home/guest/robotwin-workspace/.env/lib/python3.10/site-packages/sapien'
EPISODE_TIMEOUT = 3600
CONFIGS = ('policies/fastwam.yaml', 'policies/g05.yaml')  # policy profiles shipped with the snapshot (not in the pool fingerprint)
FASTWAM_WORKSPACE = '/opt/robotwin-policies/fastwam/workspace'
G05_WORKSPACE = '/opt/robotwin-policies/g05/workspace'
BASE = 'robotwin-fresh-pool-v2'
EXPERIMENT = 'phys-pro6000-100-v6'
MARKER = 'SUCCESS_CONTEXT_PATCH_SOURCE_PASS'
ALLOWED_SRC_CHANGES = {'robotwin_phys/worker.py', 'robotwin_phys/success_context.py'}


def checksums(folder, extensions):
    return {str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(folder.rglob('*')) if path.is_file() and path.suffix in extensions
            and '__pycache__' not in path.parts}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    with zipfile.ZipFile(ROOT / f'reports/{BASE}.zip') as base:
        prefix = BASE + '/'
        vendor = {name: base.read(prefix + 'vendor/' + name) for name in ('transforms.py', 'planner.py', 'ROBOTWIN_LICENSE')}
        prior = json.loads(base.read(prefix + 'manifest.json'))
    source = checksums(ROOT / 'src', ('.py',))
    configs = {rel: hashlib.sha256((ROOT / 'configs' / rel).read_bytes()).hexdigest() for rel in CONFIGS}
    loader = (ROOT / OIDN_LOADER).read_bytes()
    if b'libOpenImageDenoise.so.2.3.3' not in loader or b'.so.2.0.1' in loader:
        raise SystemExit(f'{OIDN_LOADER} must preload the OIDN {OIDN_VERSION} libraries')
    import yaml
    if yaml.safe_load((ROOT / 'configs/policies/fastwam.yaml').read_text())['args'].get('sigma_shift') != 5.0:
        raise SystemExit('configs/policies/fastwam.yaml must set args.sigma_shift: 5.0 (upstream released-checkpoint setting)')
    if yaml.safe_load((ROOT / 'configs/policies/g05.yaml').read_text())['args'].get('sim_task') != 'robotwin':
        raise SystemExit('configs/policies/g05.yaml must set args.sim_task: robotwin (upstream evaluator passes task=robotwin)')
    tools = checksums(ROOT / 'tools', ('.py', '.sh', '.json'))
    changed = {p for p in set(source) | set(prior['files']) if source.get(p) != prior['files'].get(p)}
    if changed != ALLOWED_SRC_CHANGES:
        raise SystemExit(f'Unreviewed source change relative to {BASE}: {sorted(changed)}')
    tool_text = (ROOT / 'tools/continue_success_context.py').read_text()
    for name, value in (('BEFORE_WORKER', prior['files']['robotwin_phys/worker.py']),
                        ('AFTER_WORKER', source['robotwin_phys/worker.py']),
                        ('CONTEXT_SHA256', source['robotwin_phys/success_context.py'])):
        if f"{name} = '{value}'" not in tool_text:
            raise SystemExit(f'tools/continue_success_context.py must pin {name} = {value}')
    manifest = dict(name=NAME, base=BASE, base_sha256=hashlib.sha256((ROOT / f'reports/{BASE}.zip').read_bytes()).hexdigest(),
                    files=source, tools=tools, configs=configs, source_sha256=digest(source), tools_sha256=digest(tools),
                    configs_sha256=digest(configs),
                    oidn=dict(version=OIDN_VERSION, url=OIDN_URL, tarball_sha256=OIDN_SHA256,
                              loader_sha256=hashlib.sha256(loader).hexdigest(), sapien_package=SAPIEN_PKG,
                              reason='SAPIEN 3.0.0b1 bundles OIDN 2.0.1, which fails on Blackwell (OIDN Error: invalid handle); '
                                     'policy observations were un-denoised ray-traced frames (job 1685/1686)'),
                    episode_timeout_seconds=EPISODE_TIMEOUT,
                    g05_runtime_fix='tools/prepare_g05.py hard-links (or copies) the G0.5 checkpoint and dataset stats into '
                                    'checkpoints/g05-runtime instead of symlinking: the pinned GalaxeaVLA adapter resolves the '
                                    'checkpoint path and reads .hydra/config.yaml beside the resolved file, so the symlinked '
                                    'bundle made it read the original config with repository-relative processor paths '
                                    '(job 1690: HF processor tokenizer.json not found)',
                    g05_profile_fix='G0.5 args.sim_task=robotwin: the upstream evaluator passes the Hydra task choice '
                                    '(README: task=robotwin) to the adapter; without it sim_robotwin.yaml composes no '
                                    'data group and get_model fails on cfg.data (job 1693)',
                    policy_profile_fix='Fast-WAM args.sigma_shift=5.0: upstream README evaluates the original released checkpoints '
                                       'with EVALUATION.sigma_shift=5.0; unset, the action scheduler used infer_shift 1.0 (job 1685)',
                    experiment=EXPERIMENT, curobo_fix=prior['curobo_fix'],
                    transforms_sha256=hashlib.sha256(vendor['transforms.py']).hexdigest(),
                    planner_sha256=hashlib.sha256(vendor['planner.py']).hexdigest(),
                    source_before=prior['source_sha256'], source_changes=sorted(changed),
                    rule='success-context-from-initial-scene-v1',
                    fix='evaluation worker derives open_laptop/place_object_scale/put_object_cabinet check_success attributes '
                        '(arm_tag, origin_z) from the verified initial scene, as the official evaluator inherits them from play_once',
                    pool_mode='continue the same v6 pool; preserve saved evidence; no expert replay')
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        def write(relative, data):
            info = zipfile.ZipInfo(NAME + '/' + relative, date_time=(2026, 10, 15, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, data)
        for base_dir, files in (('src', source), ('tools', tools)):
            for relative in files:
                write(base_dir + '/' + relative, (ROOT / base_dir / relative).read_bytes())
        for name, data in vendor.items():
            write('vendor/' + name, data)
        for relative in configs:
            write('configs/' + relative, (ROOT / 'configs' / relative).read_bytes())
        write('oidn/_oidn_tricks.py', loader)
        write('manifest.json', json.dumps(manifest, indent=2) + '\n')
    archive = ROOT / 'reports' / (NAME + '.zip')
    data = output.getvalue()
    if archive.exists() and archive.read_bytes() != data:
        raise FileExistsError('Do not overwrite a delivered snapshot; use a new package name')
    archive.write_bytes(data)
    archive_hash = hashlib.sha256(data).hexdigest()
    text = (ROOT / 'cluster/job-pro6000x2-fastwam-v6-all.sbatch').read_text()
    text = text.replace(BASE, NAME)
    text = text.replace('#SBATCH --job-name=fastwam-phys-v6', '#SBATCH --job-name=fastwam-phys-v6-ctx')
    # User's cluster setting for this run: two PRO 6000, 32 CPUs, 96 GB, 4 pool workers per GPU, 4 replicas.
    text = text.replace('#SBATCH --cpus-per-task=64', '#SBATCH --cpus-per-task=32')
    text = text.replace('#SBATCH --mem=128G', '#SBATCH --mem=96G')
    text = text.replace('export APPTAINERENV_ROBOTWIN_FREEZE_WORKERS_PER_GPU=8', 'export APPTAINERENV_ROBOTWIN_FREEZE_WORKERS_PER_GPU=4')
    text = text.replace('# Pool preparation only: eight independent tasks per allocated GPU (about 6 GB VRAM and 4 CPUs each).',
                        '# Pool continuation: four independent tasks per allocated GPU (about 6 GB VRAM and 4 CPUs each).')
    for name in ('source_sha256', 'tools_sha256'):
        text = text.replace(prior[name], manifest[name])
    text = re.sub(r"printf '%s  %s\\n' [0-9a-f]{64}", lambda m: "printf '%s  %s\\n' " + archive_hash, text)
    text = text.replace('FRESH_POOL_V2_PATCH_SOURCE_PASS: reviewed simulator snapshot verified',
                        MARKER + ': reviewed simulator snapshot verified')
    text = text.replace('echo "POOL_THEN_FASTWAM: dual-GPU layout; policy replicas=$ROBOTWIN_POLICY_REPLICAS"',
                        'echo "POOL_THEN_FASTWAM: dual-GPU layout; policy replicas=$ROBOTWIN_POLICY_REPLICAS spread over both GPUs"\n'
                        'echo "SUCCESS_CONTEXT_FIX: same v6 pool; saved evidence preserved; no expert replay"\n'
                        'echo "G05_RUNTIME_FIX: prepare_g05 hard-links the checkpoint so the adapter reads the runtime config (job 1690)"')
    exec_line = 'exec bash "$ROOT/cluster/run_parallel.sh" --seed 100000 --freeze-workers-per-gpu "$ROBOTWIN_FREEZE_WORKERS_PER_GPU"'
    if exec_line not in text:
        raise RuntimeError('Unexpected baseline launcher exec line')
    text = text.replace(exec_line,
        'export XDG_CACHE_HOME="$HOME/.cache" XDG_CONFIG_HOME="$HOME/.config" HF_HOME="$HOME/.cache/huggingface" '
        'MPLCONFIGDIR="$HOME/.cache/matplotlib" TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_extensions"\n'
        'cd "$ROOT"\n'
        '# Finish any task left unfinished by the per-task workers with all workers scanning its seeds in parallel.\n'
        '"$ROOT/.env/bin/python" "$ROOT/tools/parallel_freeze_task.py" --experiment "$ROBOTWIN_EXPERIMENT" '
        '--workers "$ROBOTWIN_PARALLEL_FREEZE_WORKERS"\n'
        '# One model after another on the same sealed pool; the continuation runs once, later models reuse the pool.\n'
        'for model in $ROBOTWIN_MODELS; do\n'
        '  case "$model" in fact) replicas=1 ;; motus) replicas=2 ;; *) replicas="$ROBOTWIN_POLICY_REPLICAS" ;; esac\n'
        '  echo "MODEL_START: $model; replicas=$replicas"\n'
        '  "$ROOT/.env/bin/python" "$ROOT/tools/continue_success_context.py" --model "$model" '
        '--experiment "$ROBOTWIN_EXPERIMENT" --phase "$ROBOTWIN_PHASE" --episodes "$ROBOTWIN_EPISODES" '
        '--video "$ROBOTWIN_VIDEO_SELECTION" --replicas "$replicas" --seed 100000 '
        '--freeze-workers-per-gpu "$ROBOTWIN_FREEZE_WORKERS_PER_GPU" --timeout ' + str(EPISODE_TIMEOUT) + '\n'
        '  echo "MODEL_DONE: $model"\n'
        'done')
    planner_bind = 'export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$ROBOTWIN_PATCH_DIR/vendor/planner.py:$ROOT/vendor/RoboTwin/envs/robot/planner.py:ro"\n'
    if planner_bind not in text:
        raise RuntimeError('Unexpected baseline vendor bind line')
    text = text.replace(planner_bind, planner_bind
        + '# Fast-WAM profile with sigma_shift=5.0 (upstream released-checkpoint setting); the policy server reads its own workspace copy.\n'
        + 'export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$ROBOTWIN_PATCH_DIR/configs/policies/fastwam.yaml:$ROOT/configs/policies/fastwam.yaml:ro"\n'
        + f'export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$ROBOTWIN_PATCH_DIR/configs/policies/fastwam.yaml:{FASTWAM_WORKSPACE}/configs/policies/fastwam.yaml:ro"\n'
        + '# G0.5 profile with sim_task=robotwin (upstream evaluator setting); bound to the evaluator tree and the G0.5 policy workspace.\n'
        + 'export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$ROBOTWIN_PATCH_DIR/configs/policies/g05.yaml:$ROOT/configs/policies/g05.yaml:ro"\n'
        + f'export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$ROBOTWIN_PATCH_DIR/configs/policies/g05.yaml:{G05_WORKSPACE}/configs/policies/g05.yaml:ro"\n')
    fastwam_bind = f'export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$ROBOTWIN_PATCH_DIR/configs/policies/fastwam.yaml:{FASTWAM_WORKSPACE}/configs/policies/fastwam.yaml:ro"\n'
    if fastwam_bind not in text:
        raise RuntimeError('Fast-WAM profile bind missing')
    oidn_block = (
        '# Intel Open Image Denoise ' + OIDN_VERSION + ' for Blackwell: SAPIEN 3.0.0b1 bundles 2.0.1, whose CUDA device fails on\n'
        '# RTX PRO 6000 ("OIDN Error: invalid handle") and leaves the ray-traced policy observations un-denoised.\n'
        'ROBOTWIN_OIDN_DIR="$DATA/downloads/oidn-' + OIDN_VERSION + '"\n'
        'mkdir -p "$ROBOTWIN_OIDN_DIR"\n'
        '(\n'
        '  flock -x 9\n'
        '  if [[ ! -r "$ROBOTWIN_OIDN_DIR/oidn_library/libOpenImageDenoise_device_cuda.so.' + OIDN_VERSION + '" ]]; then\n'
        '    if [[ ! -r "$ROBOTWIN_OIDN_DIR/oidn.tar.gz" ]]; then\n'
        '      curl -fsSL -o "$ROBOTWIN_OIDN_DIR/oidn.tar.gz" "' + OIDN_URL + '"\n'
        '    fi\n'
        "    printf '%s  %s\\n' " + OIDN_SHA256 + ' "$ROBOTWIN_OIDN_DIR/oidn.tar.gz" | sha256sum -c -\n'
        '    ROBOTWIN_OIDN_STAGE=$(mktemp -d "$ROBOTWIN_OIDN_DIR/.stage-XXXXXX")\n'
        '    tar -xzf "$ROBOTWIN_OIDN_DIR/oidn.tar.gz" -C "$ROBOTWIN_OIDN_STAGE"\n'
        '    rm -rf "$ROBOTWIN_OIDN_DIR/oidn_library"\n'
        '    mkdir -p "$ROBOTWIN_OIDN_DIR/oidn_library"\n'
        '    cp -a "$ROBOTWIN_OIDN_STAGE"/oidn-' + OIDN_VERSION + '.x86_64.linux/lib/*.so* "$ROBOTWIN_OIDN_DIR/oidn_library/"\n'
        '    rm -rf "$ROBOTWIN_OIDN_STAGE"\n'
        '  fi\n'
        ') 9> "$ROBOTWIN_OIDN_DIR/.lock"\n'
        'export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$ROBOTWIN_OIDN_DIR/oidn_library:' + SAPIEN_PKG + '/oidn_library:ro"\n'
        'export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$ROBOTWIN_PATCH_DIR/oidn/_oidn_tricks.py:' + SAPIEN_PKG + '/_oidn_tricks.py:ro"\n')
    text = text.replace(fastwam_bind, fastwam_bind + oidn_block)
    planner_check = "    raise SystemExit('Planner exception repair was not bound correctly')\n"
    if planner_check not in text:
        raise RuntimeError('Unexpected baseline planner check')
    text = text.replace(planner_check, planner_check
        + "for copy in (root / 'configs/policies/fastwam.yaml', Path('" + FASTWAM_WORKSPACE + "/configs/policies/fastwam.yaml')):\n"
        + "    if hashlib.sha256(copy.read_bytes()).hexdigest() != '" + configs['policies/fastwam.yaml'] + "':\n"
        + "        raise SystemExit(f'Fast-WAM profile (sigma_shift=5.0) was not bound correctly: {copy}')\n"
        + "print('FASTWAM_SIGMA_SHIFT: profile with sigma_shift=5.0 bound to evaluator and policy workspace', flush=True)\n"
        + "for copy in (root / 'configs/policies/g05.yaml', Path('" + G05_WORKSPACE + "/configs/policies/g05.yaml')):\n"
        + "    if hashlib.sha256(copy.read_bytes()).hexdigest() != '" + configs['policies/g05.yaml'] + "':\n"
        + "        raise SystemExit(f'G0.5 profile (sim_task=robotwin) was not bound correctly: {copy}')\n"
        + "print('G05_SIM_TASK: profile with sim_task=robotwin bound to evaluator and policy workspace', flush=True)\n"
        + "sapien_pkg = Path('" + SAPIEN_PKG + "')\n"
        + "for name in ('libOpenImageDenoise.so." + OIDN_VERSION + "', 'libOpenImageDenoise_core.so." + OIDN_VERSION + "', 'libOpenImageDenoise_device_cuda.so." + OIDN_VERSION + "'):\n"
        + "    if not (sapien_pkg / 'oidn_library' / name).exists():\n"
        + "        raise SystemExit(f'OIDN " + OIDN_VERSION + " library was not bound: {name}')\n"
        + "if hashlib.sha256((sapien_pkg / '_oidn_tricks.py').read_bytes()).hexdigest() != '" + hashlib.sha256(loader).hexdigest() + "':\n"
        + "    raise SystemExit('OIDN loader was not bound over sapien/_oidn_tricks.py')\n")
    probe = ('# Prove the renderer actually loads OIDN ' + OIDN_VERSION + ' and denoises without errors before any episode.\n'
             '"$ROOT/.env/bin/python" - <<\'PY_OIDN_PROBE\' 2> "$ROOT/reports/launchers/$SLURM_JOB_ID/oidn-probe.stderr"\n'
             'import json, os, numpy as np, sapien\n'
             'sapien.render.set_camera_shader_dir("rt"); sapien.render.set_ray_tracing_samples_per_pixel(32)\n'
             'sapien.render.set_ray_tracing_path_depth(8); sapien.render.set_ray_tracing_denoiser("oidn")\n'
             'scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem(os.environ.get("SAPIEN_RENDER_DEVICE", "cuda:0"))])\n'
             'scene.add_ground(0); scene.set_ambient_light([0.3, 0.3, 0.3]); scene.add_directional_light([0, 1, -1], [1, 1, 1])\n'
             'b = scene.create_actor_builder(); b.add_box_visual(half_size=[0.1, 0.1, 0.1]); box = b.build_static(name="box"); box.set_pose(sapien.Pose([0, 0, 0.1]))\n'
             'cam = scene.add_camera("c", 320, 240, 1.0, 0.1, 100); cam.set_entity_pose(sapien.Pose([-1.0, 0, 0.6], [0.966, 0, 0.259, 0]))\n'
             'for _ in range(3):\n'
             '    scene.update_render(); cam.take_picture()\n'
             'g = (cam.get_picture("Color")[..., :3] * 255).clip(0, 255).astype(np.float32).mean(-1)\n'
             'lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]\n'
             'mapped = sorted(set(l.split("/")[-1] for l in open("/proc/self/maps").read().split() if "OpenImageDenoise" in l))\n'
             'record = dict(oidn_libraries=mapped, ground_patch_laplacian_std=float(lap[150:230, 100:220].std()), sapien=sapien.__version__)\n'
             'json.dump(record, open(os.path.join("' + '/home/guest/robotwin-workspace' + '", "reports/launchers", os.environ["SLURM_JOB_ID"], "oidn-probe.json"), "w"), indent=2)\n'
             'if not any("' + OIDN_VERSION + '" in m for m in mapped) or "libOpenImageDenoise_device_cuda.so.' + OIDN_VERSION + '" not in mapped:\n'
             '    raise SystemExit(f"OIDN ' + OIDN_VERSION + ' CUDA device not loaded: {mapped}")\n'
             'print(f"OIDN_PASS: {mapped}; ground patch laplacian std {record[\'ground_patch_laplacian_std\']:.2f}", flush=True)\n'
             'PY_OIDN_PROBE\n'
             'if grep -q "OIDN Error" "$ROOT/reports/launchers/$SLURM_JOB_ID/oidn-probe.stderr"; then\n'
             '  echo "OIDN denoiser reported errors in the probe; see reports/launchers/$SLURM_JOB_ID/oidn-probe.stderr" >&2\n'
             '  exit 1\n'
             'fi\n')
    probe_anchor = "PY_EXPERT_PATCH\n"
    if text.count(probe_anchor) != 1:  # the opening marker is quoted (<<'PY_EXPERT_PATCH'); only the close matches
        raise RuntimeError('Expected exactly one PY_EXPERT_PATCH heredoc close marker')
    close = text.rfind(probe_anchor) + len(probe_anchor)
    text = text[:close] + probe + text[close:]
    text = text.replace('export APPTAINERENV_ROBOTWIN_MODEL="fastwam"\n',
                        'export APPTAINERENV_ROBOTWIN_MODEL="fastwam"\n'
                        '# Models evaluated in this job, in order, all on the same pool (RoboTwin 2.0 fine-tuned checkpoints pinned in configs/policies).\n'
                        'export APPTAINERENV_ROBOTWIN_MODELS="fastwam g05 pi05"\n'
                        '# Workers scanning seeds concurrently for a task the per-task loop left unfinished (same records, committed in seed order).\n'
                        'export APPTAINERENV_ROBOTWIN_PARALLEL_FREEZE_WORKERS=8\n')
    for check in (NAME, EXPERIMENT, archive_hash, manifest['source_sha256'], manifest['tools_sha256'], MARKER,
                  'tools/continue_success_context.py', 'SUCCESS_CONTEXT_FIX', 'G05_RUNTIME_FIX', 'spread over both GPUs', '--cpus-per-task=32', '--mem=96G',
                  'ROBOTWIN_MODELS="fastwam g05 pi05"', 'MODEL_START', 'MODEL_DONE', 'tools/parallel_freeze_task.py',
                  'PARALLEL_FREEZE_WORKERS=8', 'FASTWAM_SIGMA_SHIFT', configs['policies/fastwam.yaml'], FASTWAM_WORKSPACE,
                  'G05_SIM_TASK', configs['policies/g05.yaml'], G05_WORKSPACE + '/configs/policies/g05.yaml:ro',
                  'OIDN_PASS', OIDN_URL, OIDN_SHA256, SAPIEN_PKG + '/oidn_library:ro', SAPIEN_PKG + '/_oidn_tricks.py:ro', '--timeout ' + str(EPISODE_TIMEOUT),
                  'FREEZE_WORKERS_PER_GPU=4', 'POLICY_REPLICAS=4', 'gpu:pro6000:2', '--time=3-00:00:00'):
        if check not in text:
            raise RuntimeError(f'Launcher substitution failed for {check}')
    if BASE in text or any(f'robotwin-success-context-fix-v{n}' in text for n in (1, 2, 3, 4, 5, 6)):
        raise RuntimeError('Launcher still references a superseded snapshot')
    for filename in ('job-pro6000x2-fastwam-v6-context.sbatch', 'ood-fastwam-pro6000x2-v6-context.txt'):
        (ROOT / 'cluster' / filename).write_text(text)
    print(f'Packaged {archive} ({archive_hash}): {EXPERIMENT}, success-context fix, Fast-WAM sigma_shift=5.0, OIDN {OIDN_VERSION}, G0.5 runtime bundle fix, G0.5 sim_task, pool preserved')


if __name__ == '__main__':
    main()

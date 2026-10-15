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
NAME = 'robotwin-success-context-fix-v4'
PREVIOUS_DELIVERY = 'robotwin-success-context-fix-v3'
CONFIGS = ('policies/fastwam.yaml',)  # policy profiles shipped with the snapshot (not in the pool fingerprint)
FASTWAM_WORKSPACE = '/opt/robotwin-policies/fastwam/workspace'
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
    import yaml
    if yaml.safe_load((ROOT / 'configs/policies/fastwam.yaml').read_text())['args'].get('sigma_shift') != 5.0:
        raise SystemExit('configs/policies/fastwam.yaml must set args.sigma_shift: 5.0 (upstream released-checkpoint setting)')
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
                        'echo "SUCCESS_CONTEXT_FIX: same v6 pool; saved evidence preserved; no expert replay"')
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
        '--freeze-workers-per-gpu "$ROBOTWIN_FREEZE_WORKERS_PER_GPU"\n'
        '  echo "MODEL_DONE: $model"\n'
        'done')
    planner_bind = 'export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$ROBOTWIN_PATCH_DIR/vendor/planner.py:$ROOT/vendor/RoboTwin/envs/robot/planner.py:ro"\n'
    if planner_bind not in text:
        raise RuntimeError('Unexpected baseline vendor bind line')
    text = text.replace(planner_bind, planner_bind
        + '# Fast-WAM profile with sigma_shift=5.0 (upstream released-checkpoint setting); the policy server reads its own workspace copy.\n'
        + 'export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$ROBOTWIN_PATCH_DIR/configs/policies/fastwam.yaml:$ROOT/configs/policies/fastwam.yaml:ro"\n'
        + f'export APPTAINER_BINDPATH="$APPTAINER_BINDPATH,$ROBOTWIN_PATCH_DIR/configs/policies/fastwam.yaml:{FASTWAM_WORKSPACE}/configs/policies/fastwam.yaml:ro"\n')
    planner_check = "    raise SystemExit('Planner exception repair was not bound correctly')\n"
    if planner_check not in text:
        raise RuntimeError('Unexpected baseline planner check')
    text = text.replace(planner_check, planner_check
        + "for copy in (root / 'configs/policies/fastwam.yaml', Path('" + FASTWAM_WORKSPACE + "/configs/policies/fastwam.yaml')):\n"
        + "    if hashlib.sha256(copy.read_bytes()).hexdigest() != '" + configs['policies/fastwam.yaml'] + "':\n"
        + "        raise SystemExit(f'Fast-WAM profile (sigma_shift=5.0) was not bound correctly: {copy}')\n"
        + "print('FASTWAM_SIGMA_SHIFT: profile with sigma_shift=5.0 bound to evaluator and policy workspace', flush=True)\n")
    text = text.replace('export APPTAINERENV_ROBOTWIN_MODEL="fastwam"\n',
                        'export APPTAINERENV_ROBOTWIN_MODEL="fastwam"\n'
                        '# Models evaluated in this job, in order, all on the same pool (RoboTwin 2.0 fine-tuned checkpoints pinned in configs/policies).\n'
                        'export APPTAINERENV_ROBOTWIN_MODELS="fastwam g05 pi05"\n'
                        '# Workers scanning seeds concurrently for a task the per-task loop left unfinished (same records, committed in seed order).\n'
                        'export APPTAINERENV_ROBOTWIN_PARALLEL_FREEZE_WORKERS=8\n')
    for check in (NAME, EXPERIMENT, archive_hash, manifest['source_sha256'], manifest['tools_sha256'], MARKER,
                  'tools/continue_success_context.py', 'SUCCESS_CONTEXT_FIX', 'spread over both GPUs', '--cpus-per-task=32', '--mem=96G',
                  'ROBOTWIN_MODELS="fastwam g05 pi05"', 'MODEL_START', 'MODEL_DONE', 'tools/parallel_freeze_task.py',
                  'PARALLEL_FREEZE_WORKERS=8', 'FASTWAM_SIGMA_SHIFT', configs['policies/fastwam.yaml'], FASTWAM_WORKSPACE,
                  'FREEZE_WORKERS_PER_GPU=4', 'POLICY_REPLICAS=4', 'gpu:pro6000:2', '--time=3-00:00:00'):
        if check not in text:
            raise RuntimeError(f'Launcher substitution failed for {check}')
    if BASE in text or any(f'robotwin-success-context-fix-v{n}' in text for n in (1, 2, 3)):
        raise RuntimeError('Launcher still references a superseded snapshot')
    for filename in ('job-pro6000x2-fastwam-v6-context.sbatch', 'ood-fastwam-pro6000x2-v6-context.txt'):
        (ROOT / 'cluster' / filename).write_text(text)
    print(f'Packaged {archive} ({archive_hash}): {EXPERIMENT}, success-context fix, Fast-WAM sigma_shift=5.0, pool preserved')


if __name__ == '__main__':
    main()

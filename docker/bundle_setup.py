"""Relocate isolated policy runtimes at image build time and verify their imports."""
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/home/guest/robotwin-workspace')
POLICIES = Path('/opt/robotwin-policies')
records = []
for model in ['pi05', 'fact', 'motus', 'fastwam', 'g05']:
    runtime = POLICIES/model
    venv, workspace, build = runtime/'venv', runtime/'workspace', runtime/'build'
    assert (build/'model.txt').read_text().strip() == model
    # The base CPython installations retain /opt/python; only venv entry scripts
    # and any .pth source paths need relocation. Binary libraries are unchanged.
    paths = list((venv/'bin').iterdir()) + list(venv.glob('lib/python*/site-packages/*.pth'))
    for path in paths:
        if not path.is_file() or path.is_symlink():
            continue
        data = path.read_bytes()
        if data.startswith(b'\x7fELF') or b'\x00' in data:
            continue
        updated = data.replace(b'/opt/policy', str(venv).encode()).replace(str(ROOT).encode(), str(workspace).encode())
        if updated != data:
            path.write_bytes(updated)
    for name in ['checkpoints', 'reports', 'runs']:
        target = workspace/name
        if target.is_symlink(): target.unlink()
        elif target.exists():
            if any(target.iterdir()): raise ValueError(f'Unexpected data baked in image: {target}')
            target.rmdir()
        target.symlink_to(ROOT/name, target_is_directory=True)
    shutil.copyfile(ROOT/'tools/serve_policy.py', workspace/'tools/serve_policy.py')
    shutil.copyfile(ROOT/'docker/policies/probe.py', workspace/'docker/policies/probe.py')
    for vendor in ['GalaxeaVLA', 'FastWAM', 'FACT', 'Motus']:
        subprocess.run(['git', 'config', '--system', '--add', 'safe.directory', str(workspace/'vendor'/vendor)], check=True)
    env = dict(os.environ, ROBOTWIN_PHYS_ROOT=str(workspace), ROBOTWIN_BUILD_INFO=str(build),
               PYTHONPATH=str(workspace/'src'), VIRTUAL_ENV=str(venv),
               PATH=f'{venv}/bin:/usr/local/cuda/bin:/usr/bin:/bin',
               JAX_PLATFORMS='cpu', XLA_PYTHON_CLIENT_PREALLOCATE='false')
    python = str(venv/'bin/python')
    code = 'import json,sys,torch; print(json.dumps(dict(prefix=sys.prefix,python=sys.version,torch=torch.__version__,cuda=torch.version.cuda)))'
    info = json.loads(subprocess.check_output([python, '-c', code], env=env, text=True))
    assert info['prefix'] == str(venv), info
    assert info['torch'] == '2.7.1+cu128' and info['cuda'] == '12.8', info
    subprocess.run([python, str(workspace/'docker/policies/probe.py'), '--model', model], cwd=workspace, env=env, check=True)
    subprocess.run([python, str(workspace/'tools/serve_policy.py'), '--help'], cwd=workspace, env=env, check=True, stdout=subprocess.DEVNULL)
    records.append(dict(model=model, **info, imports_passed=True, learned_inference_tested=False))
(Path('/opt/robotwin-build')/'bundle-environments.json').write_text(json.dumps(records, indent=2)+'\n')

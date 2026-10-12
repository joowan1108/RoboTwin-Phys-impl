"""Import adapters/model classes without weights or GPU; fails image build on missing imports."""
import argparse
import importlib
import os
import sys
from pathlib import Path
import yaml
import torch

p = argparse.ArgumentParser(); p.add_argument('--model', required=True); a=p.parse_args()
assert torch.__version__ == '2.7.1+cu128', torch.__version__
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'src'))
from robotwin_phys.policy_server import load_adapter
cfg=yaml.safe_load((root/f'configs/policies/{a.model}.yaml').read_text())
load_adapter(cfg)
if a.model == 'pi05':
    from openpi.training import config
    from openpi.policies.policy_config import create_trained_policy
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from transformers.models.siglip import check
    assert check.check_whether_transformers_replace_is_installed_correctly()
elif a.model == 'fact':
    importlib.import_module('scripts.inference_server')
elif a.model == 'g05':
    importlib.import_module('g05.models.g05.g05_model_qwen35')
elif a.model == 'fastwam':
    importlib.import_module('fastwam.models.wan22.fastwam')
print(f'{a.model}: adapter/model imports passed; weights/inference NOT tested')

# LeRobot is imported by OpenPI training type declarations. Its training/teleop
# extras and torch<2.7 pin are intentionally outside this Blackwell inference image.
import subprocess
import json
check = subprocess.run([sys.executable, '-m', 'pip', 'check'], text=True, capture_output=True)
lines = check.stdout.splitlines() if check.returncode else []
if check.returncode and (a.model != 'pi05' or not lines or any(not line.startswith('lerobot 0.1.0 ') for line in lines)):
    raise RuntimeError('Unexpected dependency conflict: ' + check.stdout + check.stderr)
(Path(os.environ.get('ROBOTWIN_BUILD_INFO', '/opt/robotwin-build'))/'pip-check.json').write_text(json.dumps({
    'returncode': check.returncode, 'stdout': check.stdout,
    'known_inference_only_exceptions': lines}, indent=2)+'\n')

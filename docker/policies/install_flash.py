import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
r=json.loads(Path('docker/policies/flash-attn-wheel.json').read_text())
p=Path('/tmp')/r['name']
urllib.request.urlretrieve(r['browser_download_url'],p)
h=hashlib.sha256(p.read_bytes()).hexdigest()
assert 'sha256:'+h==r['digest'], 'FlashAttention wheel checksum mismatch'
subprocess.run([sys.executable,'-m','pip','install','--no-deps',str(p)],check=True)
p.unlink()
import torch
import flash_attn_2_cuda
out=subprocess.check_output(['cuobjdump','--list-elf',flash_attn_2_cuda.__file__],text=True)
assert 'sm_120' in out, 'Official FlashAttention wheel lacks native sm120 kernels'

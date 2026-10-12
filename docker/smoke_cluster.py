"""Exercise the new CUDA image's frozen manifest through RPC; explicitly unscored."""
import subprocess
import sys
import time
from robotwin_phys.rpc import PolicyClient
server=subprocess.Popen([sys.executable,'-m','robotwin_phys.policy_server','--config','configs/policies/noop.yaml','--port','19091'])
try:
    for _ in range(60):
        if server.poll() is not None: raise RuntimeError('Policy server died')
        try:
            c=PolicyClient('127.0.0.1',19091,timeout=1);c.close();break
        except OSError: time.sleep(.5)
    else: raise TimeoutError('Noop startup')
    subprocess.run([sys.executable,'-m','robotwin_phys.cli','evaluate','--manifest','manifests/cluster-cu128-local-smoke.json',
        '--output','runs/cluster-cu128-local-noop','--policy-port','19091','--smoke-max-actions','1','--video','--resume'],check=True)
finally:
    server.terminate();server.wait(timeout=10)

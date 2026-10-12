#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from robotwin_phys.rpc import PolicyClient
from policy_runtime import read_policy_config
from robotwin_phys.conditions import digest
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--port',type=int,required=True)
p.add_argument('--config',type=Path)
p.add_argument('--config-root',type=Path,default=Path(__file__).resolve().parents[1])
p.add_argument('--timeout',type=int,default=2400);p.add_argument('--process-pid',type=int)
a=p.parse_args();deadline=time.monotonic()+a.timeout
while True:
    if a.process_pid:
        try: os.kill(a.process_pid,0)
        except ProcessLookupError: raise SystemExit('Policy process exited; inspect reports/jobs/<job>/policy.log')
    if a.config and not a.config.exists():
        if time.monotonic()>deadline: raise SystemExit('Policy preparation timeout; inspect policy.log')
        time.sleep(2);continue
    try:
        c=PolicyClient('127.0.0.1',a.port,timeout=3);identity=c.identity;c.close()
    except (OSError,ConnectionError):
        if time.monotonic()>deadline: raise SystemExit('Policy startup timeout; inspect policy.log')
        time.sleep(2);continue
    if identity['name']!=f'{a.model}_robotwin2': raise SystemExit('Port occupied by another policy')
    if a.config and identity.get('config_sha256')!=digest(read_policy_config(a.config, a.config_root)):
        raise SystemExit('Port occupied by a different policy configuration')
    print(f'Policy ready: {identity["name"]}',flush=True);break

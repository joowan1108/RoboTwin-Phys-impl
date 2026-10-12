#!/usr/bin/env python3
"""Verify downloaded bytes, record runtime identity, and supervise the selected model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import yaml
from prepare_cluster import checksum

ROOT=Path(__file__).resolve().parents[1]
BUILD_INFO=Path(os.environ.get('ROBOTWIN_BUILD_INFO', '/opt/robotwin-build'))
sys.path.insert(0,str(ROOT/'src'))
from robotwin_phys.conditions import digest
from robotwin_phys.manifest import atomic_json


def verify_lock(folder, repo=None, revision=None):
    lock=json.loads((folder/'download-lock.json').read_text())
    if repo and (lock['repo']!=repo or lock['revision']!=revision):
        raise ValueError(f'Checkpoint revision mismatch: {folder}')
    for record in lock['files']:
        path=folder/record['file']
        if not path.resolve().is_relative_to(folder.resolve()): raise ValueError('Invalid artifact path')
        if path.stat().st_size!=record['size'] or checksum(path)!=record['sha256']:
            raise ValueError(f'Artifact checksum mismatch: {path}')
    if not lock['files']: raise ValueError(f'Empty download: {folder}')
    return lock


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=['g05','pi05','fact','motus','fastwam'],required=True)
    p.add_argument('--t5-device',choices=['cuda:0','cuda:1'],default='cuda:0')
    p.add_argument('--port',type=int,default=9000);p.add_argument('--native-port',type=int,default=8093)
    p.add_argument('--record-dir',type=Path,required=True);a=p.parse_args()
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',a.port))
    marker=BUILD_INFO/'model.txt'
    if not marker.exists() or marker.read_text().strip()!=a.model: raise ValueError('Wrong policy image')
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        raise SystemExit('Official BF16 models need an Ampere or newer GPU')
    if a.model == 'motus':
        from flash_attn import flash_attn_func
        q=torch.randn(1,32,4,64,device='cuda',dtype=torch.bfloat16)
        assert torch.isfinite(flash_attn_func(q,q,q)).all()
        torch.cuda.synchronize()
    cfg=yaml.safe_load((ROOT/f'configs/policies/{a.model}.yaml').read_text())
    if a.model=='motus':
        if a.t5_device=='cuda:1' and torch.cuda.device_count()<2:
            raise SystemExit('Motus cuda:1 T5 placement needs two allocated GPUs visible to the policy container')
        cfg['args']['t5_device']=a.t5_device
        cfg['minimum_vram_gb']=24 if a.t5_device=='cuda:1' else 41
    ckpt=cfg['checkpoint']
    locks={a.model:verify_lock(ROOT/'checkpoints'/a.model,ckpt['repo'],ckpt['revision'])}
    for spec in json.loads((ROOT/'configs/backbones.json').read_text()):
        if a.model in spec['models']:
            locks[spec['name']]=verify_lock(ROOT/'checkpoints'/spec['destination'],spec['repo'],spec['revision'])
    if a.model=='pi05': locks['tokenizer']=verify_lock(ROOT/'checkpoints/openpi')
    if a.model=='g05':
        locks['runtime_config_sha256']=checksum(ROOT/'checkpoints/g05-runtime/.hydra/config.yaml')
    # Record code used by the native FACT server and any patched vendor files.
    files={}
    vendor={'fact':'FACT','g05':'GalaxeaVLA','fastwam':'FastWAM','motus':'Motus','pi05':'RoboTwin/policy/pi05'}[a.model]
    for directory in ['tools','docker/policies','vendor/'+vendor]:
        for f in sorted((ROOT/directory).rglob('*')):
            if not f.is_file() or f.suffix not in ('.py','.yaml','.yml','.json') or '.git' in f.parts: continue
            files[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    cfg['runtime_provenance']={'artifacts':locks,'support_code_sha256':digest(files),
         'packages':(BUILD_INFO/'pip-freeze.txt').read_text(),
         'python':sys.version,'torch_cuda':'12.8'}
    env=os.environ.copy()
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        DIFFSYNTH_SKIP_DOWNLOAD='true',DIFFSYNTH_MODEL_BASE_PATH=str(ROOT/'checkpoints/backbones'),
        OPENPI_DATA_HOME=str(ROOT/'checkpoints/openpi'),LOG_DIR=str(a.record_dir/'model-output'))
    a.record_dir.mkdir(parents=True,exist_ok=True)
    # Runtime port is transport-only; fixed FACT adapter port still enters the config hash.
    if a.model=='fact': cfg['args']['port']=a.native_port
    file=a.record_dir/'policy.yaml';file.write_text(yaml.safe_dump(cfg,sort_keys=False))
    children=[]
    def stop(*_):
        for child in reversed(children):
            if child.poll() is None: child.terminate()
        for child in children:
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
    def interrupted(signum, frame):
        stop(); raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        if a.model=='fact':
            cmd=[sys.executable,'-m','scripts.inference_server','--model_id',str(ROOT/'checkpoints/backbones/Wan2.2-TI2V-5B-Diffusers'),
                 '--transformer_path',str(ROOT/'checkpoints/fact/transformer'),'--stats_path',str(ROOT/'checkpoints/fact/norm_stats_delta.json'),
                 '--port',str(a.native_port)]
            native=subprocess.Popen(cmd,cwd=ROOT/'vendor/FACT',env=env);children.append(native)
            # Native server's client adapter has its own retry/ping; do not consume its protocol here.
            deadline=time.monotonic()+1800
            while True:
                if native.poll() is not None: raise RuntimeError('FACT native server exited')
                try:
                    with socket.create_connection(('127.0.0.1',a.native_port),timeout=1): break
                except OSError:
                    if time.monotonic()>deadline: raise TimeoutError('FACT model loading timeout')
                    time.sleep(2)
        bridge=subprocess.Popen([sys.executable,'-m','robotwin_phys.policy_server','--config',str(file),'--port',str(a.port)],env=env)
        children.append(bridge)
        while bridge.poll() is None:
            if len(children)>1 and children[0].poll() is not None: raise RuntimeError('FACT native server died')
            time.sleep(1)
        if bridge.returncode: raise subprocess.CalledProcessError(bridge.returncode,bridge.args)
    finally: stop()


if __name__=='__main__': main()

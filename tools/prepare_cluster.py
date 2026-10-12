#!/usr/bin/env python3
"""Download pinned assets and selected models into the bound persistent directories."""
import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def main():
    from huggingface_hub import hf_hub_download
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', action='store_true')
    p.add_argument('--models', nargs='*', choices=['g05','pi05','fact','motus','fastwam'], default=[])
    a=p.parse_args()
    if a.assets:
        subprocess.run([sys.executable,str(ROOT/'tools/download_assets.py')],check=True)
    if a.models:
        subprocess.run([sys.executable,str(ROOT/'tools/download_models.py'),'--models',*a.models],check=True)
    for spec in json.loads((ROOT/'configs/backbones.json').read_text()):
        if not set(spec['models']) & set(a.models): continue
        info=json.loads((ROOT/('references/'+spec['repo'].replace('/','--')+'-info.json')).read_text())
        assert info['sha']==spec['revision']
        records=[]
        for entry in info['siblings']:
            name=entry['rfilename']
            if not any(fnmatch.fnmatch(name,p) for p in spec['patterns']): continue
            path=Path(hf_hub_download(spec['repo'],name,revision=spec['revision'],local_dir=ROOT/'checkpoints'/spec['destination']))
            sha=checksum(path)
            expected=entry.get('lfs',{}).get('sha256')
            if expected and sha!=expected: raise ValueError(f'Bad checksum: {path}')
            records.append(dict(file=name,sha256=sha,size=path.stat().st_size))
        if not records: raise ValueError(f'Empty backbone selection: {spec["name"]}')
        (ROOT/'checkpoints'/spec['destination']/'download-lock.json').write_text(json.dumps(dict(repo=spec['repo'],revision=spec['revision'],files=records),indent=2)+'\n')
    if 'fastwam' in a.models:
        link=ROOT/'checkpoints/backbones/Wan-AI/Wan2.2-TI2V-5B'
        link.parent.mkdir(parents=True,exist_ok=True)
        if not link.exists(): link.symlink_to('../Wan2.2-TI2V-5B',target_is_directory=True)
    if 'g05' in a.models:
        subprocess.run([sys.executable,str(ROOT/'tools/prepare_g05.py')],check=True)
    if 'pi05' in a.models:
        path=ROOT/'checkpoints/openpi/big_vision/paligemma_tokenizer.model'
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            tmp=path.with_suffix('.partial'); urllib.request.urlretrieve('https://storage.googleapis.com/big_vision/paligemma_tokenizer.model',tmp); tmp.replace(path)
        lock=path.parent.parent/'download-lock.json'
        record=dict(repo='gs://big_vision/paligemma_tokenizer.model',revision='content-sha256',files=[dict(file='big_vision/paligemma_tokenizer.model',sha256=checksum(path),size=path.stat().st_size)])
        if lock.exists() and json.loads(lock.read_text())!=record: raise ValueError('Tokenizer changed')
        lock.write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__': main()

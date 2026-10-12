#!/usr/bin/env python3
"""CUDA, Vulkan, and actual CuRobo kernel checks in the allocated GPU job."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-turing',action='store_true')
a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
import torch
if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
    raise SystemExit('Expected exactly one CUDA-visible GPU; preserve SLURM CUDA_VISIBLE_DEVICES')
cap=torch.cuda.get_device_capability(0)
if cap!=(12,0) and not a.allow_turing: raise SystemExit(f'Expected Blackwell sm120; got {cap}')
if torch.version.cuda!='12.8': raise SystemExit('Use the cu128 simulator image')
subprocess.run([sys.executable,'tools/doctor.py','--output',str(a.output)],check=True)
# Exercise CUDA kernels, rather than checking only importability or compiled arch names.
from curobo.types.base import TensorDeviceType
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
args=TensorDeviceType()
config=load_yaml(join_path(get_robot_configs_path(),'franka.yml'))['robot_cfg']
model=CudaRobotModel(CudaRobotModelConfig.from_data_dict(config,args))
q=torch.zeros((1,model.get_dof()),device='cuda')
state=model.get_state(q)
torch.cuda.synchronize()
assert torch.isfinite(state.ee_position).all()
info=json.loads(a.output.read_text());info['curobo_forward_kinematics_pass']=True
info['slurm_job_id']=os.environ.get('SLURM_JOB_ID')
a.output.write_text(json.dumps(info,indent=2)+'\n')
print('CUDA + Vulkan render + CuRobo forward kinematics: passed',flush=True)

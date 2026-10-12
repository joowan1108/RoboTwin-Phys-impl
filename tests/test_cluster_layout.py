import os
from pathlib import Path
import subprocess
import pytest

SCRIPT=(Path(__file__).parents[1]/'cluster/legacy/job.sbatch').read_text()
SELECTION=SCRIPT[SCRIPT.index("IFS=',' read"):SCRIPT.index('export APPTAINERENV_SAPIEN_RENDER_DEVICE')]

@pytest.mark.parametrize('allocation,layout,model,t5,expected',[
    ('3','shared','pi05','cuda:0','3|3'),
    ('2,5','split','pi05','cuda:1','2|5'),
    ('GPU-aaa,GPU-bbb','split','motus','cuda:1','GPU-aaa|GPU-bbb,GPU-aaa'),
    ('1,0','split','motus','cuda:0','1|0'),
])
def test_allocation_preserved(allocation,layout,model,t5,expected):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=allocation,GPU_LAYOUT=layout,MODEL=model,MOTUS_T5_DEVICE=t5)
    r=subprocess.run(['bash','-euc',SELECTION+'printf "%s|%s" "$SIM_GPU" "$POLICY_GPUS"'],env=env,text=True,capture_output=True)
    assert r.returncode==0,r.stderr
    assert r.stdout==expected

@pytest.mark.parametrize('allocation,layout,t5',[('0','split','cuda:1'),('0,0','split','cuda:1'),('0,1','shared','cuda:0'),('0','shared','cuda:1')])
def test_bad_allocation_rejected(allocation,layout,t5):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=allocation,GPU_LAYOUT=layout,MODEL='motus',MOTUS_T5_DEVICE=t5)
    assert subprocess.run(['bash','-euc',SELECTION],env=env,capture_output=True).returncode!=0

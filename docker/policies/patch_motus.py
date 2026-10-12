"""Optional explicit T5 placement for two 32GB GPUs; default preserves original placement."""
from pathlib import Path
p=Path('vendor/Motus/inference/robotwin/Motus/deploy_policy.py');s=p.read_text()
a='task_name: Optional[str] = None):'
assert s.count(a)==1;s=s.replace(a,'task_name: Optional[str] = None, t5_device: Optional[str] = None):')
a='        # Initialize T5 encoder for language embeddings (WAN text encoder)'
s=s.replace(a,'        self.t5_device = t5_device or device\n'+a)
a='            device=device,\n            checkpoint_path=os.path.join(self.wan_path'
assert s.count(a)==1;s=s.replace(a,'            device=self.t5_device,\n            checkpoint_path=os.path.join(self.wan_path')
a='t5_out = self.t5_encoder([instruction], self.device)'
assert s.count(a)==1;s=s.replace(a,'t5_out = [x.to(self.device) for x in self.t5_encoder([instruction], self.t5_device)]')
a="        log_dir=usr_args.get('log_dir'),"
assert s.count(a)==1;s=s.replace(a,"        t5_device=usr_args.get('t5_device'),\n"+a)
p.write_text(s)
# Constructing a default device must not initialize CUDA at module import/build time.
for p in Path('vendor/Motus/inference/robotwin/Motus').rglob('*.py'):
    s=p.read_text()
    if 'device=torch.cuda.current_device(),' in s:
        p.write_text(s.replace('device=torch.cuda.current_device(),','device=torch.device("cuda"),'))

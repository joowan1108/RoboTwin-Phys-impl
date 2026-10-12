"""Propagate bridge's episode-seeded RNG to FACT's native process. No weight/action changes."""
from pathlib import Path
root=Path(__file__).resolve().parents[2]/'vendor/FACT'
p=root/'evaluation/robotwin/model2robotwin_interface.py'; s=p.read_text()
a='            response = self.client.inference(request)'
b='            request["_robotwin_seed"] = int(torch.randint(0, 2**31, (1,)).item())\n'+a
assert s.count(a)==1
p.write_text(s.replace(a,b))
p=root/'scripts/inference_server.py'; s=p.read_text()
a='        nonlocal request_counter'
b=a+'\n        seed = int(observation["_robotwin_seed"])\n        torch.manual_seed(seed)\n        torch.cuda.manual_seed_all(seed)\n        np.random.seed(seed)'
assert s.count(a)==1
s=s.replace(a,b).replace('base_seed = int(time.time_ns() & 0xFFFFFFFF)','base_seed = seed')
p.write_text(s)

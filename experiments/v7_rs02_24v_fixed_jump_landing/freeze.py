"""Explicit source-freeze utility; run before publication, never during training."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
parent = ROOT/'experiments/v7_rs02_24v_flat_landing_reward/FROZEN.json'
original = json.loads(parent.read_text(encoding='utf-8'))
files = dict(original['sha256'])
files[str(parent.relative_to(ROOT)).replace('\\','/')] = hashlib.sha256(parent.read_bytes()).hexdigest()
for path in sorted(HERE.iterdir()):
    if path.suffix in ('.py', '.ps1', '.md') and path.is_file():
        files[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
# Dependencies remain exactly those of the previously verified public release.
for rel, expected in original['sha256'].items():
    if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('Original frozen task changed: '+rel)
payload = dict(task='fixed_jump_continuous_landing_v1', parent_frozen_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
    contract=dict(control_hz=50, physics_hz=400, action_dim=7, actor_dim=157, critic_dim=189,
                  voltage_v=24, assist_strength=.625, train_updates=128, seed=104027,
                  launch='fixed deterministic model_0064; no action or optimizer path before per-world COM apex'),
    sha256=files)
(HERE/'FROZEN.json').write_text(json.dumps(payload, indent=2)+'\n', encoding='utf-8', newline='\n')
print(hashlib.sha256((HERE/'FROZEN.json').read_bytes()).hexdigest())

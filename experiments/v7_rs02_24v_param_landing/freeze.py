"""Freeze only this new experiment; do not rewrite either parent manifest."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
parent = HERE.parent/'v7_rs02_24v_fixed_jump_landing'
digest = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest = dict(task='fixed_jump_13d_parameter_landing_v1',
    parent_policy_sha256=digest(parent/'FROZEN.json'), parent_fast_sha256=digest(parent/'RESUME_FROZEN.json'),
    contract=dict(voltage_v=24, assist_strength=.625, physics_hz=400, control_hz=50,
        action_dim=13, observation_dim=17, action_samples_per_episode=1, sample_time_s=.6,
        activation='per-world confirmed COM apex', num_envs=512, updates=128, seed=104031,
        reward='original ten-term undiscounted landing score', reward_scale=.01,
        actor_lr=5e-5, critic_lr=3e-4, epochs=2, minibatches=4, max_global_kl=.03),
    sha256={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(HERE.iterdir())
            if p.is_file() and p.suffix in ('.py','.ps1','.md')})
(HERE/'FROZEN.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8', newline='\n')
print(digest(HERE/'FROZEN.json'))

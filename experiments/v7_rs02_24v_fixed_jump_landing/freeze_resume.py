"""Freeze the additive continuation without changing the original run's manifest."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FILES = ('fast_checks.py', 'fast_env.py', 'fast_report.py', 'fast_rollout.py',
         'resume_state.py', 'resume_train.py', 'resume_audit.py', 'run_resume.ps1',
         'verify_fast.py', 'freeze_resume.py', 'RESUME.md')
manifest = dict(task='fixed_jump_landing_512_resume_v1',
    policy_frozen_sha256=hashlib.sha256((HERE/'FROZEN.json').read_bytes()).hexdigest(),
    contract=dict(num_envs=512, total_updates=128, physics_hz=400, control_hz=50,
                  guard_frequency_unchanged=True, reward_and_ppo_unchanged=True),
    sha256={(HERE/name).relative_to(ROOT).as_posix():hashlib.sha256((HERE/name).read_bytes()).hexdigest()
            for name in FILES})
(HERE/'RESUME_FROZEN.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8', newline='\n')
print(hashlib.sha256((HERE/'RESUME_FROZEN.json').read_bytes()).hexdigest())

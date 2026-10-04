from pathlib import Path
import hashlib
import json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=dict(task='guided_parameter_cushion_v1',
    parent_sha256=digest(HERE.parent/'v7_rs02_24v_param_landing/FROZEN.json'),
    contract=dict(num_envs=512,updates=128,voltage_v=24,assist_strength=.625,
        action_dim=13,observation_dim=17,physics_hz=400,control_hz=50,
        source_checkpoint_sha256='393a58a7b1abaa442c11ae858b88f332e6ab64a3c86924bd64e1a338f361c713',
        source_update=120,seed=104042,force_target_n=300.,impact_multiplier=6.,
        immutable_controller=True,immutable_physics=True,proposal_count=20,
        selection='all worlds pass and retain height, then lower mean force',
        optimizer='new Adam and critic; inherited actor; proposal samples excluded from PPO'),
    sha256={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(HERE.iterdir())
            if p.is_file() and p.suffix in ('.py','.ps1','.md')})
(HERE/'FROZEN.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
print(digest(HERE/'FROZEN.json'))

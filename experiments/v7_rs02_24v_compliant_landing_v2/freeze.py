from pathlib import Path
import hashlib
import json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=dict(task='compliant_parameter_cushion_v2',
    predecessor_sha256=sha(HERE.parent/'v7_rs02_24v_compliant_landing/FROZEN.json'),
    parent_sha256=sha(HERE.parent/'v7_rs02_24v_guided_landing/FROZEN.json'),
    contract=dict(num_envs=512,updates=128,voltage_v=24,assist_strength=.625,
        action_dim=16,observation_dim=17,physics_hz=400,landing_reference_hz=400,
        fifo_delay_physics_ticks=[0,8],fifo_channels=15,source_update=80,
        source_checkpoint_sha256='647d6322268541be83874ce18c61ff87a88211456b23713c625757bf2aff9186',
        immutable_launch=True,immutable_physics=True,controller_revised=True,
        admission=dict(all_cases_pass=True,retain_height=True,min_com_stroke_m=.035,max_force_ratio_to_parent=.90),
        motor_support_force='actuated through joint torques and unchanged envelope; never an external vertical force',
        initialization='copy matching observation features and 8 attitude feedback outputs; new 8 controller parameters, critic and Adam'),
    sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(HERE.iterdir())
            if p.is_file() and p.suffix in ('.py','.ps1','.md')})
(HERE/'FROZEN.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
print(sha(HERE/'FROZEN.json'))

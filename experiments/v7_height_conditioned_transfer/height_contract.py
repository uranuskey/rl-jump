"""One controller expressed relative to a commanded leg height.

The old actor's motor corrections remain motor corrections. The reference moves
with the requested crouch; there is no competing fixed180mm controller and no
teacher/student output blending. Delay the reference AND corrections together.
"""
import hashlib
import json
import torch
from bindings import ROOT
from observations import KEEP, PRIVILEGED_FIELDS

VERSION = 'V7_HEIGHT_RELATIVE_MOTOR_ACTOR140_CRITIC172_V1'
ACTOR_DIM, CRITIC_DIM = 140, 172
UPPER, LOWER, X = .105, .145, -.052984004467725755
pose = json.loads((ROOT/'experiments/v7_exact_rotor_mjcf/targets.json').read_text())['poses']['180mm']
NOMINAL = tuple(pose['q_by_name'][s+'_'+j+'_joint'] for s in ('left', 'right') for j in ('hip', 'knee'))
SCHEMA = dict(version=VERSION, actor_dim=ACTOR_DIM, critic_dim=CRITIC_DIM,
    history_frames=4, raw_per_frame=35, actor_raw_indices=KEEP,
    actor_joint_positions='encoder joint minus IK(commanded leg height of this frame)',
    actor_action_history='previous bounded motor corrections, not foot coordinates',
    actor_other_sensors='unmodified measured gravity, gyro, joint/wheel velocity, causal body velocity',
    commands=['jump_height_div_0p05', 'jump_requested', 'request_age_div_6', 'leg_height_reference_div_0p25'],
    actions='6 Gaussian motor-correction/wheel latents; tanh once; 4 motor offsets at0.06rad and2wheel velocities at20rad/s',
    fifo='7 channels: six bounded corrections plus(height_reference-0.18)/0.03; one shared0..8physics-tick delay',
    servo='IK(arrived height reference)+arrived motor corrections; motor PD60/2 andwheel Kv0.1',
    critic='raw actual encoder relative fixed180mm plus commands and16 privileged fields, unchanged by actor transform',
    privileged_fields=PRIVILEGED_FIELDS, teacher_blend=False, old_height_limits_inherited=False,
    initialization='copy old448 actor; per-frame new command columns zero; critic/optimizer specified by runner')
SCHEMA_SHA256 = hashlib.sha256(json.dumps(SCHEMA, sort_keys=True).encode()).hexdigest()


def nominal_like(height):
    return torch.tensor(NOMINAL, dtype=height.dtype, device=height.device)


def reference_joints(height):
    """Positive knee branch, fixed nominal local foot x, both legs symmetric."""
    knee = torch.acos((X*X+height*height-UPPER*UPPER-LOWER*LOWER)/(2*UPPER*LOWER))
    hip = torch.atan2(torch.full_like(height, X), height)-torch.atan2(LOWER*torch.sin(knee), UPPER+LOWER*torch.cos(knee))
    q = torch.stack((hip, knee, hip, knee), -1)
    # Exact neutral branch preserves the historical float32 nominal at180mm.
    return torch.where((abs(height-.18) < 1e-7)[..., None], nominal_like(height), q)


def to_motor(q):
    p = q.reshape(*q.shape[:-1], 2, 2)
    return torch.stack((p[..., 0], p.sum(-1)), -1).reshape(*q.shape[:-1], 4)


def compose(raw, commands, private):
    n = raw.shape[0]
    if raw.shape != (n, 4, 35) or commands.shape != (n, 4, 4) or private.shape != (n, 16):
        raise ValueError('Height-conditioned observation shape mismatch')
    actor_raw = raw.clone()
    reference = reference_joints(commands[..., 3]*.25)
    actor_raw[..., 6:10] = raw[..., 6:10] + nominal_like(reference[..., 0]) - reference
    actor = torch.cat((actor_raw[..., KEEP], commands), -1).reshape(n, ACTOR_DIM)
    critic = torch.cat((torch.cat((raw, commands), -1).reshape(n, 156), private), -1)
    return actor.clone(), critic.clone()


def payload(bounded, height):
    if bounded.ndim != 2 or bounded.shape[1] != 6 or height.shape != bounded.shape[:1]:
        raise ValueError('Expected six corrections and one height per world')
    return torch.cat((bounded, ((height-.18)/.03)[:, None]), -1)


def servo(arrived, q, v):
    if arrived.ndim != 2 or arrived.shape[1] != 7:
        raise ValueError('Reference height must share the motor FIFO')
    height = .18+.03*arrived[:, 6]
    reference = reference_joints(height)
    motor_target = to_motor(reference)+.06*arrived[:, :4]
    paired = motor_target.reshape(-1, 2, 2)
    joint_target = torch.stack((paired[..., 0], paired[..., 1]-paired[..., 0]), -1).reshape(-1, 4)
    tm = 60*(motor_target-to_motor(q[:, :4]))-2*to_motor(v[:, :4])
    tm_pair = tm.reshape(-1, 2, 2)
    wheel_target = 20*arrived[:, 4:6]
    return dict(motor_target_rad=motor_target, joint_target_rad=joint_target,
                wheel_target_rad_s=wheel_target, motor_request=tm,
                joint_request=torch.stack((tm_pair.sum(-1), tm_pair[..., 1]), -1).reshape(-1, 4),
                wheel_request=.1*(wheel_target-v[:, 4:]), reference_height_m=height)

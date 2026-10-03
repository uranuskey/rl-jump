"""One command-aware actor projection across standing, crouching and jumping."""
import hashlib
import json
import math
import torch
import shared
from observations import KEEP, PRIVILEGED_FIELDS

VERSION = 'V7_BALANCE_CROUCH_JUMP_ACTOR140_CRITIC172_V1'
ACTOR_DIM, CRITIC_DIM = 140, 172
SCHEMA = dict(version=VERSION, actor_dim=ACTOR_DIM, critic_dim=CRITIC_DIM, action_dim=6,
    raw_per_frame=35, history_frames=4, actor_raw_indices=KEEP, privileged_fields=PRIVILEGED_FIELDS,
    commands=['jump_height_div_0p05', 'jump_requested', 'request_age_div_6', 'leg_height_reference_div_0p25'],
    actions='six Gaussian latents -> tanh once -> original foot target FIFO -> IK -> motor PD',
    initialization='fresh actor/critic/optimizer; no old policy or smoke checkpoint',
    normalization='disabled', curriculum_stages=['balance', 'crouch', 'jump'])
SCHEMA_SHA256 = hashlib.sha256(json.dumps(SCHEMA, sort_keys=True).encode()).hexdigest()


def leg_reference(t, stage):
    if stage == 'balance':
        return torch.full_like(t, .18)
    if stage == 'crouch':
        down = .5*(1-torch.cos(math.pi*(t-1.).clamp(0, 1)))
        up = .5*(1-torch.cos(math.pi*(t-3.5).clamp(0, 1)))
        return .18-.02*down+.02*up
    if stage == 'jump':
        down = .5*(1-torch.cos(math.pi*((t-.5)/.6).clamp(0, 1)))
        up = .5*(1-torch.cos(math.pi*((t-1.3)/.3).clamp(0, 1)))
        return .18-.02*down+.02*up
    raise ValueError(stage)


def compose(raw, commands, private):
    n = raw.shape[0]
    if raw.shape != (n, 4, 35) or commands.shape != (n, 4, 4) or private.shape != (n, 16):
        raise ValueError('Curriculum observation shape mismatch')
    actor = torch.cat((raw[:, :, KEEP], commands), -1).reshape(n, ACTOR_DIM)
    critic = torch.cat((torch.cat((raw, commands), -1).reshape(n, 156), private), -1)
    return actor.clone(), critic.clone()


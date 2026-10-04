"""Continue the verified actor; reset value/Adam because the objective changed."""
import guided_paths
import torch
from param_learning import Policy, PPO, FrozenLaunch
from param_runtime import sha

SOURCE_SHA='393a58a7b1abaa442c11ae858b88f332e6ab64a3c86924bd64e1a338f361c713'
SOURCE_FROZEN='bf14a91691f27c9af41d9341032df9a6fc04387bb81949341903c2bad98861e7'
EXPLORATION=(.18,.18,.22,.20,.10)+(.035,)*8


def load_parent(policy,path):
    if sha(path)!=SOURCE_SHA:
        raise RuntimeError('Expected the audited parent update-120 checkpoint')
    state=torch.load(path,map_location='cpu',weights_only=True)
    assert state['update']==120 and state['frozen_sha256']==SOURCE_FROZEN
    assert state['voltage_v']==24 and state['assist_strength']==.625
    policy.load_state_dict(state['model_state_dict'],strict=True)
    # Critic and optimizer are deliberately new; this is not bitwise continuation.
    for layer in policy.critic:
        if hasattr(layer,'reset_parameters'):
            layer.reset_parameters()
    set_exploration(policy,0)


def set_exploration(policy,update):
    scale=max(.45,1-update/192)
    value=policy.std.new_tensor(EXPLORATION)
    value[:5]*=scale
    policy.std.copy_(value)


def apply_offset(policy,offset):
    with torch.no_grad():
        policy.actor[-1].bias.add_(policy.std.new_tensor(offset))

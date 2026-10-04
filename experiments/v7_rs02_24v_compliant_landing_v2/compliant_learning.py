"""New action contract; inherit only matching observation features and feedback."""
import compliant_paths
import torch
from torch import nn
from torch.distributions import Normal
from param_learning import PPO, FrozenLaunch
from compliant_control import initial_raw, encode
from guided_runtime import sha

SOURCE_SHA = '647d6322268541be83874ce18c61ff87a88211456b23713c625757bf2aff9186'
SOURCE_FROZEN = 'dbd6c361f2ae6fe6fe1d72060b249dd3120d402bca7801c32bcdbb8272acdf8b'


class Policy(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(17,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,16))
        self.critic = nn.Sequential(nn.Linear(17,64),nn.Tanh(),nn.Linear(64,1))
        nn.init.zeros_(self.actor[-1].weight)
        with torch.no_grad():
            self.actor[-1].bias.copy_(initial_raw())
        self.register_buffer('std',torch.tensor((.12,)*8+(.035,)*8))
        self.to(device)
    def distribution(self,obs):
        return Normal(self.actor(obs),self.std.expand(len(obs),-1))


def source(path):
    assert sha(path)==SOURCE_SHA
    state=torch.load(path,map_location='cpu',weights_only=True)
    assert state['update']==80 and state['frozen_sha256']==SOURCE_FROZEN
    assert state['voltage_v']==24 and state['assist_strength']==.625
    return state


def initialize(policy,path,parameters=None):
    old=source(path)['model_state_dict']
    with torch.no_grad():
        for i in (0,2):
            policy.actor[i].weight.copy_(old[f'actor.{i}.weight'])
            policy.actor[i].bias.copy_(old[f'actor.{i}.bias'])
        policy.actor[-1].weight[:8].zero_()
        policy.actor[-1].weight[8:].copy_(old['actor.4.weight'][5:])
        policy.actor[-1].bias[8:].copy_(old['actor.4.bias'][5:])
        if parameters is not None:
            policy.actor[-1].bias[:8].copy_(encode(parameters,policy.std.device))


def set_exploration(policy,update):
    values=policy.std.new_tensor((.12,)*8+(.035,)*8)
    values[:8]*=max(.55,1-update/256)
    policy.std.copy_(values)

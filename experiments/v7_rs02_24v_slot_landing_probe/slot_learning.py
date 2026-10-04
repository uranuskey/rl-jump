"""Embed the checked air-minus-5 mm transform inside actor for correct PPO ratios."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'v7_rs02_24v_compliant_landing_v3'))
import compliant_paths
import torch
from torch import nn
from compliant_learning import Policy as ParentPolicy,PPO,FrozenLaunch
from compliant_control import decode,encode,LOW,HIGH


class AirShift(nn.Module):
    def forward(self,raw):
        values,_=decode(raw)
        values=values.clone()
        values[:,0]=(values[:,0]-.005).clamp(LOW[0]+1e-6,HIGH[0]-1e-6)
        return torch.cat((encode(values,raw.device),raw[:,8:]),1)


class Policy(ParentPolicy):
    def __init__(self,device='cpu'):
        super().__init__(device)
        self.actor.add_module('air_shift',AirShift())


def exploration(policy,update):
    t=min(1.,max(0.,update/128))
    policy.std.copy_(policy.std.new_tensor((.06-.025*t,)*8+(.020-.008*t,)*8))

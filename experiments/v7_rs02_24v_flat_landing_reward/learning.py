"""Episodic PPO of launch, landing, and causal feedback gains."""
import bootstrap
import torch
from torch import nn
from torch.distributions import Normal
from plan_policy import Policy as LaunchPolicy, PPO
from plan_contract import decode
from bootstrap import HERE,ROOT

CHECKPOINT=ROOT/'models/launch_0064.pt'
LAND_LOW=(.165,.12,.135,.08,.25)
LAND_HIGH=(.205,.35,.18,.25,.8)
LAND_INITIAL=(.18,.22,.16,.14,.45)
LEVELS=(.625,) # Flat-ground cushioning only; no assistance withdrawal in this task.

def landing_raw():
    return torch.logit((torch.tensor(LAND_INITIAL)-torch.tensor(LAND_LOW))/(torch.tensor(LAND_HIGH)-torch.tensor(LAND_LOW)))

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        old=LaunchPolicy()
        old.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=False)['model_state_dict'])
        self.launch=old.actor
        for p in self.launch.parameters():p.requires_grad_(False)
        self.delta=nn.Sequential(nn.Linear(17,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,22))
        nn.init.zeros_(self.delta[-1].weight);nn.init.zeros_(self.delta[-1].bias)
        self.register_buffer('tail',torch.cat((landing_raw(),torch.zeros(8))))
    def forward(self,obs):
        old_obs=obs.clone();old_obs[:,-1]=1.
        base=torch.cat((self.launch(old_obs),self.tail.expand(len(obs),-1)),1)
        return base+self.delta(obs)

class Policy(nn.Module):
    def __init__(self,device='cpu'):
        super().__init__();self.actor=Actor();self.critic=nn.Sequential(nn.Linear(17,64),nn.Tanh(),nn.Linear(64,1))
        self.register_buffer('std',torch.tensor([.035]*9+[.05]*5+[.035]*8));self.to(device)
    def distribution(self,obs):return Normal(self.actor(obs),self.std.expand(len(obs),-1))

def apply_action(env,action):
    env.plan.copy_(decode(action[:,:9]))
    lo=torch.tensor(LAND_LOW,device=action.device);hi=torch.tensor(LAND_HIGH,device=action.device)
    env.landing_parameters.copy_(lo+(hi-lo)*action[:,9:14].sigmoid())
    env.feedback.copy_(action[:,14:])

def score(task,ticks,stable_seconds):
    return task.landing_reward.score(task,ticks)

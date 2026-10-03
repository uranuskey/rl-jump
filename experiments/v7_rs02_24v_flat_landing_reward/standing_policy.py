"""Shared rollout/PPO/KL/inference actor: frozen seed plus learned feedback and curve."""
import bootstrap
import torch
from torch import nn
from torch.distributions import Normal
from rsl_rl.modules import ActorCritic
from seed_policy import JumpPolicy as SeedPolicy,MatchedKLGuard
from bootstrap import TRAINED_CHECKPOINT,CURVE_CHECKPOINT
from curve_contract import ACTOR_DIM,CRITIC_DIM,ACTION_DIM,seed_input


def gate(obs):
    return ((.5+10*obs[:,138]-4.)/.25).clamp(0,1)[:,None]


class CurveActor(nn.Module):
    def __init__(self,seed):
        super().__init__()
        self.base=seed
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.correction=nn.Sequential(nn.Linear(ACTOR_DIM,96),nn.ELU(),nn.Linear(96,64),nn.ELU(),nn.Linear(64,ACTION_DIM))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self,obs):
        delta=self.correction(obs)
        amplitude=gate(obs)
        feedback=self.base(seed_input(obs))+.5*amplitude*torch.tanh(delta[:,:6])
        return torch.cat((feedback,amplitude*delta[:,6:]),1)


class JumpPolicy(ActorCritic):
    def __init__(self,device='cpu'):
        super().__init__(ACTOR_DIM,CRITIC_DIM,ACTION_DIM,actor_hidden_dims=[96,64],critic_hidden_dims=[128,64],
                         activation='elu',init_noise_std=.04)
        seed=SeedPolicy('cpu')
        ck=torch.load(TRAINED_CHECKPOINT,map_location='cpu',weights_only=False)
        seed.load_state_dict(ck['model_state_dict'],strict=True)
        self.actor=CurveActor(seed.actor)
        prior=seed.critic.state_dict()
        current=self.critic.state_dict()
        current['0.weight'].zero_()
        current['0.weight'][:,:172]=prior['0.weight']
        # Preserve seed value at initialization despite3cm->5cm command change.
        current['0.bias']=prior['0.bias']-.4*prior['0.weight'][:,[35,74,113,152]].sum(1)
        for key in current:
            if key not in ('0.weight','0.bias'):
                current[key]=prior[key]
        self.critic.load_state_dict(current)
        # Expand the retained 144/176 policy without changing its initial action.
        learned=torch.load(CURVE_CHECKPOINT,map_location='cpu',weights_only=False)['model_state_dict']
        migrated={k:v.clone() for k,v in learned.items()}
        for key,width,goal_slots,bias in (
            ('actor.correction.0.weight',ACTOR_DIM,[31,66,101,136],'actor.correction.0.bias'),
            ('critic.0.weight',CRITIC_DIM,[35,74,113,152],'critic.0.bias')):
            w=migrated[key]
            expanded=torch.zeros(w.shape[0],width)
            expanded[:,:w.shape[1]]=w
            migrated[key]=expanded
            migrated[bias]=migrated[bias]-2*w[:,goal_slots].sum(1)
        self.load_state_dict(migrated,strict=True)
        self.std.requires_grad_(False)
        with torch.no_grad():
            self.std.copy_(torch.tensor([.025]*6+[.10,.08,.08]))
        # Continue the measured best assisted actor, with a fresh optimizer.
        from bootstrap import COMPARE_CHECKPOINT
        self.load_state_dict(torch.load(COMPARE_CHECKPOINT,map_location='cpu',weights_only=False)['model_state_dict'],strict=True)
        self.to(device)

    def std_for(self,obs):
        return (1e-4+gate(obs)*self.std).expand(-1,ACTION_DIM)

    def update_distribution(self,observations):
        self.distribution=Normal(self.actor(observations),self.std_for(observations))

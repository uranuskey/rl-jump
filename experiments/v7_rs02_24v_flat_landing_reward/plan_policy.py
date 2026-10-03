"""Episodic PPO: one action is one whole-jump plan, scored at the COM apex."""
import copy
import torch
from torch import nn
from torch.distributions import Normal
from plan_contract import initial_raw

class Policy(nn.Module):
    def __init__(self,device='cpu'):
        super().__init__()
        self.actor=nn.Sequential(nn.Linear(17,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,9))
        self.critic=nn.Sequential(nn.Linear(17,64),nn.Tanh(),nn.Linear(64,1))
        nn.init.zeros_(self.actor[-1].weight)
        with torch.no_grad():self.actor[-1].bias.copy_(initial_raw())
        self.register_buffer('std',torch.full((9,),.65))
        self.to(device)
    def distribution(self,obs):return Normal(self.actor(obs),self.std.expand(len(obs),-1))

class PPO:
    def __init__(self,policy):
        self.policy=policy
        self.actor_optimizer=torch.optim.Adam(policy.actor.parameters(),lr=.001)
        self.critic_optimizer=torch.optim.Adam(policy.critic.parameters(),lr=.001)
    def update(self,obs,actions,reward,eligible):
        ids=eligible.nonzero().flatten()
        # No optimizer call at all: Adam moments and parameters both stay exact.
        if not len(ids):return dict(actor_steps=0,critic_steps=0,eligible_samples=0,skipped='no_executed_plan')
        obs,actions,reward=obs[ids],actions[ids],reward[ids]
        p=self.policy
        with torch.no_grad():
            old_mean=p.actor(obs).clone()
            old_log=Normal(old_mean,p.std).log_prob(actions).sum(-1)
            adv=reward-p.critic(obs).flatten()
            spread=adv.std(unbiased=False)
            adv=(adv-adv.mean())/(spread+1e-8)
        accepted=rejected=critic_steps=0
        max_kl=0.
        for epoch in range(2):
            for batch in torch.randperm(len(obs),device=obs.device).tensor_split(4):
                if not len(batch):continue
                self.critic_optimizer.zero_grad(set_to_none=True)
                value_loss=(p.critic(obs[batch]).flatten()-reward[batch]).square().mean()
                value_loss.backward()
                nn.utils.clip_grad_norm_(p.critic.parameters(),1.)
                self.critic_optimizer.step();critic_steps+=1
                if float(spread)<1e-8:continue
                old_params=[v.detach().clone() for v in p.actor.parameters()]
                old_state=copy.deepcopy(self.actor_optimizer.state_dict())
                self.actor_optimizer.zero_grad(set_to_none=True)
                ratio=(p.distribution(obs[batch]).log_prob(actions[batch]).sum(-1)-old_log[batch]).exp()
                loss=torch.maximum(-adv[batch]*ratio,-adv[batch]*ratio.clamp(.8,1.2)).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(p.actor.parameters(),1.)
                self.actor_optimizer.step()
                with torch.no_grad():
                    kl=float((((p.actor(obs)-old_mean)/p.std).square().sum(-1)*.5).mean())
                    if not torch.isfinite(torch.tensor(kl)) or kl>.03:
                        for v,old in zip(p.actor.parameters(),old_params):v.copy_(old)
                        self.actor_optimizer.load_state_dict(old_state);rejected+=1
                    else:accepted+=1;max_kl=max(max_kl,kl)
        assert all(bool(torch.isfinite(v).all()) for v in p.parameters())
        return dict(actor_steps=accepted,critic_steps=critic_steps,eligible_samples=len(ids),
                    rejected_actor_steps=rejected,max_accepted_kl=max_kl,
                    mean_return=float(reward.mean()),return_std=float(reward.std(unbiased=False)))

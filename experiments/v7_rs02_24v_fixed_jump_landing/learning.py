"""Masked trajectory PPO; the frozen launch is not part of either optimizer."""
import bootstrap
import copy
import torch
from torch import nn
from torch.distributions import Normal
from bootstrap import ROOT
from control import ACTOR_DIM, CRITIC_DIM, ACTION_DIM, ACTION_STD
from plan_policy import Policy as LaunchPolicy
from plan_contract import decode

GAMMA, LAMBDA, REWARD_SCALE = .995, .95, .01


class FrozenLaunch(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        old = LaunchPolicy()
        old.load_state_dict(torch.load(ROOT/'models/launch_0064.pt', map_location='cpu',
                                      weights_only=True)['model_state_dict'])
        self.actor = old.actor
        self.requires_grad_(False)
        self.to(device).eval()

    @torch.no_grad()
    def forward(self, obs):
        old_obs = obs.clone()
        old_obs[:, -1] = 1.
        return decode(self.actor(old_obs))


class Policy(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(ACTOR_DIM, 128), nn.Tanh(), nn.Linear(128, 64),
                                   nn.Tanh(), nn.Linear(64, ACTION_DIM))
        self.critic = nn.Sequential(nn.Linear(CRITIC_DIM, 128), nn.Tanh(), nn.Linear(128, 64),
                                    nn.Tanh(), nn.Linear(64, 1))
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        self.register_buffer('std', torch.tensor(ACTION_STD))
        self.to(device)

    def distribution(self, obs):
        return Normal(self.actor(obs), self.std.expand(len(obs), -1))


def advantages(rewards, values, next_values, done, valid, gamma=GAMMA, lam=LAMBDA):
    advantage = torch.zeros_like(rewards)
    tail = torch.zeros_like(rewards[0])
    for t in range(len(rewards)-1, -1, -1):
        continuing = (~done[t]).to(rewards.dtype)
        delta = rewards[t]+gamma*continuing*next_values[t]-values[t]
        tail = torch.where(valid[t], delta+gamma*lam*continuing*tail, 0.)
        advantage[t] = tail
    return advantage, advantage+values


class PPO:
    def __init__(self, policy):
        self.policy = policy
        self.actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=3e-5)
        self.critic_optimizer = torch.optim.Adam(policy.critic.parameters(), lr=3e-4)

    def update(self, rows):
        valid = rows['valid']
        count = int(valid.sum())
        if not count:
            return dict(actor_steps=0, critic_steps=0, eligible_samples=0, rejected_actor_steps=0,
                        max_accepted_kl=0., skipped='no_landing_transition')
        with torch.no_grad():
            adv, returns = advantages(rows['reward']*REWARD_SCALE, rows['value'],
                                       rows['next_value'], rows['done'], valid)
            adv, returns = adv[valid], returns[valid]
            adv = (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
        obs, critic = rows['obs'][valid], rows['critic'][valid]
        actions, logp, mean = rows['action'][valid], rows['logp'][valid], rows['mean'][valid]
        accepted = rejected = critic_steps = 0
        max_kl = 0.
        for _ in range(2):
            for batch in torch.randperm(count, device=obs.device).tensor_split(min(16, count)):
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss_v = (self.policy.critic(critic[batch]).flatten()-returns[batch]).square().mean()
                loss_v.backward()
                nn.utils.clip_grad_norm_(self.policy.critic.parameters(), 1.)
                self.critic_optimizer.step()
                critic_steps += 1
                old_params = [p.detach().clone() for p in self.policy.actor.parameters()]
                old_state = copy.deepcopy(self.actor_optimizer.state_dict())
                self.actor_optimizer.zero_grad(set_to_none=True)
                dist = self.policy.distribution(obs[batch])
                ratio = (dist.log_prob(actions[batch]).sum(-1)-logp[batch]).exp()
                loss_a = torch.maximum(-adv[batch]*ratio, -adv[batch]*ratio.clamp(.8, 1.2)).mean()
                loss_a.backward()
                nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 1.)
                self.actor_optimizer.step()
                with torch.no_grad():
                    kl = float((.5*((self.policy.actor(obs[batch])-mean[batch])/self.policy.std).square().sum(-1)).mean())
                if not torch.isfinite(torch.tensor(kl)) or kl>.02:
                    with torch.no_grad():
                        for p, previous in zip(self.policy.actor.parameters(), old_params):
                            p.copy_(previous)
                    self.actor_optimizer.load_state_dict(old_state)
                    rejected += 1
                else:
                    accepted += 1
                    max_kl = max(max_kl, kl)
        if not all(bool(torch.isfinite(p).all()) for p in self.policy.parameters()):
            raise RuntimeError('Non-finite PPO parameter')
        return dict(actor_steps=accepted, critic_steps=critic_steps, eligible_samples=count,
                    rejected_actor_steps=rejected, max_accepted_kl=max_kl)

"""Episodic PPO learns only five landing parameters and eight feedback gains."""
import path_setup
import torch
from torch import nn
from torch.distributions import Normal
from learning import FrozenLaunch
from plan_policy import PPO as EpisodicPPO
from param_control import ACTION_DIM, OBS_DIM, STD, initial_raw


class Policy(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(OBS_DIM, 64), nn.Tanh(), nn.Linear(64, 64),
                                   nn.Tanh(), nn.Linear(64, ACTION_DIM))
        self.critic = nn.Sequential(nn.Linear(OBS_DIM, 64), nn.Tanh(), nn.Linear(64, 1))
        nn.init.zeros_(self.actor[-1].weight)
        with torch.no_grad():
            self.actor[-1].bias.copy_(initial_raw())
        self.register_buffer('std', torch.tensor(STD))
        self.to(device)

    def distribution(self, obs):
        return Normal(self.actor(obs), self.std.expand(len(obs), -1))


class PPO(EpisodicPPO):
    def __init__(self, policy):
        super().__init__(policy)
        for group in self.actor_optimizer.param_groups:
            group['lr'] = 5e-5
        for group in self.critic_optimizer.param_groups:
            group['lr'] = 3e-4

    def update(self, obs, action, reward, eligible):
        return super().update(obs, action, reward*.01, eligible)

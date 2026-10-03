"""Frozen height-relative balance feedback plus a phase-gated trainable correction.

The same forward path and phase-dependent Gaussian are used for rollout, PPO,
KL evaluation, and inference. No privileged contact or simulation phase enters actor.
"""
import bootstrap
import torch
from torch import nn
from torch.distributions import Normal
from rsl_rl.modules import ActorCritic
from bootstrap import CHECKPOINT
from kl_guard import RolloutKLGuard


def gate(obs):
    # Last-frame command age is (time - request_s) / episode_s, both frozen below.
    time = .5+10*obs[:, 138]
    return ((time-4.)/.25).clamp(0, 1)[:, None]


class GatedActor(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.correction = nn.Sequential(nn.Linear(140, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 6))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, obs):
        return self.base(obs)+gate(obs)*.5*torch.tanh(self.correction(obs))


class JumpPolicy(ActorCritic):
    def __init__(self, device='cpu'):
        super().__init__(140, 172, 6, actor_hidden_dims=[128, 64], critic_hidden_dims=[128, 64],
                         activation='elu', init_noise_std=.04)
        source = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)['model_state_dict']
        self.load_state_dict(source)
        self.actor = GatedActor(self.actor)
        # Inherited large hip noise is unsuitable for an already balanced prefix.
        self.std.requires_grad_(False)
        with torch.no_grad():
            self.std.fill_(.04)
        self.to(device)

    def std_for(self, obs):
        # A tiny positive floor keeps Gaussian likelihoods finite; mean is exact old feedback before 4 s.
        return (1e-4+gate(obs)*self.std).expand(-1, 6)

    def update_distribution(self, observations):
        self.distribution = Normal(self.actor(observations), self.std_for(observations))


class MatchedKLGuard(RolloutKLGuard):
    @torch.no_grad()
    def kl(self):
        mu, sd = self.policy.actor(self.obs), self.policy.std_for(self.obs)
        if not bool(torch.isfinite(mu).all() and torch.isfinite(sd).all() and (sd > 0).all()):
            return float('inf')
        kl = (torch.log(sd/self.std)+(self.std.square()+(self.mean-mu).square())/(2*sd.square())-.5).sum(-1)
        # The frozen prefix must not dilute the KL bound for the learning phase.
        scope = gate(self.obs).flatten() > 0
        return float(kl[scope].mean()) if bool(scope.any()) else float(kl.mean())

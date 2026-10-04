"""KL-bounded episodic PPO with actor-step backtracking and exact Adam rollback."""
import copy
import torch
from torch import nn
from torch.distributions import Normal


class PPO:
    def __init__(self, policy, *, actor_lr=5e-5, attempts=7):
        self.policy = policy
        self.actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(policy.critic.parameters(), lr=3e-4)
        self.max_actor_lr, self.attempts = actor_lr, attempts

    def update(self, obs, actions, reward, eligible):
        ids = eligible.nonzero().flatten()
        if not len(ids):
            return dict(actor_steps=0,critic_steps=0,eligible_samples=0,skipped='no_executed_plan')
        obs, actions, reward = obs[ids], actions[ids], reward[ids]*.01
        p = self.policy
        with torch.no_grad():
            old_mean = p.actor(obs).clone()
            old_log = Normal(old_mean,p.std).log_prob(actions).sum(-1)
            advantages = reward-p.critic(obs).flatten()
            spread = advantages.std(unbiased=False)
            advantages = (advantages-advantages.mean())/(spread+1e-8)
        accepted = rejected = critic_steps = 0
        max_accepted_kl = max_rejected_kl = 0.
        accepted_lrs = []
        for epoch in range(2):
            for batch in torch.randperm(len(obs),device=obs.device).tensor_split(4):
                if not len(batch):
                    continue
                self.critic_optimizer.zero_grad(set_to_none=True)
                value_loss = (p.critic(obs[batch]).flatten()-reward[batch]).square().mean()
                value_loss.backward()
                nn.utils.clip_grad_norm_(p.critic.parameters(),1.)
                self.critic_optimizer.step()
                critic_steps += 1
                if float(spread)<1e-8:
                    continue
                parameters = [v.detach().clone() for v in p.actor.parameters()]
                optimizer_state = copy.deepcopy(self.actor_optimizer.state_dict())
                original_lr = self.actor_optimizer.param_groups[0]['lr']
                accepted_this_batch = False
                for attempt in range(self.attempts):
                    # Every retry starts from the same pre-step parameters/moments.
                    with torch.no_grad():
                        for v, old in zip(p.actor.parameters(),parameters):
                            v.copy_(old)
                    self.actor_optimizer.load_state_dict(copy.deepcopy(optimizer_state))
                    trial_lr = original_lr*(.5**attempt)
                    for group in self.actor_optimizer.param_groups:
                        group['lr'] = trial_lr
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    ratio = (p.distribution(obs[batch]).log_prob(actions[batch]).sum(-1)-old_log[batch]).exp()
                    loss = torch.maximum(-advantages[batch]*ratio,
                                         -advantages[batch]*ratio.clamp(.8,1.2)).mean()
                    loss.backward()
                    nn.utils.clip_grad_norm_(p.actor.parameters(),1.)
                    self.actor_optimizer.step()
                    with torch.no_grad():
                        kl = float((((p.actor(obs)-old_mean)/p.std).square().sum(-1)*.5).mean())
                    if __import__('math').isfinite(kl) and kl<=.03:
                        accepted += 1
                        max_accepted_kl = max(max_accepted_kl,kl)
                        accepted_lrs.append(trial_lr)
                        accepted_this_batch = True
                        break
                    rejected += 1
                    max_rejected_kl = max(max_rejected_kl,kl)
                if not accepted_this_batch:
                    with torch.no_grad():
                        for v, old in zip(p.actor.parameters(),parameters):
                            v.copy_(old)
                    self.actor_optimizer.load_state_dict(copy.deepcopy(optimizer_state))
                    # No attempt was accepted: retain exact pre-batch Adam state.
        assert all(bool(torch.isfinite(v).all()) for v in p.parameters())
        # Match future optimizer step size to the measured rollout-level KL.
        current_lr = self.actor_optimizer.param_groups[0]['lr']
        factor = (2/3 if max_accepted_kl>.02 else 1.5 if max_accepted_kl<.005 else 1.)
        next_lr = max(1e-7,min(self.max_actor_lr,current_lr*factor)) if accepted else max(1e-7,current_lr*.5)
        for group in self.actor_optimizer.param_groups:
            group['lr'] = next_lr
        return dict(actor_steps=accepted,critic_steps=critic_steps,eligible_samples=len(ids),
            rejected_actor_steps=rejected,max_accepted_kl=max_accepted_kl,max_rejected_kl=max_rejected_kl,
            accepted_actor_lr_min=min(accepted_lrs,default=0.),actor_lr_next=next_lr,
            mean_return=float(reward.mean()),return_std=float(reward.std(unbiased=False)))

"""Behavioral checks for continuous control, masked credit assignment and provenance."""
import bootstrap
import json
from pathlib import Path
import sys
import torch
import mujoco
from bootstrap import ROOT, HERE
from control import advance, ACTION_DIM, ACTOR_DIM, CRITIC_DIM, DT
from learning import Policy, FrozenLaunch, PPO, advantages
from runtime import verify
import test_reward


def main():
    torch.set_num_threads(1)
    torch.manual_seed(410)
    checks = list(test_reward.main())
    raw = torch.tensor([[3., -3., 1., -1., 2., -2., 4.], [-3., 3., -1., 1., -2., 2., -4.]])
    gate = torch.tensor([False, True])
    offset, height, velocity, correction, effective = advance(raw, gate, torch.zeros(2),
        torch.full((2,), .18), torch.zeros(2))
    assert offset[0]==0 and height[0]==.18 and velocity[0]==0
    assert not correction[0].any() and not effective[0].any()
    assert offset[1]!=0 and correction[1].any()
    checks.append('mixed worlds: late controller cannot act before its own apex')
    off, h, v, action, _ = advance(torch.zeros(2, ACTION_DIM), torch.ones(2, dtype=torch.bool),
        torch.zeros(2), torch.tensor([.22, .16]), torch.tensor([-.4, .1]))
    assert torch.equal(h, torch.tensor([.22, .16])) and torch.equal(v, torch.tensor([-.4, .1]))
    assert not off.any() and not action.any()
    checks.append('zero action retains the validated reference exactly')
    _, h, v, _, _ = advance(raw, torch.ones(2, dtype=torch.bool), torch.tensor([.1, -.1]),
        torch.full((2,), .18), torch.zeros(2))
    assert bool(((h>=.095-1e-7)&(h<=.225+1e-7)).all())
    checks.append('extreme actions cannot request height outside the declared workspace')
    rewards = torch.tensor([[99., 1.], [2., 3.], [5., 7.]])
    valid = torch.tensor([[False, True], [True, True], [True, False]])
    done = torch.tensor([[False, False], [False, True], [True, True]])
    values = torch.zeros_like(rewards)
    adv, ret = advantages(rewards, values, torch.full_like(values, 123.), done, valid, gamma=0., lam=1.)
    assert torch.equal(adv, torch.tensor([[0., 1.], [2., 3.], [5., 0.]]))
    adv, ret = advantages(rewards, values, values, done, valid, gamma=1., lam=1.)
    assert torch.equal(adv, torch.tensor([[0., 4.], [7., 3.], [5., 0.]]))
    checks.append('GAE respects per-world first landing decision, terminal boundary and invalid rows')
    launch, policy = FrozenLaunch(), Policy()
    plans = launch(torch.randn(4, 17))
    old = {k:v.clone() for k,v in launch.state_dict().items()}
    assert plans.shape==(4, 9) and not any(p.requires_grad for p in launch.parameters())
    obs = torch.randn(3, 4, ACTOR_DIM)
    critic = torch.randn(3, 4, CRITIC_DIM)
    with torch.no_grad():
        dist = policy.distribution(obs.reshape(-1, ACTOR_DIM))
        action = dist.sample().reshape(3, 4, ACTION_DIM)
        batch = dict(obs=obs, critic=critic, action=action,
            mean=dist.mean.reshape_as(action), logp=dist.log_prob(action.reshape(-1, ACTION_DIM)).sum(-1).reshape(3, 4),
            value=policy.critic(critic).squeeze(-1), next_value=torch.zeros(3,4),
            done=torch.tensor([[False]*4, [False]*4, [True]*4]),
            valid=torch.tensor([[False,True,False,True], [True]*4, [True]*4]), reward=torch.randn(3,4))
    ppo = PPO(policy)
    actor_before = [p.clone() for p in policy.actor.parameters()]
    stats = ppo.update(batch)
    assert stats['eligible_samples']==10 and stats['actor_steps']>0
    assert any(not torch.equal(p,q) for p,q in zip(actor_before,policy.actor.parameters()))
    assert all(torch.equal(v,old[k]) for k,v in launch.state_dict().items())
    checks.append('real PPO update uses only post-apex rows and leaves the separate launch untouched')
    optim_before = {k:v.clone() for k,v in policy.state_dict().items()}
    batch['valid'].zero_()
    assert ppo.update(batch)['actor_steps']==0
    assert all(torch.equal(v,optim_before[k]) for k,v in policy.state_dict().items())
    checks.append('no landing transitions means no optimizer update')
    model = mujoco.MjModel.from_xml_path(str(ROOT/'experiments/v7_mujoco_training_gate/v7_full_collision.xml'))
    for module in list(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if file and 'serial_wheel_leg_rl' in str(file) and '.venv' not in str(file):
            raise RuntimeError('Borrowed original source: '+str(file))
    result = dict(status='PASS_OFFLINE_ONLY', frozen_sha256=verify(), checks=checks,
        actor_dim=ACTOR_DIM, critic_dim=CRITIC_DIM, action_dim=ACTION_DIM,
        model_nq=model.nq, model_nv=model.nv, control_dt_s=DT)
    print(json.dumps(result, indent=2))


if __name__=='__main__':
    main()

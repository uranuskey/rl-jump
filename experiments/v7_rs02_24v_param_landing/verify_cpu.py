"""Behavioral checks: fixed launch, bounded episode parameters, episodic PPO."""
import copy
import json
from types import SimpleNamespace as NS
import path_setup
import torch
from control import reference as old_reference
from param_control import ACTION_DIM, LOW, HIGH, decode, initial_raw, prepare, reference
from param_env import ParameterEnv
from param_learning import Policy, PPO, FrozenLaunch
from param_runtime import verify
import test_reward


def main():
    torch.set_num_threads(1)
    torch.manual_seed(41031)
    checks = list(test_reward.main())
    parameters, gains = decode(initial_raw().expand(16, -1))
    assert parameters.shape==(16,5) and not gains.any()
    plan = torch.zeros(16,9)
    plan[:, 3] = .22
    time = torch.linspace(1, 3.5, 16)
    start = torch.full((16,),1.)
    touchdown = torch.where(time>1.2,1.2,0.)
    expected = old_reference(plan, time, start, touchdown)
    actual = reference(parameters, plan, time, start, touchdown)
    assert all(torch.allclose(a,b,atol=1e-7,rtol=1e-6) for a,b in zip(actual,expected))
    checks.append('initial curve and zero gains reproduce the validated landing reference')
    raw = torch.randn(16,13)*100
    parameters, gains = decode(raw)
    assert ((parameters>=torch.tensor(LOW)) & (parameters<=torch.tensor(HIGH))).all()
    assert gains.abs().max()<=4
    h, v = reference(parameters, plan, torch.full((16,),4.), start, torch.full((16,),1.2))
    assert torch.allclose(h,torch.full((16,),.18),atol=1e-7) and not v.any()
    checks.append('extreme parameters remain bounded and completed recovery targets 18 cm')
    apex = torch.arange(16)%2==0
    terminal = torch.arange(16)%3==0
    ticks = torch.full((16,),500)
    values = (ticks,apex,terminal,torch.full((16,),-1),torch.zeros(16),torch.zeros(16,4),
              torch.randn(16,6),torch.randn(16,4),torch.zeros(16),torch.randn(16,3),
              torch.randn(16,3),torch.randn(16,3),torch.randn(16,2))
    snapshots = [value.clone() for value in values]
    a = prepare(plan,initial_raw().expand(16,-1),*values)
    b = prepare(plan,raw,*values)
    assert all(torch.equal(x,y) for x,y in zip(values,snapshots))
    assert all(torch.equal(a[key][~apex],b[key][~apex]) for key in a)
    assert not b['effective_action'][~(apex&~terminal)].any()
    assert not torch.equal(a['actions'][apex&~terminal],b['actions'][apex&~terminal])
    checks.append('mixed-world apex and terminal gates isolate all parameter effects without mutating inputs')
    mock = NS(n=16, parameter_action=torch.zeros(16,13),locked_parameters=torch.zeros(16,13),
              parameters_locked=torch.zeros(16,dtype=torch.bool))
    ParameterEnv.lock_parameters(mock,raw)
    ParameterEnv.assert_parameters_fixed(mock)
    try:
        ParameterEnv.lock_parameters(mock,raw)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Second parameter sample accepted')
    mock.parameter_action[0,0] += 1
    try:
        ParameterEnv.assert_parameters_fixed(mock)
    except RuntimeError:
        pass
    else:
        raise AssertionError('In-episode mutation accepted')
    checks.append('exactly one locked parameter sample per episode; mutations are rejected')
    launch, policy = FrozenLaunch(), Policy()
    ppo = PPO(policy)
    frozen = copy.deepcopy(launch.state_dict())
    ids = {id(p) for group in ppo.actor_optimizer.param_groups for p in group['params']}
    assert not ids.intersection(id(p) for p in launch.parameters())
    obs = torch.randn(32,17)
    with torch.no_grad():
        action = policy.distribution(obs).sample()
    before = copy.deepcopy(policy.actor.state_dict())
    eligible = torch.arange(32)%2==0
    stats = ppo.update(obs,action,torch.linspace(-120,120,32),eligible)
    assert stats['eligible_samples']==16 and stats['actor_steps']>0
    assert any(not torch.equal(value,before[key]) for key,value in policy.actor.state_dict().items())
    assert all(torch.equal(value,frozen[key]) for key,value in launch.state_dict().items())
    assert policy.actor(obs).shape==(32,ACTION_DIM)
    before = copy.deepcopy(policy.state_dict())
    assert ppo.update(obs,action,torch.zeros(32),torch.zeros(32,dtype=torch.bool))['actor_steps']==0
    assert all(torch.equal(value,before[key]) for key,value in policy.state_dict().items())
    checks.append('real episodic PPO updates eligible 13D actions only; frozen launch stays unchanged')
    print(json.dumps(dict(status='PASS_OFFLINE_ONLY',frozen_sha256=verify(),checks=checks),indent=2))


if __name__=='__main__':
    main()

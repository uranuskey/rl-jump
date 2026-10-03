"""CPU regressions against the frozen implementation and complete resume state."""
import ast
from collections import Counter
import copy
import inspect
import json
from pathlib import Path
import tempfile
import textwrap
from types import SimpleNamespace as NS
import bootstrap
import torch
from fast_checks import external_fault, post_step_faults
from fast_report import summarize
from jump_task import REASONS
from learning import Policy, PPO
from control import ACTOR_DIM, CRITIC_DIM, ACTION_DIM
from resume_state import restore


def source_equivalence():
    from controlled_env import HeightEnv
    from fast_env import FastLandingEnv
    old = inspect.getsource(HeightEnv.step).replace('def step(', 'def _step_controlled(', 1)
    old = old.replace('not torch.equal(self.external[:,self.base],assist) or bool(self.external[:,:,:3].any())',
                      'bool(external_fault(self.external, self.base, assist))')
    start = old.index('            bad = (')
    end = old.index('            phase_before =', start)
    fast = inspect.getsource(FastLandingEnv._step_controlled)
    begin_fast = fast.index('            faults =')
    end_fast = fast.index('            phase_before =', begin_fast)
    old = old[:start]+fast[begin_fast:end_fast]+old[end:]
    assert ast.dump(ast.parse(textwrap.dedent(old)))==ast.dump(ast.parse(textwrap.dedent(fast)))
    old = inspect.getsource(HeightEnv.sensors).replace(
        "torch.full_like(self.paused, bool(wp.to_torch(self.gd.overflow).any()))",
        'wp.to_torch(self.gd.overflow).any().expand_as(self.paused)')
    assert ast.dump(ast.parse(textwrap.dedent(old)))==ast.dump(ast.parse(textwrap.dedent(inspect.getsource(FastLandingEnv.sensors))))


def guard_equivalence():
    for mode in range(12):
        q, v, acc = [torch.zeros(3, 5) for _ in range(3)]
        active = torch.tensor([True, True, False])
        overflow = torch.zeros(3, dtype=torch.bool)
        force, effort, actuator = torch.zeros(3, 6), torch.zeros(3, 2), torch.zeros(3, 6)
        if mode==1: q[0, 0] = float('nan')
        if mode==2: v[1, 0] = float('inf')
        if mode==3: acc[0, 0] = float('-inf')
        if mode==4: overflow[0] = True
        if mode==5: overflow[2] = True; q[2, 0] = float('nan')
        if mode==6: force[0, 0] = 1e-6
        if mode==7: force[0, 0] = 1.01e-6
        if mode==8: force[0, 2] = 1e-20
        if mode==9: actuator[2, 3] = .01
        if mode==10: q[0, 0] = float('nan'); force[1, 0] = 1
        if mode==11: force[0, 0] = float('nan')
        original = [bool(((overflow | ~torch.isfinite(q).all(1) | ~torch.isfinite(v).all(1) |
                            ~torch.isfinite(acc).all(1)) & active).any()),
                    bool((force[:, :2]-effort).abs().max()>1e-6) or bool(force[:, 2:].abs().max()>0)
                    or bool(actuator.abs().max()>0)]
        assert post_step_faults(overflow, q, v, acc, active, force, [0,1], effort, [2,3,4,5], actuator).tolist()==original
    for mode in range(4):
        external, assist = torch.zeros(3, 4, 6), torch.zeros(3, 6)
        if mode==1: external[0, 1, 3] = 1
        if mode==2: external[0, 0, 2] = 1
        if mode==3: external[0, 1, 3] = assist[0, 3] = float('nan')
        expected = not torch.equal(external[:, 1], assist) or bool(external[:, :, :3].any())
        assert bool(external_fault(external, 1, assist))==expected


def report_equivalence():
    import rollout
    source = inspect.getsource(rollout.trial)
    body = textwrap.dedent(source[source.index('        cases = []'):source.index('        if trace_path:')])
    code = 'def original(env, score, passed, retained, metrics, terms, smooth_cost, residual, transitions, seen_gate):\n'
    namespace = dict(REASONS=REASONS, Counter=Counter)
    exec(code+textwrap.indent(body+'return summary\n', '    '), namespace)
    n = 512
    task = NS(reason=torch.randint(len(REASONS), (n,)), phase=torch.randint(8, (n,)),
        peak_clearance_m=torch.rand(n), height_score=NS(peak=torch.rand(n)),
        touchdown_time=torch.rand(n), success_time=torch.rand(n))
    env = NS(n=n, task=task, ticks=torch.randint(2001, (n,)), plan=torch.rand(n, 9), gate_tick=torch.randint(-1, 1900, (n,)))
    values = (env, torch.rand(n), torch.rand(n)>.2, torch.rand(n)>.4,
        {f'm{i}':torch.rand(n) for i in range(11)}, {f'r{i}':torch.rand(n) for i in range(10)},
        torch.rand(n), torch.rand(n)*.0001, 91283, torch.rand(n)>.2)
    assert summarize(*values)==namespace['original'](*values)
    values = (*values[:7], torch.empty(0), 0, torch.zeros(n, dtype=torch.bool))
    assert summarize(*values)==namespace['original'](*values)


def resume_equivalence():
    policy = Policy()
    ppo = PPO(policy)
    obs, critic = torch.randn(3, 4, ACTOR_DIM), torch.randn(3, 4, CRITIC_DIM)
    with torch.no_grad():
        dist = policy.distribution(obs.reshape(-1, ACTOR_DIM))
        action = dist.sample().reshape(3, 4, ACTION_DIM)
        batch = dict(obs=obs, critic=critic, action=action, mean=dist.mean.reshape_as(action),
            logp=dist.log_prob(action.reshape(-1, ACTION_DIM)).sum(-1).reshape(3, 4),
            value=policy.critic(critic).squeeze(-1), next_value=torch.zeros(3,4),
            done=torch.tensor([[False]*4, [False]*4, [True]*4]), valid=torch.ones(3,4,dtype=torch.bool), reward=torch.randn(3,4))
    ppo.update(batch)
    with tempfile.TemporaryDirectory(prefix='resume_cpu_') as folder:
        checkpoint = Path(folder)/'state.pt'
        torch.save(dict(model_state_dict=policy.state_dict(), actor_optimizer=ppo.actor_optimizer.state_dict(),
            critic_optimizer=ppo.critic_optimizer.state_dict(), update=1, frozen_sha256='fixture',
            voltage_v=24, assist_strength=.625, torch_rng=torch.get_rng_state(), cuda_rng=[]), checkpoint)
        stats = ppo.update(batch)
        expected, expected_rng = copy.deepcopy(policy.state_dict()), torch.get_rng_state().clone()
        expected_optimizers = copy.deepcopy([ppo.actor_optimizer.state_dict(), ppo.critic_optimizer.state_dict()])
        restored, new_ppo = Policy(), None
        new_ppo = PPO(restored)
        restore(restored, new_ppo, checkpoint, 1, 'fixture', cuda=False)
        assert new_ppo.update(batch)==stats
        assert all(torch.equal(v, expected[k]) for k, v in restored.state_dict().items())
        assert torch.equal(torch.get_rng_state(), expected_rng)
        for current, previous in zip([new_ppo.actor_optimizer.state_dict(), new_ppo.critic_optimizer.state_dict()], expected_optimizers):
            assert current['param_groups']==previous['param_groups']
            assert all(torch.equal(value, previous['state'][key][field]) for key, state in current['state'].items()
                       for field, value in state.items())


if __name__=='__main__':
    torch.set_num_threads(1)
    torch.manual_seed(682)
    checks = [source_equivalence, guard_equivalence, report_equivalence, resume_equivalence]
    for check in checks:
        check()
    print(json.dumps(dict(status='PASS_OFFLINE_ONLY', checks=[f.__name__ for f in checks])))

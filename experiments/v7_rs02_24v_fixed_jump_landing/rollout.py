"""Whole episodes with masked 50 Hz decisions and 400 Hz measurements."""
import bootstrap
from collections import Counter
import hashlib
from pathlib import Path
import numpy as np
import torch
from control import ACTION_DIM, features
from environment import reset_cases
from jump_task import REASONS


def reset(env):
    env.assist_strength.fill_(.625)
    reset_cases(env)


def compact(summary):
    return {k: v for k, v in summary.items() if k != 'cases'}


def qualifies(summary):
    return (summary['passed']==summary['worlds'] and summary['original45_passed']==45
            and summary['mean_wheel_cm']>=9.065 and summary['mean_com_cm']>=11.74
            and summary['launch_plan_unchanged'] and summary['pre_apex_actions_zero'])


def trial(env, standing, launch, policy, limit, *, stochastic=False, collect=False, trace_path=None):
    reset(env)
    env.record = trace_path is not None
    rows, blocks = [], []
    smooth_cost = torch.zeros(env.n, device=env.device)
    dense_return = smooth_cost.clone()
    score_at_gate = smooth_cost.clone()
    seen_gate = torch.zeros(env.n, device=env.device, dtype=torch.bool)
    transitions = 0
    with torch.no_grad():
        for step in range(250):
            limit()
            if step == 30:
                env.lock_launch(launch(env.plan_observation()))
            valid = env.task.height_score.apex & ~env.terminal_mask() & (env.ticks<2000)
            obs, critic = features(env)
            if bool(valid.any()):
                dist = policy.distribution(obs)
                action = dist.sample() if stochastic else dist.mean
                value = policy.critic(critic).flatten()
                logp = dist.log_prob(action).sum(-1)
                mean = dist.mean
            else:
                action = torch.zeros(env.n, ACTION_DIM, device=env.device)
                value = torch.zeros(env.n, device=env.device)
                logp = value.clone()
                mean = action.clone()
            before_score = env.task.landing_reward.score(env.task, env.ticks)[0]
            first = valid & ~seen_gate
            score_at_gate[first] = before_score[first]
            seen_gate |= valid
            previous = env.previous_policy_action.clone()
            teacher = standing.actor(env.obs) if step<30 else torch.zeros(env.n, 6, device=env.device)
            env.step(teacher, policy_action=action, auto_reset=False)
            after_score = env.task.landing_reward.score(env.task, env.ticks)[0]
            regularizer = valid*(.02*(env.effective_action-previous).square().mean(1)
                                 +.001*env.effective_action.square().mean(1))
            reward = torch.where(valid, after_score-before_score, 0.)-regularizer
            smooth_cost += regularizer
            dense_return += reward
            done = env.terminal_mask() | (env.ticks>=2000)
            if collect and bool(valid.any()):
                _, next_critic = features(env)
                next_value = policy.critic(next_critic).flatten()
                rows.append(dict(obs=obs, critic=critic, action=action, value=value,
                                 next_value=next_value, mean=mean, logp=logp,
                                 reward=reward, done=done.clone(), valid=valid.clone()))
                transitions += int(valid.sum())
            if env.record:
                physical = env.last['traces']
                block = {k: torch.stack([r[k] for r in physical]).cpu().numpy()
                         for k in ('active', 'ticks', 'phase', 'reason', 'assist_wrench',
                                   'arrived_reference_height', 'arrived_corrections', 'q', 'v')}
                block.update({'sensor_'+k: torch.stack([r['sample'][k] for r in physical]).cpu().numpy()
                              for k in physical[0]['sample']})
                block['landing_control_enabled'] = np.repeat(env.control_gate.cpu().numpy()[None], len(physical), 0)
                block['landing_action'] = np.repeat(env.effective_action.cpu().numpy()[None], len(physical), 0)
                blocks.append(block)
            if step>=30 and bool(done.all()):
                break
        assert bool((env.terminal_mask() | (env.ticks>=2000)).all())
        env.assert_launch_fixed()
        score, passed, retained = env.task.landing_reward.score(env.task, env.ticks)
        # Potential differences preserve the complete undiscounted landing score.
        residual = (dense_return+smooth_cost-(score-score_at_gate))[seen_gate].abs()
        if len(residual) and float(residual.max())>.001:
            raise RuntimeError('Dense reward does not telescope to the measured landing score')
        metrics, terms = env.task.landing_reward.metrics(), env.task.landing_reward.last_terms
        cases = []
        for i in range(env.n):
            cases.append(dict(world=i, case=i%45, passed=bool(passed[i]), visible=bool(retained[i]),
                reason=REASONS[int(env.task.reason[i])], phase=int(env.task.phase[i]),
                end_s=float(env.ticks[i])*.0025, wheel_cm=float(env.task.peak_clearance_m[i])*100,
                com_cm=float(env.task.height_score.peak[i])*100,
                touchdown_s=float(env.task.touchdown_time[i]), success_s=float(env.task.success_time[i]),
                reward=float(score[i]), plan=env.plan[i].cpu().tolist(), gate_tick=int(env.gate_tick[i]),
                landing_metrics={k: float(v[i]) for k, v in metrics.items()},
                reward_terms={k: float(v[i]) for k, v in terms.items()}))
        summary = dict(assist_strength=.625, worlds=env.n, passed=int(passed.sum()),
            original45_passed=int(passed[:45].sum()), mean_wheel_cm=float(env.task.peak_clearance_m.mean())*100,
            mean_com_cm=float(env.task.height_score.peak.mean())*100, mean_return=float(score.mean()),
            reasons=dict(Counter(row['reason'] for row in cases)), cases=cases,
            mean_landing_metrics={k: float(v.mean()) for k, v in metrics.items()},
            mean_reward_terms={k: float(v.mean()) for k, v in terms.items()},
            mean_action_regularization=float(smooth_cost.mean()),
            dense_reward_identity_max_error=float(residual.max()) if len(residual) else 0.,
            landing_transitions=transitions, launch_plan_unchanged=True, pre_apex_actions_zero=True,
            gate_reached_worlds=int(seen_gate.sum()))
        if trace_path:
            arrays = {k: np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
            np.savez_compressed(trace_path, **arrays)
    env.record = False
    batch = {k: torch.stack([r[k] for r in rows]) for k in rows[0]} if rows else None
    return batch, summary


def prefix_check(env, standing, launch, limit, out):
    """Paired native proof through each world's apex, with different future actions."""
    results = []
    fields = ('q', 'v', 'arrived_reference_height', 'arrived_corrections',
              'requested_height', 'requested_motor_velocity', 'requested_thrust_force',
              'assist_wrench')
    initial = []
    for treatment in (False, False, True):
        reset(env)
        initial.append(dict(q=env.q.cpu().numpy().copy(), v=env.v.cpu().numpy().copy(),
                            obs=env.obs.cpu().numpy().copy()))
        env.record = True
        records = []
        with torch.no_grad():
            for step in range(125):
                limit()
                if step == 30:
                    env.lock_launch(launch(env.plan_observation()))
                enabled = env.task.height_score.apex.clone()
                if bool(enabled.all()):
                    break
                if bool((env.terminal_mask() & ~enabled).any()):
                    raise RuntimeError('A world failed before the paired apex proof completed')
                raw = torch.zeros(env.n, ACTION_DIM, device=env.device)
                if treatment:
                    raw[:] = torch.tensor([.9, -.7, .6, -.5, .4, -.3, .2], device=env.device)
                teacher = standing.actor(env.obs) if step<30 else torch.zeros(env.n, 6, device=env.device)
                env.step(teacher, policy_action=raw, auto_reset=False)
                for r in env.last['traces']:
                    row = {k: r[k].detach().cpu().numpy() for k in fields}
                    row['motor_torque'] = r['sample']['motor_torque_nm'].cpu().numpy()
                    row['com_z'] = r['sample']['com_z_m'].cpu().numpy()
                    row['com_vz'] = r['sample']['com_vz_mps'].cpu().numpy()
                    row['mask'] = (~enabled).cpu().numpy()
                    records.append(row)
            else:
                raise RuntimeError('Paired proof did not reach all 45 COM apices')
        results.append({k: np.stack([r[k] for r in records]) for k in records[0]})
    env.record = False
    from runtime import write
    names = ('zero_first', 'zero_repeat', 'distinct')
    for name, arrays in zip(names, results):
        np.savez_compressed(out/f'prefix_{name}.npz', **arrays)
    diagnostics = []
    for index in (1, 2):
        a, b = results[0], results[index]
        length = min(len(a['mask']), len(b['mask']))
        mask = a['mask'][:length] & b['mask'][:length]
        differences, first_difference = {}, {}
        for name in a:
            if name=='mask':
                continue
            delta = np.abs(a[name][:length]-b[name][:length])
            differences[name] = float(np.max(delta[mask], initial=0))
            while delta.ndim>2:
                delta = delta.max(-1)
            where = np.argwhere((delta>1e-7)&mask)
            first_difference[name] = where[0].tolist() if len(where) else None
        diagnostics.append(dict(comparison=names[index],
            shape_a=list(a['mask'].shape), shape_b=list(b['mask'].shape),
            masks_equal=a['mask'].shape==b['mask'].shape and np.array_equal(a['mask'], b['mask']),
            initial_max_difference={k:float(np.max(np.abs(initial[0][k]-initial[index][k]))) for k in initial[0]},
            common_pre_apex_max_difference=differences, first_difference_tick_world=first_difference))
    write(out/'prefix_diagnostics.json', diagnostics)
    a, _, b = results
    if a['mask'].shape != b['mask'].shape or not np.array_equal(a['mask'], b['mask']):
        raise RuntimeError('Landing actions changed apex timing')
    mask = a['mask']
    differences = {}
    for name in a:
        if name == 'mask':
            continue
        difference = float(np.max(np.abs(a[name][mask]-b[name][mask]), initial=0))
        tolerance = .0001 if name=='motor_torque' else .000002
        if difference>tolerance:
            raise RuntimeError(f'Landing action changed pre-apex {name}: {difference}')
        differences[name] = difference
    return dict(status='PASS', worlds=env.n, physics_samples_compared=int(mask.sum()),
        max_absolute_differences=differences, comparison='zero versus distinct 7D actions; per-world pre-apex 400 Hz state and commands',
        baseline_state_sha256=hashlib.sha256(a['q'][mask].tobytes()+a['v'][mask].tobytes()).hexdigest(),
        treatment_state_sha256=hashlib.sha256(b['q'][mask].tobytes()+b['v'][mask].tobytes()).hexdigest())

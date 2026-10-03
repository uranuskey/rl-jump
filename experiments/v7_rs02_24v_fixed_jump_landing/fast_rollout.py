"""Original trajectory PPO inputs/rewards; copy result metrics in batches."""
import bootstrap
import numpy as np
import torch
from control import ACTION_DIM, features
from rollout import reset, compact, qualifies
from fast_report import summarize


def trial(env, standing, launch, policy, limit, *, stochastic=False, collect=False, trace_path=None):
    reset(env)
    env.record = trace_path is not None
    rows, blocks = [], []
    smooth_cost = torch.zeros(env.n, device=env.device)
    dense_return = smooth_cost.clone()
    score_at_gate = smooth_cost.clone()
    seen_gate = torch.zeros(env.n, device=env.device, dtype=torch.bool)
    transitions = torch.zeros((), dtype=torch.long, device=env.device)
    with torch.no_grad():
        for step in range(250):
            limit()
            if step == 30:
                env.lock_launch(launch(env.plan_observation()))
            valid = env.task.height_score.apex & ~env.terminal_mask() & (env.ticks<2000)
            obs, critic = features(env)
            has_valid = bool(valid.any())
            if has_valid:
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
            if collect and has_valid:
                _, next_critic = features(env)
                next_value = policy.critic(next_critic).flatten()
                rows.append(dict(obs=obs, critic=critic, action=action, value=value,
                                 next_value=next_value, mean=mean, logp=logp,
                                 reward=reward, done=done.clone(), valid=valid.clone()))
                transitions += valid.sum()
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
        summary = summarize(env, score, passed, retained, metrics, terms, smooth_cost,
                            residual, transitions, seen_gate)
        if trace_path:
            arrays = {k: np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
            np.savez_compressed(trace_path, **arrays)
    env.record = False
    batch = {k: torch.stack([r[k] for r in rows]) for k in rows[0]} if rows else None
    return batch, summary

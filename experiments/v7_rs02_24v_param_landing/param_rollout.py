"""Sample once at 0.6 s; gate all effects until apex; score the whole landing."""
import path_setup
import numpy as np
import torch
from fast_report import summarize
from rollout import reset, compact, qualifies
from param_control import decode


def trial(env, standing, launch, policy, limit, *, stochastic=False, trace_path=None):
    reset(env)
    env.record = trace_path is not None
    blocks, observation, action = [], None, None
    with torch.no_grad():
        for step in range(250):
            limit()
            if step==30:
                observation = env.plan_observation().clone()
                env.lock_launch(launch(observation))
                distribution = policy.distribution(observation)
                action = distribution.sample() if stochastic else distribution.mean
                env.lock_parameters(action)
            teacher = standing.actor(env.obs) if step<30 else torch.zeros(env.n, 6, device=env.device)
            env.step(teacher, auto_reset=False)
            if env.record:
                rows = env.last['traces']
                block = {k:torch.stack([row[k] for row in rows]).cpu().numpy()
                         for k in ('active','ticks','phase','reason','assist_wrench',
                                   'arrived_reference_height','arrived_corrections','q','v')}
                block.update({'sensor_'+k:torch.stack([row['sample'][k] for row in rows]).cpu().numpy()
                              for k in rows[0]['sample']})
                block['landing_control_enabled'] = np.repeat(env.control_gate.cpu().numpy()[None], len(rows), 0)
                block['landing_action'] = np.repeat(env.effective_action.cpu().numpy()[None], len(rows), 0)
                blocks.append(block)
            if step>=30 and bool((env.terminal_mask() | (env.ticks>=2000)).all()):
                break
        assert observation is not None and bool((env.terminal_mask() | (env.ticks>=2000)).all())
        env.assert_launch_fixed()
        env.assert_parameters_fixed()
        reward, passed, retained = env.task.landing_reward.score(env.task, env.ticks)
        eligible = env.gate_tick>=0
        summary = summarize(env, reward, passed, retained, env.task.landing_reward.metrics(),
            env.task.landing_reward.last_terms, torch.zeros_like(reward), reward.new_empty(0),
            eligible.sum(), eligible)
        summary.pop('dense_reward_identity_max_error')
        summary.pop('landing_transitions')
        summary.update(executed_landing_plans=int(eligible.sum()), parameter_samples_per_episode=1,
                       parameter_plan_unchanged=True, reward_mode='undiscounted ten-term episodic landing score')
        parameters, gains = decode(env.parameter_action)
        data = torch.cat((parameters, gains), 1).cpu().tolist()
        for row, values in zip(summary['cases'], data):
            row.update(landing_parameters=values[:5], feedback_gains=values[5:])
        if trace_path:
            arrays = {k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
            np.savez_compressed(trace_path, **arrays)
    env.record = False
    return observation, action, reward, eligible, summary

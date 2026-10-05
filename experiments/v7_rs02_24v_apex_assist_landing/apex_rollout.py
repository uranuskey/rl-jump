"""Preserve the takeoff prefix and report the actual phase-specific assistance."""
import apex_runtime
import numpy as np
import torch
from environment import reset_cases
from fast_report import summarize
from rollout import compact
from compliant_control import decode

TRACE_FIELDS = ('active', 'ticks', 'phase', 'reason', 'assist_wrench',
    'arrived_reference_height', 'arrived_corrections', 'q', 'v',
    'requested_payload', 'arrived_payload', 'impedance_kp', 'impedance_kd',
    'landing_control_enabled', 'landing_action', 'assist_strength',
    'assist_effective_strength', 'assist_world_omega_pre', 'assist_world_omega_post',
    'base_rotation_pre', 'assist_power_w', 'assist_work_j')


def trial(env, standing, launch, policy, limit, *, strength, stochastic=False, trace_path=None):
    assert 0 < strength <= .625
    env.assist_target = strength
    env.assist_strength.fill_(.625)
    reset_cases(env)
    assert bool((env.assist_strength == .625).all()), 'Reset changed takeoff assistance'
    env.record = trace_path is not None
    blocks, observation, action = [], None, None
    with torch.no_grad():
        for step in range(250):
            limit()
            if step == 30:
                observation = env.plan_observation().clone()
                env.lock_launch(launch(observation))
                distribution = policy.distribution(observation)
                action = distribution.sample() if stochastic else distribution.mean
                env.lock_parameters(action)
            teacher = standing.actor(env.obs) if step < 30 else torch.zeros(env.n, 6, device=env.device)
            env.step(teacher, auto_reset=False)
            if env.record:
                rows = env.last['traces']
                block = {k: torch.stack([row[k] for row in rows]).cpu().numpy() for k in TRACE_FIELDS}
                block.update({'sensor_'+k: torch.stack([row['sample'][k] for row in rows]).cpu().numpy()
                              for k in rows[0]['sample']})
                blocks.append(block)
            if step >= 30 and bool((env.terminal_mask() | (env.ticks >= 2000)).all()):
                break
        assert observation is not None and bool((env.terminal_mask() | (env.ticks >= 2000)).all())
        assert bool(((env.assist_strength >= strength-1e-7) & (env.assist_strength <= .625+1e-7)).all())
        env.assert_launch_fixed()
        env.assert_parameters_fixed()
        reward, passed, retained = env.task.landing_reward.score(env.task, env.ticks)
        eligible = env.gate_tick >= 0
        summary = summarize(env, reward, passed, retained, env.task.landing_reward.metrics(),
            env.task.landing_reward.last_terms, torch.zeros_like(reward), reward.new_empty(0),
            eligible.sum(), eligible)
        summary.pop('dense_reward_identity_max_error')
        summary.pop('landing_transitions')
        summary.update(assist_strength=None, takeoff_assist_strength=.625,
            after_apex_assist_strength=strength, assist_schedule='observed_COM_apex_then_100ms_ramp', executed_landing_plans=int(eligible.sum()),
            parameter_samples_per_episode=1, parameter_plan_unchanged=True,
            reward_mode='unchanged guided 13-term actual-landing score', force_target_n=300.,
            controller_hz=400, sample_kind='stochastic_training' if stochastic else 'deterministic_evaluation')
        parameters, gains = decode(env.parameter_action)
        for row, values in zip(summary['cases'], torch.cat((parameters, gains), 1).cpu().tolist()):
            row.update(landing_parameters=values[:8], feedback_gains=values[8:])
        diagnostics = torch.stack((env.max_pre_mimic_q, env.max_pre_mimic_v,
                                   env.max_all_mimic_q, env.max_all_mimic_v), 1).cpu().tolist()
        terminal = {k: v.cpu().tolist() for k, v in env.terminal.items()}
        for i, row in enumerate(summary['cases']):
            row['constraint_residual_maxima'] = dict(zip(
                ('pre_apex_q_rad', 'pre_apex_v_rad_s', 'all_q_rad', 'all_v_rad_s'), diagnostics[i]))
            if not row['passed']:
                row['terminal_diagnostics'] = {k: v[i] for k, v in terminal.items()}
        summary['max_pre_apex_mimic_v_rad_s'] = max(d[1] for d in diagnostics)
        summary['max_mimic_v_rad_s'] = max(d[3] for d in diagnostics)
        if trace_path is not None:
            arrays = {k: np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
            np.savez_compressed(trace_path, **arrays)
    env.record = False
    return observation, action, reward, eligible, summary

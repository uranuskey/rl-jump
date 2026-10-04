"""Native same-state/FIFO proof for 13D episodic landing parameters."""
import path_setup
import hashlib
import numpy as np
import torch
from param_rollout import reset
from param_control import initial_raw
OFFSETS = (.9, -.7, .6, -.5, .4, -.3, .2, .7, -.6, .5, -.4, .3, -.2)


def prefix_check(env, standing, launch, limit, out):
    """Exact same-state command/FIFO proof plus separate native replay diagnostics."""
    from launch_contract import payload
    from velocity_contract import reference_motor_velocity
    from rs02_actuator import PhysicalStepFIFO
    results = []
    control_comparisons = physics_comparisons = mixed_world_steps = 0
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
                    episode_action = initial_raw(env.device).expand(env.n, -1).clone()
                    if treatment:
                        episode_action += episode_action.new_tensor(OFFSETS)
                    env.lock_parameters(episode_action)
                enabled = env.task.height_score.apex.clone()
                if bool(enabled.all()):
                    break
                if bool((env.terminal_mask() & ~enabled).any()):
                    raise RuntimeError('A world failed before the paired apex proof completed')
                raw = initial_raw(env.device).expand(env.n, -1)
                teacher = standing.actor(env.obs) if step<30 else torch.zeros(env.n, 6, device=env.device)
                zero = env.prepare_commands(teacher, raw)
                other = env.prepare_commands(teacher, raw+raw.new_tensor(OFFSETS))
                for name in zero:
                    if not torch.equal(zero[name][~enabled], other[name][~enabled]):
                        raise RuntimeError('Landing command depends on pre-apex policy action: '+name)
                control_comparisons += int((~enabled).sum())
                mixed_world_steps += int(bool(enabled.any()) and not bool(enabled.all()))
                queues, requests = [], []
                for commands in (zero, other):
                    curve = commands['curve_state']
                    height = .18+curve[:, 0]
                    velocity = curve[:, 3, None]*reference_motor_velocity(height, curve[:, 1])
                    requests.append(payload(torch.tanh(commands['actions']), height, velocity, curve[:, 2]))
                    queue = PhysicalStepFIFO(env.n, 12, env.device)
                    queue.history.copy_(env.fifo.history)
                    queue.delay_steps.copy_(env.fifo.delay_steps)
                    queue.write_index = env.fifo.write_index
                    queues.append(queue)
                expected = []
                for _ in range(8):
                    a, b = [q.push(r) for q, r in zip(queues, requests)]
                    if not torch.equal(a[~enabled], b[~enabled]):
                        raise RuntimeError('Landing action changed a pre-apex FIFO payload')
                    expected.append(a)
                env.step(teacher, auto_reset=False)
                for r, predicted in zip(env.last['traces'], expected):
                    actual = torch.cat((r['arrived_corrections'], r['arrived_reference_height'][:, None],
                                        r['arrived_motor_velocity'], r['thrust_requested_force'][:, None]), 1)
                    decoded = torch.cat((predicted[:, :6], (.18+.03*predicted[:, 6])[:, None], predicted[:, 7:]), 1)
                    if not torch.equal(actual[~enabled], decoded[~enabled]):
                        raise RuntimeError('Native pre-apex FIFO differs from the checked command path')
                    physics_comparisons += int((~enabled).sum())
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
    from param_runtime import write
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
    assert mixed_world_steps>0
    return dict(status='PASS', worlds=env.n, physics_samples_compared=physics_comparisons,
        control_samples_compared=control_comparisons, mixed_world_steps=mixed_world_steps,
        counterfactual_command_max_difference=0., counterfactual_fifo_max_difference=0.,
        native_fifo_readback_max_difference=0.,
        comparison='Exact zero versus distinct 13D episode parameters at identical physical/controller state; 400 Hz native FIFO readback',
        replay_diagnostics=diagnostics, independent_physics_replays_are_bitwise_equal=False,
        limitation='GPU repeated zero-action trajectories can diverge; replay differences do not prove action leakage',
        trace_sha256={name:hashlib.sha256((out/f'prefix_{name}.npz').read_bytes()).hexdigest() for name in names})

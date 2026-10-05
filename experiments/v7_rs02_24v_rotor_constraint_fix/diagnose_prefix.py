"""Matched prefix-only experiments; no training or landing qualification claims."""
import argparse
import time
import re
import traceback
import numpy as np
from fix_runtime import HERE, verify, contract, read, write, sha, now, exclusive, limit_for
from fix_contract import VARIANTS, prefix_qualified


def trial(env, standing, launch, policy, limit, path):
    import torch
    from environment import reset_cases
    env.capture_prefix = False
    reset_cases(env)
    env.prefix_calls, env.prefix_rows = 0, []
    env.capture_prefix = True
    with torch.no_grad():
        for step in range(60):
            limit()
            if step == 30:
                obs = env.plan_observation().clone()
                env.lock_launch(launch(obs))
                env.lock_parameters(policy.distribution(obs).mean)
            teacher = standing.actor(env.obs) if step < 30 else torch.zeros(env.n, 6, device=env.device)
            env.step(teacher, auto_reset=False)
    env.capture_prefix = False
    env.assert_launch_fixed(); env.assert_parameters_fixed()
    z = {k:torch.stack([r[k] for r in env.prefix_rows]).cpu().numpy() for k in env.prefix_rows[0]}
    np.savez_compressed(path, **z)
    live = z['active']
    v64 = np.abs(z['rotor_v'].astype(np.float64)-7.75*z['joint_v'].astype(np.float64)).max(-1)
    q64 = np.abs(z['rotor_q'].astype(np.float64)-7.75*z['joint_q'].astype(np.float64)).max(-1)
    delta = np.abs(z['reported_v']-v64)
    q, v = env.max_all_mimic_q.cpu().tolist(), env.max_all_mimic_v.cpu().tolist()
    failures = env.terminal_mask().cpu().tolist()
    reasons = env.task.reason.cpu().tolist()
    from jump_task import REASONS
    cases = [dict(world=i, case=i%45, failed=failures[i], reason=REASONS[reasons[i]],
                  max_q_rad=q[i], max_v_rad_s=v[i]) for i in range(env.n)]
    return dict(worlds=env.n, prefix_s=1.20, failed=sum(failures), cases=cases,
        max_mimic_q_rad=max(q), max_mimic_v_rad_s=max(v),
        captured_window_s=[.90, 1.20], float64_recomputed_peak_v=float(v64[live].max()),
        float64_recomputed_peak_q=float(q64[live].max()),
        max_float32_measurement_difference=float(delta[live].max()),
        max_solver_iterations=int(z['solver_niter'][live].max()),
        mean_solver_iterations=float(z['solver_niter'][live].mean()),
        trace=path.name, trace_sha256=sha(path), physical_guards_unchanged=True,
        landing_qualified=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-id', required=True)
    args = p.parse_args()
    assert re.fullmatch('[A-Za-z0-9_-]+', args.run_id)
    frozen, source = verify(), contract()
    out = HERE/'runs'/args.run_id
    assert not out.exists()
    with exclusive(512) as resources:
        out.mkdir(parents=True)
        started = time.monotonic()
        result = dict(status='RUNNING', utc_start=now(), frozen_sha256=frozen, contract=source,
            resources=resources, variants={}, selected_variant=None, training_updates=0,
            before=.6125, after=.60, prefix_only=True)
        write(out/'progress.json', result)
        try:
            import torch
            from fix_env import ConstraintEnv
            from slot_learning import Policy, FrozenLaunch
            from standing_policy import JumpPolicy
            torch.set_num_threads(1)
            ck = source['source_checkpoint']
            assert sha(ck['path']) == ck['sha256']
            state = torch.load(ck['path'], map_location='cpu', weights_only=True)
            for name in VARIANTS:
                result['stage'] = name
                write(out/'progress.json', result)
                torch.manual_seed(105067); torch.cuda.manual_seed_all(105067)
                env = ConstraintEnv(512, [source['profile']], before=.6125, after=.60,
                    variant=name, proof=True, abort_dir=out/(name+'_aborts'))
                standing = JumpPolicy(env.device).eval().requires_grad_(False)
                launch, policy = FrozenLaunch(env.device), Policy(env.device).eval().requires_grad_(False)
                policy.load_state_dict(state['model_state_dict'], strict=True)
                record = dict(physics_receipt=env.physics_receipt, repeats=[])
                result['variants'][name] = record
                for repeat in (1, 2, 3):
                    result['stage'] = f'{name}_{repeat}'
                    write(out/'progress.json', result)
                    row = trial(env, standing, launch, policy, limit_for(started, 5400), out/f'{name}_{repeat}.npz')
                    row['repeat'] = repeat
                    write(out/f'{name}_{repeat}.json', row)
                    record['repeats'].append({k:v for k,v in row.items() if k!='cases'})
                    write(out/'progress.json', result)
                    print(name, repeat, {k:v for k,v in row.items() if k!='cases'}, flush=True)
                record['prefix_qualified'] = prefix_qualified(record['repeats'])
                del env, standing, launch, policy
                # Always compare the unchanged model, stricter solve and10ms
                # coupling. Test5ms only if neither other correction has margin.
                if name == 'rotor10ms' and any(result['variants'][k]['prefix_qualified'] for k in ('solve_strict','rotor10ms')):
                    break
            result['selected_variant'] = next((n for n in ('solve_strict','rotor10ms','rotor5ms')
                if n in result['variants'] and result['variants'][n]['prefix_qualified']), None)
            result.update(status='DIAGNOSED', final_frozen_sha256=verify())
        except BaseException as error:
            result.update(status='ERROR_STOPPED', error=repr(error), traceback=traceback.format_exc())
            print(result['traceback'], flush=True)
        finally:
            result.update(utc_end=now(), wall_s=time.monotonic()-started)
            write(out/'result.json', result); write(out/'progress.json', result)
    return int(result['status']=='ERROR_STOPPED')


if __name__ == '__main__':
    raise SystemExit(main())

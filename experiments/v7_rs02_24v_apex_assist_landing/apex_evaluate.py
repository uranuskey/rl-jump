"""Independent native45 comparison at the original and reduced assistance."""
import argparse
from pathlib import Path
import re
import time
import traceback
from apex_runtime import HERE, verify, read, write, sha, now, exclusive, limit_for, metrics
from apex_training_runtime import verify_training, checked_result
from withdrawal_contract import SOURCE_LEVEL, PARENT_PHYSICS_SHA, admission


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--training', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    args = p.parse_args()
    assert re.fullmatch('[A-Za-z0-9_-]+', args.run_id)
    frozen, training_frozen = verify(), verify_training()
    train = checked_result(args.training/'result.json', 'BUDGET_COMPLETED', 'train')
    assert train['completed_updates'] == 128 and train['num_envs'] == 512
    assert train['training_frozen_sha256'] == train['final_training_frozen_sha256'] == training_frozen
    contract = train['contract']
    out = HERE/'runs'/args.run_id
    assert not out.exists(), 'Use a fresh run id'
    with exclusive(45) as resources:
        out.mkdir(parents=True)
        started = time.monotonic()
        result = dict(status='RUNNING', utc_start=now(), resources=resources, num_envs=45,
            frozen_sha256=frozen, training_frozen_sha256=training_frozen,
            training_result_sha256=sha(args.training/'result.json'), models={}, voltage_v=24,
            sample_kind='independent_native_deterministic', zero_assistance_qualified=False,
            hardware_qualified=False)
        write(out/'progress.json', result)
        try:
            import torch
            from apex_env import ApexEnv
            from slot_learning import Policy, FrozenLaunch
            from standing_policy import JumpPolicy
            from apex_rollout import trial, compact
            from apex_audit import audit
            torch.set_num_threads(1)
            entries = [('reference625', train['source_checkpoint'], SOURCE_LEVEL),
                       ('seed', train['initial_checkpoint'], contract['strength']),
                       ('selected', train['selected']['checkpoint'], contract['strength']),
                       ('latest', train['final_checkpoint'], contract['strength'])]
            for name, checkpoint, level in entries:
                result['stage'] = name
                write(out/'progress.json', result)
                torch.manual_seed(105061)
                torch.cuda.manual_seed_all(105061)
                assert sha(checkpoint['path']) == checkpoint['sha256']
                state = torch.load(checkpoint['path'], map_location='cpu', weights_only=True)
                assert state['frozen_sha256'] == PARENT_PHYSICS_SHA
                assert state['voltage_v'] == 24
                if name == 'reference625':
                    assert state['assist_strength'] == .625
                else:
                    assert state['takeoff_assist_strength'] == .625
                    assert state['after_apex_assist_strength'] == level
                    assert state['assist_schedule'] == 'observed_COM_apex_then_100ms_ramp'
                if name != 'reference625':
                    assert state['apex_frozen_sha256'] == frozen
                    assert state['training_frozen_sha256'] == training_frozen
                    assert state['profile'] == contract['profile'] and state['air_shift_in_actor_m'] == -.005
                env = ApexEnv(45, [contract['profile']], fast_backend=False, proof=True, abort_dir=out/(name+'_aborts'))
                standing = JumpPolicy(env.device).eval().requires_grad_(False)
                launch, policy = FrozenLaunch(env.device), Policy(env.device).eval().requires_grad_(False)
                policy.load_state_dict(state['model_state_dict'], strict=True)
                trace = out/f'{name}_traces.npz'
                *_, summary = trial(env, standing, launch, policy, limit_for(started, 7200), strength=level, trace_path=trace)
                write(out/f'{name}.json', summary)
                measured = metrics(summary)
                assert env.proof_samples > 0
                checks = audit(trace, summary, env.mass, contract['profile'], level)
                anchor = result['models'].get('reference625', {}).get('metrics', contract['anchor_metrics'])
                qualified = admission(measured, anchor)
                result['models'][name] = dict(checkpoint=checkpoint, strength=level,
                    metrics=measured, takeoff_assist_strength=.625, after_apex_assist_strength=level,
                    evaluation=compact(summary), admission=qualified,
                    audits=checks, trace_sha256=sha(trace), summary_sha256=sha(out/f'{name}.json'),
                    prefix_proof_samples=env.proof_samples)
                result['mass_kg'] = env.mass
                write(out/'progress.json', result)
                print(name, measured, qualified, flush=True)
                if name == 'reference625':
                    assert qualified['passed'] and checks['status'] == 'PASS', 'Independent reference drifted'
                del env, standing, launch, policy
            result.update(status='EVALUATED', final_frozen_sha256=verify(),
                          final_training_frozen_sha256=verify_training())
        except BaseException as error:
            result.update(status='ERROR_STOPPED', error=repr(error), traceback=traceback.format_exc())
            print(result['traceback'], flush=True)
        finally:
            result.update(utc_end=now(), wall_s=time.monotonic()-started)
            write(out/'result.json', result)
            write(out/'progress.json', result)
    return int(result['status'] == 'ERROR_STOPPED')


if __name__ == '__main__':
    raise SystemExit(main())

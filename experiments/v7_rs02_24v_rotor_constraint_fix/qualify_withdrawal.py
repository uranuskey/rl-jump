"""Bounded, sequential withdrawal qualification; no optimizer or reward search."""
import argparse
import re
import time
import traceback
from pathlib import Path
from fix_runtime import (HERE, verify, contract as source_contract, metrics, read, write, sha,
                         now, exclusive, limit_for)
from probe_runtime import PARENT_SHA as SOURCE_TRAINING_SHA
from fix_contract import LEVELS, admission, qualified, deepest_qualified, prefix_qualified


def execute(out, result, contract, limit):
    import torch
    from fix_env import ConstraintEnv
    from slot_learning import Policy, FrozenLaunch
    from standing_policy import JumpPolicy
    from probe_rollout import trial, compact
    from fix_audit import audit
    from withdrawal_contract import PARENT_PHYSICS_SHA
    torch.set_num_threads(1)
    reference = contract['original_reference']['source_checkpoint']
    entries = [('reference625', .625, .625, reference, 1)] + [
        (name, before, after, contract['source_checkpoint'], 3) for name, before, after in LEVELS]
    result['models'] = {}
    for name, before, after, checkpoint, repeats in entries:
        assert sha(checkpoint['path']) == checkpoint['sha256']
        state = torch.load(checkpoint['path'], map_location='cpu', weights_only=True)
        assert state['frozen_sha256'] == PARENT_PHYSICS_SHA and state['voltage_v'] == 24
        assert state['profile'] == contract['profile'] and state['action_dim'] == 16
        if name != 'reference625':
            assert state['training_frozen_sha256'] == SOURCE_TRAINING_SHA
            assert state['update'] == checkpoint['update']
            assert state['takeoff_assist_strength'] == .625 and state['after_apex_assist_strength'] == .60
        record = dict(checkpoint=checkpoint, before=before, after=after, batch_replays=[], variant=result['constraint_variant'])
        result['models'][name] = record
        # Native45 first, retaining complete 400 Hz physical/assistance traces.
        result['stage'] = name+'_native45'
        write(out/'progress.json', result)
        torch.manual_seed(105065)
        torch.cuda.manual_seed_all(105065)
        env = ConstraintEnv(45, [contract['profile']], before=before, after=after, variant=result['constraint_variant'],
            fast_backend=False, proof=True, abort_dir=out/(name+'_native_aborts'))
        record['physics_receipt'] = env.physics_receipt
        standing = JumpPolicy(env.device).eval().requires_grad_(False)
        launch, policy = FrozenLaunch(env.device), Policy(env.device).eval().requires_grad_(False)
        policy.load_state_dict(state['model_state_dict'], strict=True)
        trace = out/(name+'_traces.npz')
        *_, summary = trial(env, standing, launch, policy, limit,
            before=before, after=after, trace_path=trace)
        summary_path = out/(name+'_native.json')
        write(summary_path, summary)
        measured = metrics(summary)
        anchor = (contract['historical_anchor'] if name == 'reference625'
                  else result['models']['reference625']['native']['metrics'])
        checks = audit(trace, summary, env.mass, contract['profile'], before, after, result['constraint_variant'])
        assert env.proof_samples > 0
        old_native = contract['old_models'].get(name, {}).get('native', {}).get('metrics')
        old_check = None if old_native is None else admission(measured, old_native)
        old_reference = admission(measured, contract['old_models']['reference625']['native']['metrics'])
        record['native'] = dict(metrics=measured, admission=admission(measured, anchor),
            old_physics_comparison=old_check, old_reference_admission=old_reference, audits=checks,
            max_mimic_v_rad_s=summary['max_mimic_v_rad_s'],
            evaluation=compact(summary), summary=summary_path.name, summary_sha256=sha(summary_path),
            trace=trace.name, trace_sha256=sha(trace), prefix_proof_samples=env.proof_samples)
        result['mass_kg'] = env.mass
        write(out/'progress.json', result)
        print(result['stage'], measured, record['native']['admission'], flush=True)
        del env, standing, launch, policy
        if not record['native']['admission']['passed'] or checks['status'] != 'PASS' or not old_reference['passed'] or (old_check is not None and not old_check['passed']):
            record['qualified'] = False
            result['stopped_at'] = name
            break
        # Rare rebounds appeared only with batched execution. A fixed three
        # repeats are required for each policy/level, never retries-until-pass.
        torch.manual_seed(105066)
        torch.cuda.manual_seed_all(105066)
        env = ConstraintEnv(512, [contract['profile']], before=before, after=after, variant=result['constraint_variant'],
            proof=True, abort_dir=out/(name+'_batch_aborts'))
        record['physics_receipt'] = env.physics_receipt
        standing = JumpPolicy(env.device).eval().requires_grad_(False)
        launch, policy = FrozenLaunch(env.device), Policy(env.device).eval().requires_grad_(False)
        policy.load_state_dict(state['model_state_dict'], strict=True)
        for repeat in range(1, repeats+1):
            result['stage'] = f'{name}_batch_{repeat}'
            write(out/'progress.json', result)
            *_, summary = trial(env, standing, launch, policy, limit, before=before, after=after)
            path = out/f'{name}_batch_{repeat}.json'
            write(path, summary)
            measured = metrics(summary)
            anchor = (contract['historical_anchor'] if name == 'reference625'
                      else result['models']['reference625']['batch_replays'][0]['metrics'])
            assert env.proof_samples > 0
            record['batch_replays'].append(dict(repeat=repeat, metrics=measured,
                admission=admission(measured, anchor),
                old_reference_admission=admission(measured, contract['old_models']['reference625']['batch_replays'][0]['metrics']),
                max_mimic_v_rad_s=summary['max_mimic_v_rad_s'],
                max_pre_apex_mimic_v_rad_s=summary['max_pre_apex_mimic_v_rad_s'], summary=path.name, summary_sha256=sha(path),
                prefix_proof_samples=env.proof_samples, traced=False,
                sample_kind='fresh_512_deterministic_replay'))
            write(out/'progress.json', result)
            print(result['stage'], measured, record['batch_replays'][-1]['admission'], flush=True)
        del env, standing, launch, policy
        record['qualified'] = qualified(record, repeats)
        if not record['qualified']:
            result['stopped_at'] = name
            break
    chosen = deepest_qualified(result['models'])
    result.update(status='PROBE_COMPLETED', deepest_qualified=chosen,
        further_withdrawal_qualified=chosen not in (None, 'current625_600'),
        stopping_rule='First unqualified level stops further reductions; all failures retained',
        selection_rule='Lowest qualified assistance; no force-improvement ranking or requirement')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-id', required=True)
    p.add_argument('--diagnosis', type=Path, required=True)
    args = p.parse_args()
    assert re.fullmatch('[A-Za-z0-9_-]+', args.run_id)
    frozen, contract = verify(), source_contract()
    diagnosis, receipt = read(args.diagnosis/'result.json'), read(args.diagnosis/'exit_receipt.json')
    assert receipt['process_exited'] and receipt['exit_code'] == 0
    assert diagnosis['status'] == 'DIAGNOSED' and diagnosis['contract'] == contract
    assert diagnosis['frozen_sha256'] == diagnosis['final_frozen_sha256'] == frozen
    variant = diagnosis['selected_variant']
    assert variant is not None, 'No correction achieved the required prefix residual margin'
    assert prefix_qualified(diagnosis['variants'][variant]['repeats'])
    out = HERE/'runs'/args.run_id
    assert not out.exists(), 'Use a new run id'
    with exclusive(512) as resources:
        out.mkdir(parents=True)
        started = time.monotonic()
        result = dict(status='RUNNING', utc_start=now(), frozen_sha256=frozen,
            resources=resources, contract=contract, models={}, training_updates=0,
            constraint_variant=variant, diagnosis=dict(path=str(args.diagnosis/'result.json'), sha256=sha(args.diagnosis/'result.json')),
            physics_revision='explicit rotor coupling or convergence correction; original asset retained',
            objective='Withdraw takeoff and landing attitude assistance with retained capability',
            voltage_v=24, motor_curve='estimated_24V', native_worlds=45, batch_worlds=512,
            distinct_initial_conditions=45, fixed_policy_weights=True,
            actual_launch_plan_may_change_with_state=True,
            zero_assistance_qualified=False, hardware_qualified=False)
        write(out/'progress.json', result)
        try:
            execute(out, result, contract, limit_for(started, 10800))
            result['final_frozen_sha256'] = verify()
            assert result['final_frozen_sha256'] == frozen
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

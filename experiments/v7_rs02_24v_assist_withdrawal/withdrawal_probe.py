"""Matched native45 replays before any optimizer or further assistance change."""
import argparse
import json
import re
import time
import traceback
from withdrawal_runtime import *
from withdrawal_contract import SOURCE_LEVEL, PROBE_LEVELS, admission


def execute(args, out, result, contract, limit):
    import torch
    from slot_env import SlotEnv
    from slot_learning import Policy, FrozenLaunch
    from standing_policy import JumpPolicy
    from withdrawal_rollout import trial, compact
    from withdrawal_audit import audit
    torch.set_num_threads(1)
    result['models'] = {}
    for name, level in [('reference625', SOURCE_LEVEL), ('candidate600', PROBE_LEVELS[0]),
                        ('fallback6125', PROBE_LEVELS[1])]:
        result['stage'] = name
        write(out/'progress.json', result)
        torch.manual_seed(105059)
        torch.cuda.manual_seed_all(105059)
        env = SlotEnv(45, [contract['profile']], fast_backend=False, proof=True, abort_dir=out/(name+'_aborts'))
        standing = JumpPolicy(env.device).eval().requires_grad_(False)
        launch, policy = FrozenLaunch(env.device), Policy(env.device).eval().requires_grad_(False)
        load_source(policy, contract)
        trace = out/f'{name}_traces.npz'
        *_, summary = trial(env, standing, launch, policy, limit, strength=level, trace_path=trace)
        write(out/f'{name}.json', summary)
        measured = metrics(summary)
        assert env.proof_samples > 0
        checks = audit(trace, summary, env.mass, contract['profile'], level)
        anchor = result['models'].get('reference625', {}).get('metrics', contract['previous_metrics'])
        admitted = admission(measured, anchor)
        record = dict(strength=level, metrics=measured, evaluation=compact(summary),
            admission=admitted, audits=checks, prefix_proof_samples=env.proof_samples,
            trace_sha256=sha(trace), summary_sha256=sha(out/f'{name}.json'))
        result['models'][name] = record
        result['mass_kg'] = env.mass
        write(out/'progress.json', result)
        print(json.dumps(dict(stage=name, metrics=measured, admission=admitted, audit=checks['status'])), flush=True)
        if name == 'reference625':
            assert admitted['passed'] and checks['status'] == 'PASS', 'Source failed current-machine reproduction'
        del env, standing, launch, policy
        if name == 'candidate600' and admitted['passed'] and checks['status'] == 'PASS':
            break
    admitted = [(name, row) for name, row in result['models'].items()
                if name != 'reference625' and row['admission']['passed'] and row['audits']['status'] == 'PASS']
    chosen = min(admitted, key=lambda item: item[1]['strength'], default=None)
    result.update(status='PROBE_COMPLETED', selected=None if chosen is None else
                  dict(name=chosen[0], strength=chosen[1]['strength'], metrics=chosen[1]['metrics']),
                  lower_assistance_admitted=chosen is not None, training_updates=0,
                  selection_rule='60 percent first; 61.25 percent only if 60 percent does not qualify')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    assert re.fullmatch('[A-Za-z0-9_-]+', args.run_id)
    frozen, contract = verify(), source_contract()
    out = HERE/'runs'/args.run_id
    assert not out.exists(), 'Use a fresh run id'
    with exclusive(45) as resources:
        out.mkdir(parents=True)
        started = time.monotonic()
        result = dict(status='RUNNING', utc_start=now(), num_envs=45, resources=resources,
            frozen_sha256=frozen, contract=contract, voltage_v=24, deterministic=True,
            zero_assistance_qualified=False, hardware_qualified=False, fixed_takeoff_weights=True,
            actual_launch_plan_may_change_with_state=True)
        write(out/'progress.json', result)
        try:
            execute(args, out, result, contract, limit_for(started, 5400))
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

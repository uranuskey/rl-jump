"""Independent native 45-case comparisons without training exploration noise."""
import argparse
from pathlib import Path
import re
import time
import traceback
from slot_training_runtime import *


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--training', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    args = p.parse_args()
    assert re.fullmatch('[A-Za-z0-9_-]+', args.run_id)
    frozen = verify_training()
    train = checked_result(args.training/'result.json', 'BUDGET_COMPLETED', 'train')
    receipt_identity(train, frozen)
    assert train['completed_updates'] == 128 and train['num_envs'] == 512
    out = HERE/'runs'/args.run_id
    assert not out.exists(), 'Use a fresh run id'
    with exclusive(45) as resources:
        out.mkdir(parents=True)
        started = time.monotonic()
        result = dict(status='RUNNING', utc_start=now(), resources=resources,
            frozen_sha256=SOURCE_FROZEN, training_frozen_sha256=frozen,
            training_result_sha256=sha(args.training/'result.json'), models={},
            voltage_v=24, assist_strength=.625, num_envs=45, deterministic=True,
            sample_kind='independent_native_deterministic_evaluation')
        write(out/'progress.json', result)
        try:
            import torch
            from slot_control import PROFILES
            from slot_env import SlotEnv
            from slot_learning import Policy, FrozenLaunch
            from compliant_rollout import trial, compact
            from standing_policy import JumpPolicy
            torch.set_num_threads(1)
            standing = JumpPolicy('cuda:0').eval().requires_grad_(False)
            launch = FrozenLaunch('cuda:0')
            for name, checkpoint, profile in (
                ('baseline', train['source_checkpoint'], PROFILES[0]),
                ('seed', train['initial_checkpoint'], train['contract']['profile']),
                ('selected', train['selected']['checkpoint'], train['contract']['profile']),
                ('latest', train['final_checkpoint'], train['contract']['profile'])):
                result['stage'] = name
                write(out/'progress.json', result)
                # Match initial conditions across independent controller/model replays.
                torch.manual_seed(105058)
                torch.cuda.manual_seed_all(105058)
                assert sha(checkpoint['path']) == checkpoint['sha256']
                state = torch.load(checkpoint['path'], map_location='cpu', weights_only=True)
                assert state['frozen_sha256'] == SOURCE_FROZEN and state['action_dim'] == 16
                if name != 'baseline':
                    assert state['training_frozen_sha256'] == frozen
                    assert state['profile'] == profile and state['air_shift_in_actor_m'] == -.005
                env = SlotEnv(45, [profile], fast_backend=False, proof=True, abort_dir=out/(name+'_aborts'))
                policy = Policy(env.device).eval().requires_grad_(False)
                policy.load_state_dict(state['model_state_dict'], strict=True)
                trace = out/f'{name}_traces.npz'
                *_, summary = trial(env, standing, launch, policy, limit_for(started,7200), trace_path=trace)
                write(out/f'{name}.json', summary)
                assert env.proof_samples > 0
                result['models'][name] = dict(checkpoint=checkpoint, profile=profile,
                    metrics=metrics(summary), evaluation=compact(summary), prefix_proof_samples=env.proof_samples)
                result['mass_kg'] = env.mass
                write(out/'progress.json', result)
                print(name, result['models'][name]['metrics'], flush=True)
                del env, policy
            m = result['models']
            for name in ('seed','selected','latest'):
                m[name]['seed_qualified'] = seed_qualified(m[name]['metrics'],m['baseline']['metrics'])
                m[name]['strict_improvement_vs_baseline'] = eligible(m[name]['metrics'],m['baseline']['metrics'])
            result.update(status='EVALUATED',
                controller_improved=m['seed']['strict_improvement_vs_baseline'],
                ppo_improved=(m['selected']['checkpoint']['update']>0 and m['selected']['seed_qualified']
                    and m['selected']['metrics']['mean_force_n'] < m['seed']['metrics']['mean_force_n']-1.),
                latest_policy_qualified=m['latest']['seed_qualified'],
                zero_assistance_qualified=False, hardware_qualified=False,
                final_frozen_sha256=verify(), final_training_frozen_sha256=verify_training())
            receipt_identity(result, frozen)
        except BaseException as error:
            result.update(status='ERROR_STOPPED', error=repr(error), traceback=traceback.format_exc())
            print(result['traceback'], flush=True)
        finally:
            result.update(utc_end=now(), wall_s=time.monotonic()-started)
            write(out/'result.json', result)
            write(out/'progress.json', result)
    return int(result['status']=='ERROR_STOPPED')


if __name__ == '__main__':
    raise SystemExit(main())

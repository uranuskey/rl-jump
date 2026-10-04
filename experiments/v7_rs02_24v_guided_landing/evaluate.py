"""Independent original backend: parent, guided seed, selected AND latest."""
import argparse
import json
from pathlib import Path
import re
import time
import traceback
import guided_paths
from guided_paths import HERE
from guided_runtime import exclusive, now, read, resource_limit, sha, verify, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--training', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    args = p.parse_args()
    assert re.fullmatch('[A-Za-z0-9_-]+', args.run_id)
    frozen = verify()
    train, receipt = read(args.training/'result.json'), read(args.training/'exit_receipt.json')
    assert train['status']=='BUDGET_COMPLETED' and train['completed_updates']==128
    assert train['frozen_sha256']==train['final_frozen_sha256']==receipt['frozen_sha256']==frozen
    assert receipt['process_exited'] and receipt['exit_code']==0 and receipt['stage']=='train'
    out = HERE/'runs'/args.run_id
    if out.exists():
        raise RuntimeError('Use a fresh run id')
    with exclusive(45) as resource:
        out.mkdir(parents=True)
        started = time.monotonic()
        result = dict(status='STARTING', frozen_sha256=frozen, resource=resource, utc_start=now(),
                      training_result_sha256=sha(args.training/'result.json'), num_envs=45,
                      scope='45 known simulation conditions; 24 V; 62.5 percent attitude assistance')
        try:
            import torch
            from guided_env import GuidedEnv as ParameterEnv
            from guided_learning import Policy, FrozenLaunch
            from guided_rollout import trial, compact, qualifies
            from standing_policy import JumpPolicy
            torch.set_num_threads(1)
            torch.manual_seed(104042)
            torch.cuda.manual_seed_all(104042)
            env = ParameterEnv(45, fast_backend=False, abort_dir=out/'aborts')
            standing = JumpPolicy(env.device).eval().requires_grad_(False)
            launch, policy = FrozenLaunch(env.device), Policy(env.device)
            initial = args.training/'parent_initial.pt'
            checkpoints = [('baseline', dict(path=str(initial), sha256=sha(initial), update=0)),
                           ('guided',train['guided_checkpoint']), ('selected', train['selected']['checkpoint']), ('latest', train['final_checkpoint'])]
            result['mass_kg'] = float(env.m.body_subtreemass[env.base])
            for name, checkpoint in checkpoints:
                assert sha(checkpoint['path'])==checkpoint['sha256']
                state = torch.load(checkpoint['path'], map_location='cpu', weights_only=True)
                assert state['frozen_sha256']==frozen and state['voltage_v']==24 and state['assist_strength']==.625
                policy.load_state_dict(state['model_state_dict'], strict=True)
                *_, summary = trial(env, standing, launch, policy, resource_limit(started, 2400),
                                    trace_path=out/f'{name}_traces.npz')
                write(out/f'{name}.json', summary)
                result[name] = dict(checkpoint=checkpoint, evaluation=compact(summary), qualifies=qualifies(summary))
                print(json.dumps(dict(event=name, **result[name])), flush=True)
            result.update(status='EVALUATED', final_frozen_sha256=verify(),
                trained_policy_promoted=result['selected']['checkpoint']['update']>0 and result['selected']['qualifies']
                    and result['selected']['evaluation']['mean_landing_metrics']['peak_force_n']<result['baseline']['evaluation']['mean_landing_metrics']['peak_force_n']-1.,
                ppo_improved_over_guided_seed=result['selected']['checkpoint']['update']>0 and result['selected']['qualifies']
                    and result['selected']['evaluation']['mean_landing_metrics']['peak_force_n']<result['guided']['evaluation']['mean_landing_metrics']['peak_force_n']-1.,
                latest_policy_qualified=result['latest']['qualifies'],
                zero_assistance_qualified=False, hardware_qualified=False)
        except BaseException as error:
            result.update(status='ERROR_STOPPED', error=repr(error), traceback=traceback.format_exc())
            print(result['traceback'], flush=True)
        finally:
            result.update(wall_s=time.monotonic()-started, utc_end=now())
            write(out/'result.json', result)
    return int(result['status']=='ERROR_STOPPED')


if __name__=='__main__':
    raise SystemExit(main())

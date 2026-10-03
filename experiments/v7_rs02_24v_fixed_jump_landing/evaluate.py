"""Independent 45-condition baseline/selected evaluation, after training has exited."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
import bootstrap
from bootstrap import HERE
from runtime import exclusive, sha, verify, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-result', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    args = p.parse_args()
    frozen = verify()
    trained = json.loads(args.train_result.read_text(encoding='utf-8-sig'))
    receipt = json.loads((args.train_result.parent/'exit_receipt.json').read_text(encoding='utf-8-sig'))
    assert trained['status']=='BUDGET_COMPLETED' and trained['completed_updates']==128
    assert trained['frozen_sha256']==trained['final_frozen_sha256']==frozen
    assert receipt['process_exited'] and receipt['exit_code']==0 and receipt['frozen_sha256']==frozen
    out = HERE/'runs'/args.run_id
    if out.exists():
        raise RuntimeError('Use a fresh run id')
    with exclusive(45) as resource:
        out.mkdir(parents=True)
        started = time.monotonic()
        result = dict(status='STARTING', resource=resource, frozen_sha256=frozen,
            utc_start=datetime.now(timezone.utc).isoformat(), training_result_sha256=sha(args.train_result),
            selected_is_initial=trained['selected']['checkpoint']['update']==0,
            case_scope='45 known initial conditions; no held-out or zero-assistance qualification')
        def limit():
            import torch
            if (HERE/'STOP').exists() or (HERE.parent/'v7_jump_in_place/STOP').exists():
                raise RuntimeError('STOP requested')
            if time.monotonic()-started>1800 or torch.cuda.mem_get_info()[0]<512*1024**2:
                raise RuntimeError('Evaluation resource bound')
        try:
            import torch
            from landing import LandingEnv
            from standing_policy import JumpPolicy
            from learning import Policy, FrozenLaunch
            from rollout import trial, compact, qualifies
            torch.set_num_threads(1)
            torch.manual_seed(104027)
            torch.cuda.manual_seed_all(104027)
            env = LandingEnv(45, abort_dir=out/'aborts')
            standing = JumpPolicy(env.device).eval().requires_grad_(False)
            launch, policy = FrozenLaunch(env.device), Policy(env.device)
            initial = args.train_result.parent/'initial.pt'
            checkpoints = [('baseline', dict(path=str(initial), sha256=sha(initial), update=0))]
            if not result['selected_is_initial']:
                checkpoints.append(('selected', trained['selected']['checkpoint']))
            result['mass_kg'] = float(env.m.body_subtreemass[env.base])
            for name, checkpoint in checkpoints:
                path = Path(checkpoint['path'])
                assert sha(path)==checkpoint['sha256']
                payload = torch.load(path, map_location=env.device, weights_only=False)
                assert payload['frozen_sha256']==frozen and payload['voltage_v']==24
                policy.load_state_dict(payload['model_state_dict'])
                _, summary = trial(env, standing, launch, policy, limit, trace_path=out/f'{name}_traces.npz')
                write(out/f'{name}.json', summary)
                result[name] = dict(checkpoint=checkpoint, evaluation=compact(summary), qualifies=qualifies(summary))
                print(json.dumps(dict(event=name, **result[name])), flush=True)
            if result['selected_is_initial']:
                result['selected'] = result['baseline']
            result.update(status='EVALUATED', final_frozen_sha256=verify(),
                trained_policy_promoted=not result['selected_is_initial'] and result['selected']['qualifies'],
                zero_assistance_qualified=False, hardware_qualified=False)
        except BaseException as error:
            result.update(status='ERROR_STOPPED', error=repr(error), traceback=traceback.format_exc())
            print(result['traceback'], flush=True)
        finally:
            result.update(wall_s=time.monotonic()-started, utc_end=datetime.now(timezone.utc).isoformat())
            write(out/'result.json', result)
    return int(result['status']=='ERROR_STOPPED')


if __name__=='__main__':
    raise SystemExit(main())

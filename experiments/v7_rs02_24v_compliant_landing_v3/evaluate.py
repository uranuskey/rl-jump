"""Independent native-loop evaluation: old controller, new seed, selected, latest."""
import argparse
from pathlib import Path
import time
import traceback
import compliant_paths
from compliant_runtime import HERE,read,write,verify,sha,exclusive,now,resource_limit


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--training',type=Path,required=True)
    p.add_argument('--run-id',required=True)
    args=p.parse_args()
    assert __import__('re').fullmatch('[A-Za-z0-9_-]+',args.run_id)
    frozen=verify()
    train=read(args.training/'result.json')
    receipt=read(args.training/'exit_receipt.json')
    assert train['status']=='BUDGET_COMPLETED' and train['completed_updates']==128
    assert train['frozen_sha256']==train['final_frozen_sha256']==frozen
    assert receipt['process_exited'] and receipt['exit_code']==0
    out=HERE/'runs'/args.run_id
    assert not out.exists()
    with exclusive(45) as resource:
        out.mkdir(parents=True)
        started=time.monotonic()
        result=dict(status='RUNNING',utc_start=now(),frozen_sha256=frozen,resource=resource,
                    training_result_sha256=sha(args.training/'result.json'))
        try:
            import torch
            from compliant_env import CompliantEnv
            from compliant_learning import Policy,FrozenLaunch,source
            from compliant_rollout import trial,compact,qualifies
            from standing_policy import JumpPolicy
            from guided_env import GuidedEnv
            from param_learning import Policy as ParentPolicy
            from guided_rollout import trial as parent_trial
            torch.set_num_threads(1)
            torch.manual_seed(104053)
            torch.cuda.manual_seed_all(104053)
            standing=JumpPolicy('cuda:0').eval().requires_grad_(False)
            launch=FrozenLaunch('cuda:0')
            oldenv=GuidedEnv(45,fast_backend=False,abort_dir=out/'parent_aborts')
            oldpolicy=ParentPolicy(oldenv.device)
            oldpolicy.load_state_dict(source(train['source_checkpoint']['path'])['model_state_dict'],strict=True)
            *_,summary=parent_trial(oldenv,standing,launch,oldpolicy,resource_limit(started,5400),trace_path=out/'baseline_traces.npz')
            write(out/'baseline.json',summary)
            result['baseline']=dict(checkpoint=train['source_checkpoint'],evaluation=compact(summary),qualifies=qualifies(summary))
            del oldenv,oldpolicy
            env=CompliantEnv(45,fast_backend=False,proof=True,abort_dir=out/'aborts')
            policy=Policy(env.device)
            result['mass_kg']=env.mass
            for name,checkpoint in (('seed',train['initial_checkpoint']),('selected',train['selected']['checkpoint']),('latest',train['final_checkpoint'])):
                assert sha(checkpoint['path'])==checkpoint['sha256']
                state=torch.load(checkpoint['path'],map_location='cpu',weights_only=True)
                assert state['frozen_sha256']==frozen and state['action_dim']==16
                policy.load_state_dict(state['model_state_dict'],strict=True)
                *_,summary=trial(env,standing,launch,policy,resource_limit(started,5400),trace_path=out/f'{name}_traces.npz')
                write(out/f'{name}.json',summary)
                result[name]=dict(checkpoint=checkpoint,evaluation=compact(summary),qualifies=qualifies(summary))
                print(name,summary['passed'],summary['mean_landing_metrics'],flush=True)
            force=lambda name:result[name]['evaluation']['mean_landing_metrics']['peak_force_n']
            result.update(status='EVALUATED',final_frozen_sha256=verify(),
                controller_improved=result['seed']['qualifies'] and force('seed')<force('baseline')-1,
                ppo_improved=result['selected']['checkpoint']['update']>0 and result['selected']['qualifies'] and force('selected')<force('seed')-1,
                latest_policy_qualified=result['latest']['qualifies'],zero_assistance_qualified=False,hardware_qualified=False)
        except BaseException as error:
            result.update(status='ERROR_STOPPED',error=repr(error),traceback=traceback.format_exc())
            print(result['traceback'],flush=True)
        finally:
            result.update(utc_end=now(),wall_s=time.monotonic()-started)
            write(out/'result.json',result)
    return int(result['status']=='ERROR_STOPPED')


if __name__=='__main__':
    raise SystemExit(main())

"""Physical candidate screening and independent admission before any PPO run."""
import argparse
import json
import time
import traceback
import compliant_paths
from compliant_runtime import HERE, write, read, sha, verify, exclusive, now, resource_limit, admission
from compliant_control import INITIAL


PROFILES=[]
for air in (.155,.17,.185):
    for approach in (.16,.22,.28):
        for kd in (.20,.40,.65):
            PROFILES.append(dict(name=f'h{air}_a{approach}_d{kd}',values=(air,approach,.05,.12,.60,.45,kd,.70)))


def execute(args,out,result,limit):
    import numpy as np
    import torch
    from torch.distributions import Normal
    from standing_policy import JumpPolicy
    from compliant_env import CompliantEnv
    from compliant_learning import Policy,FrozenLaunch,initialize,source
    from compliant_control import encode
    from compliant_rollout import trial,compact,qualifies
    from native_checks import readback,selective_reset
    torch.set_num_threads(1)
    torch.manual_seed(104051)
    torch.cuda.manual_seed_all(104051)
    source(args.checkpoint)
    result['source_checkpoint']=dict(path=str(args.checkpoint),sha256=sha(args.checkpoint),update=80)
    if args.mode=='parent':
        from guided_env import GuidedEnv
        from param_learning import Policy as ParentPolicy
        from guided_rollout import trial as parent_trial
        env=GuidedEnv(45,fast_backend=False,abort_dir=out/'aborts')
        standing=JumpPolicy(env.device).eval().requires_grad_(False)
        policy=ParentPolicy(env.device)
        policy.load_state_dict(source(args.checkpoint)['model_state_dict'],strict=True)
        *_,summary=parent_trial(env,standing,FrozenLaunch(env.device),policy,limit,trace_path=out/'parent_traces.npz')
        write(out/'summary.json',summary)
        assert qualifies(summary), 'Parent must still qualify'
        result.update(status='PARENT_QUALIFIED',baseline=compact(summary))
        return
    baseline=read(args.baseline)['baseline']
    result['baseline_result_sha256']=sha(args.baseline)
    result['baseline']=baseline
    env=CompliantEnv(result['num_envs'],fast_backend=args.mode=='search',proof=True,abort_dir=out/'aborts')
    standing=JumpPolicy(env.device).eval().requires_grad_(False)
    launch=FrozenLaunch(env.device)
    policy=Policy(env.device)
    initialize(policy,args.checkpoint)
    if args.mode=='search':
        candidates=[]
        for batch in range(3):
            profiles=PROFILES[batch*9:(batch+1)*9]
            table=torch.stack([encode(p['values'],env.device) for p in profiles])
            class Proposal:
                def distribution(self,obs):
                    mean=policy.actor(obs).clone()
                    mean[:,:8]=table[torch.arange(len(obs),device=obs.device)//45]
                    return Normal(mean,policy.std.expand_as(mean))
            *_,summary=trial(env,standing,launch,Proposal(),limit)
            write(out/f'batch_{batch:02d}.json',summary)
            for index,profile in enumerate(profiles):
                rows=summary['cases'][index*45:(index+1)*45]
                avg=lambda k:sum(r['landing_metrics'][k] for r in rows)/45
                passed=sum(r['passed'] for r in rows)
                retained=sum(r['wheel_cm'] for r in rows)/45>=9.065 and sum(r['com_cm'] for r in rows)/45>=11.74
                force,stroke=avg('peak_force_n'),avg('first_stop_com_drop_m')
                candidates.append(dict(**profile,batch=batch,passed=passed,retained=retained,
                    force_n=force,stroke_m=stroke,qualified=passed==45 and retained,
                    admitted=passed==45 and retained and force<=.90*baseline['mean_landing_metrics']['peak_force_n'] and stroke>=.035))
            write(out/'candidates.json',candidates)
            result['completed_batches']=batch+1
            write(out/'progress.json',result)
            print(json.dumps(dict(event='probe_batch',batch=batch,qualified=sum(x['qualified'] for x in candidates),admitted=sum(x['admitted'] for x in candidates))),flush=True)
        admitted=[x for x in candidates if x['admitted']]
        result['candidates']=candidates
        result['chosen']=min(admitted,key=lambda x:x['force_n']) if admitted else None
        result['status']='SEARCH_ADMITTED' if admitted else 'SEARCH_NO_ADMITTED_CANDIDATE'
    else:
        search=read(args.search)
        assert search['status']=='SEARCH_ADMITTED' and search['frozen_sha256']==result['frozen_sha256']
        result['chosen']=search['chosen']
        initialize(policy,args.checkpoint,result['chosen']['values'])
        *_,summary=trial(env,standing,launch,policy,limit,trace_path=out/'seed_traces.npz')
        write(out/'summary.json',summary)
        result['evaluation']=compact(summary)
        assert admission(summary,baseline), 'Independent controller admission failed'
        from audit_training import audit_trace
        result['native_trace_audit']=audit_trace(out/'seed_traces.npz',summary,env.mass)
        seed=out/'seed.pt'
        torch.save(dict(task='compliant_parameter_cushion_v2',model_state_dict=policy.state_dict(),
            update=0,frozen_sha256=result['frozen_sha256'],voltage_v=24,assist_strength=.625,
            source_checkpoint_sha256=sha(args.checkpoint),action_dim=16),seed)
        result.update(status='CONTROLLER_ADMITTED',seed=dict(path=str(seed),sha256=sha(seed),update=0))
    result['prefix_proof']=dict(status='PASS',counterfactual_physics_samples=env.proof_samples,
        mixed_world_ticks=env.proof_mixed_ticks,same_state_payload_and_fifo_exact=True)
    assert env.proof_samples>0 and env.proof_mixed_ticks>0
    result['final_frozen_sha256']=verify()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('parent','search','validate'),required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--checkpoint',type=__import__('pathlib').Path,required=True)
    p.add_argument('--baseline',type=__import__('pathlib').Path)
    p.add_argument('--search',type=__import__('pathlib').Path)
    args=p.parse_args()
    assert __import__('re').fullmatch('[A-Za-z0-9_-]+',args.run_id)
    frozen=verify()
    out=HERE/'runs'/args.run_id
    assert not out.exists(), 'Use a fresh run id'
    n=405 if args.mode=='search' else 45
    with exclusive(n) as resource:
        out.mkdir(parents=True)
        started=time.monotonic()
        result=dict(status='RUNNING',mode=args.mode,num_envs=n,resource=resource,frozen_sha256=frozen,
                    utc_start=now(),voltage_v=24,assist_strength=.625)
        write(out/'progress.json',result)
        try:
            execute(args,out,result,resource_limit(started,5400))
        except BaseException as error:
            result.update(status='ERROR_STOPPED',error=repr(error),traceback=traceback.format_exc())
            print(result['traceback'],flush=True)
        finally:
            result.update(utc_end=now(),wall_s=time.monotonic()-started)
            write(out/'result.json',result)
            write(out/'progress.json',result)
            print(json.dumps(dict(status=result['status'],mode=args.mode)),flush=True)
    return int(result['status'] in ('ERROR_STOPPED','SEARCH_NO_ADMITTED_CANDIDATE'))


if __name__=='__main__':
    raise SystemExit(main())

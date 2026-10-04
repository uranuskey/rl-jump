"""Train only after measured controller admission, then smoke and 512-world checks."""
import argparse
import json
from pathlib import Path
import re
import time
import traceback
import compliant_paths
from compliant_runtime import HERE,read,write,verify,sha,exclusive,now,resource_limit,admission


def execute(args,out,result,limit,validation):
    import torch
    from compliant_env import CompliantEnv
    from compliant_learning import Policy,PPO,FrozenLaunch,set_exploration
    from compliant_rollout import trial,compact,qualifies
    from standing_policy import JumpPolicy
    torch.set_num_threads(1)
    torch.manual_seed(104052)
    torch.cuda.manual_seed_all(104052)
    env=CompliantEnv(args.num_envs,abort_dir=out/'aborts')
    standing=JumpPolicy(env.device).eval().requires_grad_(False)
    launch,policy=FrozenLaunch(env.device),Policy(env.device)
    seed=validation['seed']
    assert sha(seed['path'])==seed['sha256']
    state=torch.load(seed['path'],map_location='cpu',weights_only=True)
    assert state['frozen_sha256']==result['frozen_sha256'] and state['action_dim']==16
    policy.load_state_dict(state['model_state_dict'],strict=True)
    ppo=PPO(policy)
    launch_state={k:v.clone() for k,v in launch.state_dict().items()}
    def save(name,update):
        path=out/name
        temporary=path.with_suffix('.pt.tmp')
        torch.save(dict(task='compliant_parameter_cushion_v2',model_state_dict=policy.state_dict(),
            actor_optimizer=ppo.actor_optimizer.state_dict(),critic_optimizer=ppo.critic_optimizer.state_dict(),
            update=update,num_envs=args.num_envs,voltage_v=24,assist_strength=.625,action_dim=16,
            frozen_sha256=result['frozen_sha256'],source_checkpoint_sha256=validation['source_checkpoint']['sha256'],
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),temporary)
        temporary.replace(path)
        return dict(path=str(path),sha256=sha(path),update=update)
    initial=save('initial.pt',0)
    *_,baseline=trial(env,standing,launch,policy,limit)
    write(out/'baseline.json',baseline)
    assert admission(baseline,validation['baseline']), 'Controller must retain admission at training batch size'
    result.update(completed_updates=0,actor_steps=0,executed_landing_plans=0,
        source_checkpoint=validation['source_checkpoint'],initial_checkpoint=initial,
        parent_baseline=validation['baseline'],baseline=compact(baseline),
        selected=dict(checkpoint=initial,evaluation=compact(baseline)),evaluations=[],
        prefix_proof=validation['prefix_proof'],frozen_launch_unchanged=True)
    print(json.dumps(dict(event='baseline',**compact(baseline))),flush=True)
    write(out/'progress.json',result)
    budget=2 if args.mode=='smoke' else 128
    for update in range(1,budget+1):
        set_exploration(policy,update-1)
        started=time.monotonic()
        obs,action,reward,eligible,summary=trial(env,standing,launch,policy,limit,stochastic=True)
        rollout_s=time.monotonic()-started
        assert bool(eligible.any()), 'No landing activation'
        ppo_started=time.monotonic()
        stats=ppo.update(obs,action,reward,eligible)
        ppo_s=time.monotonic()-ppo_started
        assert all(torch.equal(v,launch_state[k]) for k,v in launch.state_dict().items())
        result['completed_updates']=update
        result['actor_steps']+=stats['actor_steps']
        result['executed_landing_plans']+=stats['eligible_samples']
        row=dict(update=update,num_envs=args.num_envs,wall_s=time.monotonic()-started,
                 rollout_s=rollout_s,ppo_s=ppo_s,**stats,trial=compact(summary))
        result['latest_checkpoint']=save('latest.pt',update)
        result['last_update']=row
        write(out/f'trials_{update:04d}.json',summary)
        write(out/f'update_{update:04d}.json',row)
        write(out/'progress.json',result)
        print(json.dumps(row),flush=True)
        if update%8==0 or update==budget:
            checkpoint=save(f'model_{update:04d}.pt',update)
            *_,evaluation=trial(env,standing,launch,policy,limit)
            entry=dict(update=update,checkpoint=checkpoint,evaluation=compact(evaluation),qualifies=qualifies(evaluation),
                       controller_admission=admission(evaluation,validation['baseline']))
            write(out/f'evaluation_{update:04d}.json',dict(**entry,cases=evaluation['cases']))
            result['evaluations'].append(entry)
            result['latest_evaluation']=entry
            force=evaluation['mean_landing_metrics']['peak_force_n']
            if entry['controller_admission'] and force<result['selected']['evaluation']['mean_landing_metrics']['peak_force_n']-.25:
                result['selected']=entry
            print(json.dumps(dict(event='evaluation',**entry)),flush=True)
            if evaluation['passed']==0:
                raise RuntimeError('Latest deterministic evaluation has zero passing worlds')
        result['final_frozen_sha256']=verify()
        write(out/'progress.json',result)
    if args.mode=='smoke':
        assert result['latest_evaluation']['controller_admission'], 'Smoke failed controller admission'
    result.update(final_checkpoint=save('final.pt',budget),
        status='SMOKE_COMPLETED' if args.mode=='smoke' else 'BUDGET_COMPLETED')
    assert result['actor_steps']>0


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('smoke','train'),required=True)
    p.add_argument('--num-envs',type=int,choices=(45,512),required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--smoke-result',type=Path)
    args=p.parse_args()
    assert re.fullmatch('[A-Za-z0-9_-]+',args.run_id)
    assert args.num_envs==(45 if args.mode=='smoke' else 512)
    frozen=verify()
    validation=read(args.validation)
    receipt=read(args.validation.parent/'exit_receipt.json')
    assert validation['status']=='CONTROLLER_ADMITTED' and validation['final_frozen_sha256']==frozen
    assert receipt['process_exited'] and receipt['exit_code']==0 and receipt['stage']=='validate'
    if args.mode=='train':
        smoke=read(args.smoke_result)
        sr=read(args.smoke_result.parent/'exit_receipt.json')
        assert smoke['status']=='SMOKE_COMPLETED' and smoke['frozen_sha256']==frozen
        assert sr['process_exited'] and sr['exit_code']==0 and sr['stage']=='smoke'
    out=HERE/'runs'/args.run_id
    assert not out.exists(), 'Use a fresh run id'
    with exclusive(args.num_envs) as resource:
        out.mkdir(parents=True)
        started=time.monotonic()
        result=dict(status='RUNNING',mode=args.mode,num_envs=args.num_envs,resource=resource,
            frozen_sha256=frozen,utc_start=now(),voltage_v=24,assist_strength=.625,
            action_dim=16,observation_dim=17,reference_hz=400,physics_hz=400,
            initialization='admitted controller seed; new Adam; inherited observation features and attitude feedback only',
            validation_result_sha256=sha(args.validation),hardware_used=False)
        write(out/'progress.json',result)
        try:
            execute(args,out,result,resource_limit(started,2400 if args.mode=='smoke' else 43200),validation)
        except BaseException as error:
            result.update(status='ERROR_STOPPED',error=repr(error),traceback=traceback.format_exc())
            print(result['traceback'],flush=True)
        finally:
            result.update(utc_end=now(),wall_s=time.monotonic()-started)
            write(out/'result.json',result)
            write(out/'progress.json',result)
            print(json.dumps({k:result.get(k) for k in ('status','completed_updates','actor_steps','wall_s')}),flush=True)
    return int(result['status']=='ERROR_STOPPED')


if __name__=='__main__':
    raise SystemExit(main())

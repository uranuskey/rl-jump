"""One serial native executor. Frozen sources and per-update immutable logs."""
import argparse,json,time,traceback
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import bootstrap
from bootstrap import HERE
from runtime import exclusive,verify,write,sha

def trial(env,standing,policy,limit,strength,stochastic,trace_path=None):
    import torch,numpy as np
    from environment import reset_cases
    from learning import apply_action,score
    from jump_task import REASONS,SUCCESS
    assert strength==.625,'This task is fixed62.5% flat-ground landing only'
    env.assist_strength.fill_(strength);env.feedback.zero_();reset_cases(env)
    env.record=trace_path is not None;blocks=[]
    stable_seconds=torch.zeros(env.n,device=env.device)
    eligible=None;obs=None;action=None
    with torch.no_grad():
        for step in range(250):
            limit()
            if step==30:
                obs=env.plan_observation().clone();eligible=~env.terminal_mask()
                d=policy.distribution(obs);action=d.sample() if stochastic else d.mean
                apply_action(env,action)
            act=standing.actor(env.obs) if step<30 else torch.zeros(env.n,6,device=env.device)
            env.step(act,auto_reset=False)
            stable_seconds=torch.maximum(stable_seconds,env.task.recovery_ticks*.0025)
            if env.record:
                rows=env.last['traces']
                block={k:torch.stack([r[k] for r in rows]).cpu().numpy() for k in ('active','ticks','phase','reason','assist_wrench','arrived_reference_height','q','v')}
                block.update({'sensor_'+k:torch.stack([r['sample'][k] for r in rows]).cpu().numpy() for k in rows[0]['sample']})
                blocks.append(block)
            if step>=30 and bool((env.terminal_mask()|(env.ticks>=2000)).all()):break
        assert obs is not None and bool((env.terminal_mask()|(env.ticks>=2000)).all())
        reward,passed,visible=score(env.task,env.ticks,stable_seconds)
        metrics=env.task.landing_reward.metrics()
        terms=env.task.landing_reward.last_terms
        rows=[dict(world=i,case=i%45,passed=bool(passed[i]),visible=bool(visible[i]),reason=REASONS[int(env.task.reason[i])],phase=int(env.task.phase[i]),end_s=float(env.ticks[i])*.0025,wheel_cm=float(env.task.peak_clearance_m[i])*100,com_cm=float(env.task.height_score.peak[i])*100,touchdown_s=float(env.task.touchdown_time[i]),success_s=float(env.task.success_time[i]),reward=float(reward[i]),max_stable_s=float(stable_seconds[i]),plan=env.plan[i].cpu().tolist(),landing=env.landing_parameters[i].cpu().tolist(),feedback=env.feedback[i].cpu().tolist()) for i in range(env.n)]
        summary=dict(assist_strength=strength,passed=int(passed.sum()),worlds=env.n,original45_passed=int(passed[:45].sum()),mean_wheel_cm=float(env.task.peak_clearance_m.mean())*100,mean_com_cm=float(env.task.height_score.peak.mean())*100,mean_return=float(reward.mean()),reasons=dict(Counter(r['reason'] for r in rows)),cases=rows)
        for i,row in enumerate(rows):
            row['landing_metrics']={k:float(v[i]) for k,v in metrics.items()}
            row['reward_terms']={k:float(v[i]) for k,v in terms.items()}
        summary['mean_landing_metrics']={k:float(v.mean()) for k,v in metrics.items()}
        summary['mean_reward_terms']={k:float(v.mean()) for k,v in terms.items()}
        if trace_path:
            arrays={k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]};np.savez_compressed(trace_path,**arrays)
    env.record=False
    return obs,action,reward,eligible,summary

def compact(s):return {k:v for k,v in s.items() if k!='cases'}
def qualifies(s):
    return s['original45_passed']==45 and s['passed']==s['worlds'] and s['mean_wheel_cm']>=9.065 and s['mean_com_cm']>=11.74

def execute(args,out,result,limit):
    import torch
    from landing import LandingEnv
    from standing_policy import JumpPolicy
    from learning import Policy,PPO,LEVELS
    from environment import reset_cases
    from native_checks import readback,selective_reset
    torch.set_num_threads(1);torch.manual_seed(827);torch.cuda.manual_seed_all(827)
    env=LandingEnv(args.num_envs,abort_dir=out/'aborts');standing=JumpPolicy(env.device).eval()
    for p in standing.parameters():p.requires_grad_(False)
    reset_cases(env)
    with torch.no_grad():env.step(standing.actor(env.obs),auto_reset=False)
    result['readback']=readback(env,world=1);result['selective_reset']=selective_reset(env)
    torch.manual_seed(827);torch.cuda.manual_seed_all(827)
    policy=Policy(env.device);ppo=PPO(policy)
    for group in ppo.actor_optimizer.param_groups:group['lr']=.00005
    frozen={k:v.clone() for k,v in policy.actor.launch.state_dict().items()}
    def save(name,update,level):
        path=out/name
        torch.save(dict(model_state_dict=policy.state_dict(),actor_optimizer=ppo.actor_optimizer.state_dict(),critic_optimizer=ppo.critic_optimizer.state_dict(),update=update,level=level,voltage_v=24,frozen_sha256=result['frozen_sha256'],torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),path)
        return dict(path=str(path),sha256=sha(path),update=update,assist_strength=level)
    ck=save('initial.pt',0,.625)
    *_,baseline=trial(env,standing,policy,limit,.625,False)
    write(out/'baseline.json',baseline)
    assert qualifies(baseline),'Migration failed full landing baseline'
    result.update(completed_updates=0,actor_steps=0,executed_plans=0,selected=dict(checkpoint=ck,evaluation=compact(baseline)),evaluations=[],ppo_updates=0)
    print(json.dumps(dict(event='baseline',**compact(baseline))),flush=True)
    level=0 # Assistance remains fixed; optimize flat-ground landing quality.
    budget=2 if args.mode=='smoke' else 128
    for update in range(1,budget+1):
        before=time.monotonic()
        obs,action,reward,eligible,s=trial(env,standing,policy,limit,LEVELS[level],True)
        params=[p.detach().clone() for p in policy.actor.delta.parameters()]
        stats=ppo.update(obs,action,reward,eligible)
        delta=max(float((p.detach()-old).abs().max()) for p,old in zip(policy.actor.delta.parameters(),params))
        assert all(torch.equal(v,frozen[k]) for k,v in policy.actor.launch.state_dict().items())
        result['completed_updates']=result['ppo_updates']=update;result['actor_steps']+=stats['actor_steps'];result['executed_plans']+=stats['eligible_samples']
        row=dict(update=update,wall_s=time.monotonic()-before,actor_delta=delta,**stats,trial=compact(s))
        write(out/f'trials_{update:04d}.json',s);write(out/f'update_{update:04d}.json',row)
        print(json.dumps(row),flush=True)
        if update%8==0 or update==budget:
            ck=save(f'model_{update:04d}.pt',update,LEVELS[level])
            *_,ev=trial(env,standing,policy,limit,LEVELS[level],False)
            entry=dict(update=update,checkpoint=ck,evaluation=compact(ev));result['evaluations'].append(entry)
            write(out/f'evaluation_{update:04d}.json',dict(**entry,cases=ev['cases']))
            print(json.dumps(dict(event='evaluation',**entry)),flush=True)
            if qualifies(ev) and ev['mean_return']>result['selected']['evaluation']['mean_return']:
                result['selected']=entry
    result['final_checkpoint']=save('final.pt',budget,LEVELS[level]);result['final_training_assist']=LEVELS[level]
    result.update(status='SMOKE_COMPLETED' if args.mode=='smoke' else 'BUDGET_COMPLETED',final_frozen_sha256=verify(),frozen_launch_unchanged=True,hardware_used=False)
    if args.mode=='smoke':assert result['actor_steps']>0,'No accepted actor optimizer updates'

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('smoke','train'),required=True);p.add_argument('--num-envs',type=int,choices=(45,256,2048),required=True);p.add_argument('--run-id',required=True);p.add_argument('--smoke-result',type=Path);args=p.parse_args()
    frozen=verify();out=HERE/'runs'/args.run_id;assert not out.exists()
    if args.mode=='train':
        s=json.loads(args.smoke_result.read_text());assert s['status']=='SMOKE_COMPLETED' and s['frozen_sha256']==frozen
    with exclusive(args.num_envs) as resource:
        out.mkdir(parents=True);start=time.monotonic();result=dict(status='STARTING',mode=args.mode,resource=resource,frozen_sha256=frozen,utc_start=datetime.now(timezone.utc).isoformat())
        def limit():
            import torch
            if (HERE/'STOP').exists() or (HERE.parent/'v7_jump_in_place/STOP').exists():raise RuntimeError('STOP requested')
            if time.monotonic()-start>(10800 if args.mode=='train' else 900):raise RuntimeError('Wall time bound')
            if torch.cuda.mem_get_info()[0]<512*1024**2:raise RuntimeError('GPU reserve')
        try:execute(args,out,result,limit)
        except BaseException as e:result.update(status='ERROR_STOPPED',error=repr(e),traceback=traceback.format_exc());print(result['traceback'],flush=True)
        finally:
            result.update(wall_s=time.monotonic()-start,utc_end=datetime.now(timezone.utc).isoformat());write(out/'result.json',result);print(json.dumps({k:result.get(k) for k in ('status','completed_updates','actor_steps','wall_s')}),flush=True)
    return int(result['status']=='ERROR_STOPPED')
if __name__=='__main__':raise SystemExit(main())

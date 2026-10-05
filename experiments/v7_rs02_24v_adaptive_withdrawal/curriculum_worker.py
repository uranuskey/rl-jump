"""One fresh GPU process per qualification or PPO chunk; no concurrent physics."""
import argparse
import hashlib
from pathlib import Path
import time
import traceback
import curriculum_runtime as rt
from curriculum_contract import admission,qualification,learning_entry


def actor_hash(policy):
    h=hashlib.sha256()
    for key,value in sorted(policy.actor.state_dict().items()):
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def build(request,out,worlds,native=False):
    import torch
    from fix_env import ConstraintEnv
    from slot_learning import Policy,FrozenLaunch
    from standing_policy import JumpPolicy
    torch.set_num_threads(1); torch.manual_seed(105070); torch.cuda.manual_seed_all(105070)
    c=request['contract']; ck=request['checkpoint']
    assert rt.sha(ck['path'])==ck['sha256']
    state=torch.load(ck['path'],map_location='cpu',weights_only=True)
    assert state['voltage_v']==24 and state['action_dim']==16 and state['profile']==c['profile']
    if ck['sha256']!=c['source_checkpoint']['sha256']:
        assert state['curriculum_frozen_sha256']==rt.verify()
    env=ConstraintEnv(worlds,[c['profile']],before=request['before'],after=request['after'],variant='rotor5ms',
        fast_backend=not native,proof=True,abort_dir=out/'aborts')
    standing=JumpPolicy(env.device).eval().requires_grad_(False)
    launch,policy=FrozenLaunch(env.device),Policy(env.device)
    policy.load_state_dict(state['model_state_dict'],strict=True)
    return (env,standing,launch,policy),state


def sample(parts,request,out,name,limit,native=False,stochastic=False):
    from probe_rollout import trial
    from fix_audit import audit
    env,standing,launch,policy=parts
    trace=out/(name+'_traces.npz') if native else None
    obs,action,reward,eligible,s=trial(env,standing,launch,policy,limit,
        before=request['before'],after=request['after'],stochastic=stochastic,trace_path=trace)
    path=out/(name+'.json'); rt.write(path,s)
    m=rt.metrics(s); pre=s['max_pre_apex_mimic_v_rad_s']
    anchors=request['contract']['anchors_native'] if native else request['contract']['anchors_batch']
    row=dict(evidence_dir=str(out),metrics=m,summary=path.name,summary_sha256=rt.sha(path),
        strict_admission=admission(m,anchors,pre),learning_admission=admission(m,anchors,pre,True),
        pre_apex_v_rad_s=pre,max_mimic_v_rad_s=s['max_mimic_v_rad_s'],
        before=request['before'],after=request['after'],physics_receipt=env.physics_receipt,
        prefix_proof_samples=env.proof_samples,sample_kind=s['sample_kind'])
    assert env.proof_samples>0
    if native: row.update(trace=trace.name,trace_sha256=rt.sha(trace),mass_kg=env.mass,
        audits=audit(trace,s,env.mass,request['contract']['profile'],request['before'],request['after'],'rotor5ms'))
    return obs,action,reward,eligible,row


def qualify(request,out,result,limit):
    rows=dict(batch=[])
    if request.get('reuse_preflight'):
        c=request['contract']; folder=Path(c['reused_preflight']['folder'])
        assert rt.sha(folder/'result.json')==c['reused_preflight']['sha256']
        pre=rt.read(folder/'result.json')
        assert (request['before'],request['after'])==(.6125,.60)
        assert request['checkpoint']==c['source_checkpoint']
        rows['native']={**pre['native'],'evidence_dir':str(folder)}
        rows['batch'].append({**pre['batch'],'evidence_dir':str(folder),'repeat':1})
    else:
        result['phase']='native45'; rt.write(out/'progress.json',result)
        parts,state=build(request,out,45,True)
        result['actor_hash']=actor_hash(parts[3])
        *_,rows['native']=sample(parts,request,out,'native',limit,native=True)
        del parts,state
    result['qualification']=rows
    parts,state=build(request,out,512)
    result['actor_hash']=actor_hash(parts[3])
    for repeat in range(len(rows['batch'])+1,4):
        result['phase']=f'batch_{repeat}'; rt.write(out/'progress.json',result)
        *_,row=sample(parts,request,out,f'batch_{repeat}',limit)
        rows['batch'].append(dict(repeat=repeat,**row))
        rt.write(out/'progress.json',result)
    result.update(qualified=qualification(rows),learning_entry=learning_entry(rows))


def train(request,out,result,limit):
    import torch
    from slot_ppo import PPO
    parts,state=build(request,out,512)
    env,standing,launch,policy=parts
    ppo=PPO(policy)
    if request.get('resume_optimizer'):
        assert state['level_index']==request['level_index']
        ppo.actor_optimizer.load_state_dict(state['actor_optimizer'])
        ppo.critic_optimizer.load_state_dict(state['critic_optimizer'])
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
    launch_state={k:v.clone() for k,v in launch.state_dict().items()}
    result.update(actor_hash_before=actor_hash(policy),actor_steps=0,completed_updates=0,updates=[])
    for offset in range(1,request['updates']+1):
        global_update=request['global_updates']+offset
        rt.exploration(policy,global_update-1)
        started=time.monotonic()
        obs,action,reward,eligible,row=sample(parts,request,out,f'trial_{global_update:04d}',limit,stochastic=True)
        assert bool(eligible.any())
        stats=ppo.update(obs,action,reward,eligible)
        assert all(torch.equal(v,launch_state[k]) for k,v in launch.state_dict().items())
        entry=dict(global_update=global_update,sample=row,stats=stats,wall_s=time.monotonic()-started)
        result['updates'].append(entry); result['actor_steps']+=stats['actor_steps']
        result['completed_updates']=offset
        ck=out/'latest.pt'; tmp=ck.with_suffix('.pt.tmp')
        torch.save(dict(model_state_dict=policy.state_dict(),voltage_v=24,action_dim=16,profile=request['contract']['profile'],
            curriculum_frozen_sha256=result['frozen_sha256'],level_index=request['level_index'],
            before=request['before'],after=request['after'],global_update=global_update,
            actor_optimizer=ppo.actor_optimizer.state_dict(),critic_optimizer=ppo.critic_optimizer.state_dict(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),tmp)
        tmp.replace(ck)
        result['checkpoint']=dict(path=str(ck),sha256=rt.sha(ck),global_update=global_update)
        rt.write(out/'progress.json',result)
        print(dict(update=global_update,accepted_actor_steps=result['actor_steps']),flush=True)
    result['actor_hash_after']=actor_hash(policy)
    assert result['actor_steps']>0 and result['actor_hash_after']!=result['actor_hash_before'],'No changed actor; never retry identical policy until passing'


def main():
    p=argparse.ArgumentParser(); p.add_argument('--request',type=Path,required=True); args=p.parse_args()
    request=rt.read(args.request); assert request['contract']==rt.contract()
    assert request['frozen_sha256']==rt.verify()
    out=args.request.parent; assert not (out/'result.json').exists()
    with rt.exclusive(512) as resources:
        start=time.monotonic(); result=dict(status='RUNNING',mode=request['mode'],request=request,
            frozen_sha256=rt.verify(),resources=resources,utc_start=rt.now())
        rt.write(out/'progress.json',result)
        try:
            {'qualify':qualify,'train':train}[request['mode']](request,out,result,rt.limit_for(start))
            result.update(status='COMPLETED',final_frozen_sha256=rt.verify())
        except BaseException as error:
            result.update(status='ERROR_STOPPED',error=repr(error),traceback=traceback.format_exc())
            print(result['traceback'],flush=True)
        finally:
            result.update(utc_end=rt.now(),wall_s=time.monotonic()-start)
            rt.write(out/'result.json',result); rt.write(out/'progress.json',result)
    return int(result['status']!='COMPLETED')


if __name__=='__main__': raise SystemExit(main())

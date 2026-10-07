"""One fresh GPU process per qualification or PPO chunk; no concurrent physics."""
import argparse
from contextlib import nullcontext
import hashlib
from pathlib import Path
import time
import traceback
import zsnap_runtime as rt
from zsnap_contract import admission,qualification,learning_entry,learning_admission,validate_request,reuse_key,learning_evidence,EXPLORATION_OFFSET


def actor_hash(policy):
    h=hashlib.sha256()
    for key,value in sorted(policy.actor.state_dict().items()):
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def build(request,out,worlds,native=False):
    import torch
    from zsnap_env import BalancedAssistEnv
    from slot_learning import Policy,FrozenLaunch
    from standing_policy import JumpPolicy
    torch.set_num_threads(1); torch.manual_seed(105070); torch.cuda.manual_seed_all(105070)
    c=request['contract']; ck=request['checkpoint']
    assert rt.sha(ck['path'])==ck['sha256']
    state=torch.load(ck['path'],map_location='cpu',weights_only=True)
    assert state['voltage_v']==24 and state['action_dim']==16
    expected_profile=c['source_profile'] if ck['sha256']==c['source_checkpoint']['sha256'] else c['profile']
    assert state['profile']==expected_profile
    if ck['sha256']==c['source_checkpoint']['sha256']:
        assert state[c['source_frozen_field']]==c['source_parent_frozen_sha256']
    else:
        assert state['zsnap_frozen_sha256']==rt.verify()
    env=BalancedAssistEnv(worlds,[c['profile']],before=request['before'],after=request['after'],variant='rotor5ms',
        fast_backend=not native,proof=True,abort_dir=out/'aborts')
    standing=JumpPolicy(env.device).eval().requires_grad_(False)
    launch,policy=FrozenLaunch(env.device),Policy(env.device)
    policy.load_state_dict(state['model_state_dict'],strict=True)
    return (env,standing,launch,policy),state


def sample(parts,request,out,name,limit,native=False,stochastic=False):
    from probe_rollout import trial
    from zsnap_trace_audit import audit
    env,standing,launch,policy=parts
    trace=out/(name+'_traces.npz') if native else None
    obs,action,reward,eligible,s=trial(env,standing,launch,policy,limit,
        before=request['before'],after=request['after'],stochastic=stochastic,trace_path=trace)
    peak=env.external_peak.max(0).values.detach().cpu().tolist()
    external=dict(max_abs_linear_force_n=peak[0],max_abs_body_torque_nm=peak[1],
        max_abs_root_generalized_force=peak[2],min_sampled_ticks=int(env.external_sample_ticks.min()))
    s['external_wrench']=external
    s['slot_profile']=request['contract']['profile']
    s['controller_revision']='declared bounded slot motor-offset allocation'
    path=out/(name+'.json'); rt.write(path,s)
    m=rt.metrics(s); pre=s['max_pre_apex_mimic_v_rad_s']
    anchors=request['contract']['anchors_native'] if native else request['contract']['anchors_batch']
    row=dict(external_wrench=external,evidence_dir=str(out),metrics=m,summary=path.name,summary_sha256=rt.sha(path),
        strict_admission=admission(m,anchors,pre),learning_admission=learning_admission(s,anchors,pre),
        pre_apex_v_rad_s=pre,max_mimic_v_rad_s=s['max_mimic_v_rad_s'],
        before=request['before'],after=request['after'],physics_receipt=env.physics_receipt,
        max_slot_error_mm=float(env.worst_slot_error.max())*1000,min_height_mm=float(env.min_height.min())*1000,
        prefix_proof_samples=env.proof_samples,sample_kind=s['sample_kind'])
    assert env.proof_samples>0
    if not stochastic:
        import numpy as np
        pose=out/(name+'_poses.npz')
        np.savez_compressed(pose,q=env.q.detach().cpu().numpy(),v=env.v.detach().cpu().numpy(),
            ticks=env.ticks.detach().cpu().numpy(),minimum_q=env.min_height_q.detach().cpu().numpy())
        row.update(poses=pose.name,poses_sha256=rt.sha(pose))
    if native: row.update(trace=trace.name,trace_sha256=rt.sha(trace),mass_kg=env.mass,
        audits=audit(trace,s,env.mass,request['contract']['profile'],request['before'],request['after'],'rotor5ms'))
    return obs,action,reward,eligible,row


def qualify(request,out,result,limit):
    if request['reuse_source_evidence']:
        key=reuse_key(request);assert key is not None
        old=rt.source_evidence(request['contract'],key)
        assert (old['request']['before'],old['request']['after'])==(request['before'],request['after'])
        result.update(actor_hash=old['actor_hash'],qualification=old['qualification'],qualified=old['qualified'],
            learning_entry=learning_entry(old['qualification'],request['contract']),new_physical_trials=0,
            learning_evidence=learning_evidence(old['qualification'],request['contract']),
            original_source_learning_entry=old['learning_entry'],original_source_qualified=old['qualified'],
            reused_qualification=request['contract'][key],phase='CPU_SOURCE_EVIDENCE')
        return
    result['new_physical_trials']=45+3*512
    rows=dict(batch=[])
    assert not request.get('reuse_preflight')
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
    result.update(qualified=qualification(rows),learning_entry=learning_entry(rows,request['contract']),learning_evidence=learning_evidence(rows,request['contract']))




def main():
    p=argparse.ArgumentParser(); p.add_argument('--request',type=Path,required=True); args=p.parse_args()
    request=rt.read(args.request); assert request['contract']==rt.contract()
    assert request['frozen_sha256']==rt.verify()
    out=args.request.parent; assert not (out/'result.json').exists()
    validate_request(request)
    scope=nullcontext({'cpu_only_source_evidence':True}) if request['reuse_source_evidence'] else rt.exclusive(512)
    with scope as resources:
        start=time.monotonic(); result=dict(status='RUNNING',mode=request['mode'],request=request,
            frozen_sha256=rt.verify(),resources=resources,utc_start=rt.now())
        rt.write(out/'progress.json',result)
        try:
            {'qualify':qualify}[request['mode']](request,out,result,rt.limit_for(start))
            result.update(status='COMPLETED',final_frozen_sha256=rt.verify())
        except BaseException as error:
            result.update(status='ERROR_STOPPED',error=repr(error),traceback=traceback.format_exc())
            print(result['traceback'],flush=True)
        finally:
            result.update(utc_end=rt.now(),wall_s=time.monotonic()-start)
            rt.write(out/'result.json',result); rt.write(out/'progress.json',result)
    return int(result['status']!='COMPLETED')


if __name__=='__main__': raise SystemExit(main())

"""A fixed actor with a declared timing profile, evaluated once per level."""
import argparse,time,traceback
from pathlib import Path
import late_runtime as rt
from late_contract import qualification,validate_request,controller_id
from recovery_worker import sample,actor_hash

def build(request,out,worlds,native=False):
    import torch
    from boundary_env import BalancedAssistEnv
    from slot_learning import Policy,FrozenLaunch
    from standing_policy import JumpPolicy
    torch.set_num_threads(1);torch.manual_seed(105070);torch.cuda.manual_seed_all(105070)
    c=request['contract'];ck=request['checkpoint'];assert rt.sha(ck['path'])==ck['sha256']
    state=torch.load(ck['path'],map_location='cpu',weights_only=True)
    assert state['voltage_v']==24 and state['action_dim']==16 and state['profile']==c['source_profile']
    assert state['recovery_frozen_sha256']==rt.SOURCE_PARENT_SHA
    env=BalancedAssistEnv(worlds,[c['profile']],before=request['before'],after=request['after'],variant='rotor5ms',fast_backend=not native,proof=True,abort_dir=out/'aborts')
    standing=JumpPolicy(env.device).eval().requires_grad_(False)
    launch,policy=FrozenLaunch(env.device),Policy(env.device)
    policy.load_state_dict(state['model_state_dict'],strict=True)
    assert actor_hash(policy)==c['source_actor_hash']
    return (env,standing,launch,policy),state

def qualify(req,out,r,limit):
    rows=dict(batch=[]);r['qualification']=rows
    r['phase']='native45';rt.write(out/'progress.json',r)
    parts,state=build(req,out,45,True)
    r['actor_hash']=actor_hash(parts[3])
    *_,rows['native']=sample(parts,req,out,'native',limit,native=True)
    del parts,state
    parts,state=build(req,out,512)
    assert actor_hash(parts[3])==r['actor_hash']
    for repeat in range(1,4):
        r['phase']=f'batch_{repeat}';rt.write(out/'progress.json',r)
        *_,row=sample(parts,req,out,f'batch_{repeat}',limit)
        rows['batch'].append(dict(repeat=repeat,**row));rt.write(out/'progress.json',r)
    r.update(qualified=qualification(rows),learning_entry=False,training_disabled=True,
        controller_id=controller_id(r['actor_hash'],req['contract']['profile']))

def main():
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);a=p.parse_args()
    req=rt.read(a.request);validate_request(req)
    assert req['contract']==rt.case_contract(req['candidate_index']) and req['frozen_sha256']==rt.verify()
    out=a.request.parent;assert not (out/'result.json').exists()
    with rt.exclusive(512) as resources:
        start=time.monotonic();r=dict(status='RUNNING',mode='qualify',request=req,frozen_sha256=rt.verify(),resources=resources,
            utc_start=rt.now(),planned_physical_trials=45+3*512,completed_updates=0,actor_steps=0)
        try:
            qualify(req,out,r,rt.limit_for(start));r.update(status='COMPLETED',final_frozen_sha256=rt.verify())
        except BaseException as error:
            r.update(status='ERROR_STOPPED',error=repr(error),traceback=traceback.format_exc());print(r['traceback'],flush=True)
        finally:
            # Count only rollouts whose complete summary actually exists, even after a trace-audit failure.
            r['new_physical_trials']=sum(rt.read(out/(n+'.json'))['worlds'] for n in ('native','batch_1','batch_2','batch_3') if (out/(n+'.json')).exists())
            r.update(utc_end=rt.now(),wall_s=time.monotonic()-start)
            rt.write(out/'result.json',r);rt.write(out/'progress.json',r)
    return int(r['status']!='COMPLETED')
if __name__=='__main__':raise SystemExit(main())

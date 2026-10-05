"""One bounded gain scan at52.5%; no PPO or promotion from this screen."""
import argparse,time,traceback
from pathlib import Path
from collections import Counter
import allocation_bootstrap as rt
from allocation_profiles import candidates,validate_profile,choose

def verify():
    return rt.verify_search()

def execute(out,result):
    import torch,numpy as np,mujoco
    from allocation_env import AllocationEnv
    from slot_learning import Policy,FrozenLaunch
    from standing_policy import JumpPolicy
    from probe_rollout import trial
    from transfer_contract import admission
    from compliant_runtime import resource_limit
    torch.set_num_threads(1);torch.manual_seed(105070);torch.cuda.manual_seed_all(105070)
    contract,ck=rt.source()
    result.update(source_checkpoint=ck,parent_completed_course=True)
    profiles=candidates(contract['profile'])
    for p in profiles:validate_profile(contract['profile'],p)
    env=AllocationEnv(45*len(profiles),profiles,before=.525,after=.525,variant='rotor5ms',
        fast_backend=True,proof=True,abort_dir=out/'aborts')
    standing=JumpPolicy(env.device).eval().requires_grad_(False)
    policy=Policy(env.device).eval().requires_grad_(False)
    state=torch.load(ck['path'],map_location='cpu',weights_only=True)
    policy.load_state_dict(state['model_state_dict'],strict=True)
    *_,s=trial(env,standing,FrozenLaunch(env.device),policy,rt.limit_for(time.monotonic(),3600),before=.525,after=.525)
    rt.write(out/'screen.json',s)
    np.savez_compressed(out/'minimum_poses.npz',q=env.min_height_q.cpu().numpy())
    result.update(candidates=[],screen_sha256=rt.sha(out/'screen.json'),
        physics_receipt=env.physics_receipt,prefix_proof_samples=env.proof_samples)
    errors=env.worst_slot_error.cpu().tolist()
    h=env.min_height.cpu().tolist()
    for i,p in enumerate(profiles):
        rows=s['cases'][45*i:45*(i+1)]
        m=rt.metrics(dict(cases=rows,reasons=dict(Counter(x['reason'] for x in rows))))
        pre=max(x['constraint_residual_maxima']['pre_apex_v_rad_s'] for x in rows)
        entry=admission(m,contract['anchors_native'],pre,True)
        strict=admission(m,contract['anchors_native'],pre)
        result['candidates'].append(dict(profile=p,metrics=m,pre_apex_v_rad_s=pre,
            learning_admission=entry,strict_admission=strict,
            max_slot_error_mm=1000*max(errors[45*i:45*(i+1)]),
            min_actual_height_mm=1000*min(h[45*i:45*(i+1)])))
    # Prefer the smallest declared gain revision satisfying the unchanged bounds.
    result.update(status='SCREENED',selected=choose(result['candidates']),
        training_updates=0,qualification=False,independent_native_and_3x512_required=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
    assert a.run_id.replace('_','').isalnum()
    out=rt.HERE/'runs'/a.run_id;assert not out.exists();out.mkdir(parents=True)
    result=dict(status='RUNNING',utc_start=rt.now(),frozen_sha256=verify())
    rt.write(out/'progress.json',result);code=0
    with rt.exclusive(180) as resources:
        result['resources']=resources
        try:execute(out,result)
        except BaseException as e:
            result.update(status='ERROR_STOPPED',error=repr(e),traceback=traceback.format_exc())
            print(result['traceback'],flush=True);code=1
        finally:
            result.update(utc_end=rt.now(),final_frozen_sha256=verify())
            rt.write(out/'result.json',result);rt.write(out/'progress.json',result)
    return code
if __name__=='__main__':raise SystemExit(main())

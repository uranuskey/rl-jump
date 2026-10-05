"""One 225-world bounded parameter screen; never qualification or PPO."""
import argparse,time,traceback
from pathlib import Path
from collections import Counter
import repair_bootstrap as rt
from repair_profiles import candidates,validate_profile,choose
from repair_contract import admission,wrench_valid

def execute(out,result):
    import torch,numpy as np
    from zero_env import ZeroAssistEnv
    from slot_learning import Policy,FrozenLaunch
    from standing_policy import JumpPolicy
    from probe_rollout import trial
    torch.set_num_threads(1);torch.manual_seed(105070);torch.cuda.manual_seed_all(105070)
    c,ck=rt.source();profiles=candidates(c['profile'])
    for p in profiles:validate_profile(c['profile'],p)
    env=ZeroAssistEnv(45*len(profiles),profiles,before=.425,after=.425,variant='rotor5ms',
        fast_backend=True,proof=True,abort_dir=out/'aborts')
    standing=JumpPolicy(env.device).eval().requires_grad_(False)
    policy=Policy(env.device).eval().requires_grad_(False)
    state=torch.load(ck['path'],map_location='cpu',weights_only=True)
    assert state['voltage_v']==24 and state['action_dim']==16 and state['profile']==c['source_profile']
    policy.load_state_dict(state['model_state_dict'],strict=True)
    *_,s=trial(env,standing,FrozenLaunch(env.device),policy,rt.limit_for(time.monotonic(),3600),before=.425,after=.425)
    rt.write(out/'screen.json',s)
    arrays=dict(q=env.q.cpu().numpy(),v=env.v.cpu().numpy(),ticks=env.ticks.cpu().numpy(),
        minimum_q=env.min_height_q.cpu().numpy(),min_height=env.min_height.cpu().numpy(),
        slot_error=env.worst_slot_error.cpu().numpy(),external_peak=env.external_peak.cpu().numpy(),
        external_ticks=env.external_sample_ticks.cpu().numpy())
    np.savez_compressed(out/'poses.npz',**arrays)
    result.update(source_checkpoint=ck,candidates=[],screen_sha256=rt.sha(out/'screen.json'),
        poses_sha256=rt.sha(out/'poses.npz'),physics_receipt=env.physics_receipt,
        prefix_proof_samples=env.proof_samples,before=.425,after=.425)
    assert env.proof_samples>0
    for i,p in enumerate(profiles):
        sl=slice(45*i,45*(i+1));rows=s['cases'][sl]
        m=rt.metrics(dict(cases=rows,reasons=dict(Counter(x['reason'] for x in rows))))
        pre=max(x['constraint_residual_maxima']['pre_apex_v_rad_s'] for x in rows)
        peak=arrays['external_peak'][sl].max(0).tolist()
        external=dict(max_abs_linear_force_n=peak[0],max_abs_body_torque_nm=peak[1],
            max_abs_root_generalized_force=peak[2],min_sampled_ticks=int(arrays['external_ticks'][sl].min()))
        result['candidates'].append(dict(profile=p,metrics=m,pre_apex_v_rad_s=pre,
            learning_admission=admission(m,c['anchors_native'],pre,True),
            strict_admission=admission(m,c['anchors_native'],pre),external_wrench=external,
            external_valid=wrench_valid(dict(before=.425,after=.425,external_wrench=external)),
            max_slot_error_mm=1000*float(arrays['slot_error'][sl].max()),
            min_actual_height_mm=1000*float(arrays['min_height'][sl].min())))
    result.update(status='SCREENED',selected=choose(result['candidates']),training_updates=0,
        qualification=False,independent_native_and_3x512_required=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
    assert a.run_id.replace('_','').isalnum()
    out=rt.HERE/'runs'/a.run_id;assert not out.exists();out.mkdir(parents=True)
    result=dict(status='RUNNING',utc_start=rt.now(),frozen_sha256=rt.verify())
    rt.write(out/'progress.json',result);code=0
    with rt.exclusive(225) as resources:
        result['resources']=resources
        try:execute(out,result)
        except BaseException as e:
            result.update(status='ERROR_STOPPED',error=repr(e),traceback=traceback.format_exc())
            print(result['traceback'],flush=True);code=1
        finally:
            result.update(utc_end=rt.now(),final_frozen_sha256=rt.verify())
            rt.write(out/'result.json',result);rt.write(out/'progress.json',result)
    return code

if __name__=='__main__':raise SystemExit(main())

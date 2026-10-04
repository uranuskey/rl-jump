"""Bounded deterministic slot-tracking landing experiment over immutable v3 physics."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PARENT=HERE.parent/'v7_rs02_24v_compliant_landing_v3'
IMPACT=HERE.parent/'v7_rs02_24v_impact_spread_probe'
sys.path.insert(0,str(PARENT))
import compliant_paths
sys.path.insert(0,str(IMPACT))
from compliant_runtime import verify,sha,read,write,exclusive,now,resource_limit
from spread_probe import SOURCE_SHA,SOURCE_FROZEN,change_parameters,metrics,load_controller_audit
PRE=HERE.parent/'v7_rs02_24v_precontact_landing_probe'
sys.path.insert(0,str(PRE))
from slot_control import PROFILES
from slot_env import SlotEnv
from slot_audit import slot_audit


def code_hashes():
    files=[HERE/name for name in ('slot_probe.py','slot_env.py','slot_control.py','slot_learning.py','slot_audit.py','test_slot.py','run_probe.ps1')]
    files += [PRE/name for name in ('precontact_control.py','analyze_precontact.py')]
    files += [HERE.parent/'v7_rs02_24v_geometry_pose_probe/cad_path.json']
    files += [IMPACT/'spread_probe.py',IMPACT/'analyze_saved.py']
    return {str(f.relative_to(ROOT)):sha(f) for f in files}


def eligible(candidate,base):
    return (candidate['passed']==45 and candidate['mean_com_stroke_m']>=.05
        and candidate['mean_force_n']<=.97*base['mean_force_n']
        and candidate['max_force_n']<=base['max_force_n']
        and candidate['mean_wheel_cm']>=.99*base['mean_wheel_cm']
        and candidate['mean_com_cm']>=.99*base['mean_com_cm']
        and candidate['mean_recovery_s']<=base['mean_recovery_s']+.10)


def seed_qualified(candidate,base):
    # This admits a stable PPO starting point, not an improved policy.
    return (candidate['passed']==candidate['worlds'] and candidate['mean_com_stroke_m']>=.05
        and candidate['mean_force_n']<=1.02*base['mean_force_n']
        and candidate['max_force_n']<=1.02*base['max_force_n']
        and candidate['mean_wheel_cm']>=.99*base['mean_wheel_cm']
        and candidate['mean_com_cm']>=.99*base['mean_com_cm']
        and candidate['mean_recovery_s']<=base['mean_recovery_s']+.10)


def force_windows(path):
    import numpy as np
    rows=[]
    with np.load(path) as z:
        for w in range(z['active'].shape[1]):
            live=z['active'][:,w]
            phase=z['phase'][live,w]
            forces=z['sensor_wheel_force_n'][live,w]
            hits=np.flatnonzero((np.r_[0,phase[:-1]]==2)&(forces.max(1)>=1))
            if not len(hits):
                rows.append(dict(world=w,touched=False))
                continue
            t=int(hits[0]); f=forces.sum(1)[t:]
            rows.append(dict(world=w,touched=True,first_sample_n=float(f[0]),
                first20_peak_n=float(f[:8].max()),later_peak_n=float(f[8:].max(initial=0)),
                overall_peak_n=float(f.max()),peak_after_touch_ms=float(f.argmax()*2.5)))
    valid=[r for r in rows if r['touched']]
    means={k:float(np.mean([r[k] for r in valid])) for k in
           ('first_sample_n','first20_peak_n','later_peak_n','overall_peak_n','peak_after_touch_ms')}
    return dict(rows=rows,means=means,under300=sum(r['overall_peak_n']<=300 for r in valid),
                under250=sum(r['overall_peak_n']<=250 for r in valid))


def schedule_audit(path,profile,protect_landing=False):
    import numpy as np
    import torch
    from velocity_contract import reference_motor_velocity
    with np.load(path) as z:
        live=z['active'] & z['landing_control_enabled']
        active=z['sensor_pre_active'] & live
        requested=z['requested_payload']
        for i,key in ((12,'kp'),(13,'kd')):
            assert np.array_equal(requested[...,i][live],z['sensor_pre_requested_'+key][live])
        assert np.array_equal(requested[...,11][live],z['sensor_pre_original_force'][live])
        post=live & (z['sensor_pre_touchdown_s']>0) & protect_landing
        u=np.clip((profile['brake_start_m']-z['sensor_pre_leg_height_m'].min(-1)) /
            (profile['brake_start_m']-profile['brake_full_m']),0,1)
        depth=np.where(post,u*u*(3-2*u),0)
        assert np.allclose(depth[live],z['sensor_pre_landing_brake'][live],atol=1e-6)
        for i,key,fraction in ((12,'kp',profile['brake_kp_extra']),(13,'kd',profile['brake_kd_extra'])):
            expected_gain=np.minimum(1.,z['sensor_pre_original_'+key]*(1+fraction*depth))
            assert np.allclose(requested[...,i][live],expected_gain[live],atol=1e-6)
            assert np.array_equal(requested[...,i][live & ~post],z['sensor_pre_original_'+key][live & ~post])
        height=z['sensor_pre_requested_height_m']
        velocity=z['sensor_pre_requested_velocity_mps']
        assert np.max(abs((.18+.03*requested[...,6]-height)[live]))<1e-6
        expected=reference_motor_velocity(torch.from_numpy(height[live]),torch.from_numpy(velocity[live])).numpy()
        assert np.allclose(expected,requested[...,7:11][live],atol=2e-5,rtol=1e-5)
        offset=z['sensor_pre_offset_m'];dv=z['sensor_pre_offset_velocity_mps']
        assert np.all(offset<=1e-7) and np.all(offset>=-profile['extra_limit_m']-1e-7)
        assert np.all(offset[~active]==0) and np.all(dv[~active]==0)
        assert np.all(z['sensor_pre_touchdown_s'][active]<=0)
        assert np.all(z['sensor_pre_com_vz_mps'][active]<-.15)
        assert np.allclose(height,z['sensor_pre_original_height_m']+offset,atol=1e-7)
        assert np.allclose(velocity,z['sensor_pre_original_velocity_mps']+dv,atol=1e-7)
        assert np.all(height[active]>=.160-1e-7)
        previous=np.concatenate((np.zeros_like(offset[:1]),offset[:-1]),axis=0)
        expected_dv=(offset-previous)/.0025
        assert np.allclose(expected_dv[active],dv[active],atol=1e-6)
        clearance=z['sensor_pre_absolute_clearance_m']
        assert np.allclose(clearance[1:][live[1:]],z['sensor_wheel_clearance_m'][:-1][live[1:]],atol=1e-7)
        measured=(clearance[2:]-clearance[:-2])/.005
        assert np.allclose(measured[live[2:]],z['sensor_pre_wheel_vz_mps'][2:][live[2:]],atol=2e-5)
        if not profile['enabled']:
            assert not active.any() and np.all(offset==0) and np.all(dv==0)
        maximum_extra=float(-offset.min())
    return dict(status='PASS',air_impedance_and_all_support_unchanged=True,
        landing_impedance_protection=protect_landing,protection_formula_recomputed=True,
        causal_clearance_and_velocity_verified=True,precontact_only=True,
        requested_velocity_matches_reference=True,max_extra_retraction_m=maximum_extra,
        baseline_exact=not profile['enabled'] and not protect_landing,
        precontact_disabled=not profile['enabled'],proximity_input='ideal simulated measurement')


def execute(args,out,result,limit):
    import numpy as np
    import torch
    import warp as wp
    from torch.distributions import Normal
    from compliant_env import CompliantEnv
    from compliant_control import tick
    from compliant_servo import payload
    from velocity_contract import reference_motor_velocity
    from compliant_learning import FrozenLaunch
    from slot_learning import Policy
    from compliant_rollout import trial,compact
    from standing_policy import JumpPolicy
    from analyze_saved import contact_metrics
    from audit_training import audit_trace
    from precontact_control import modulate,table,initial
    from analyze_precontact import analyze
    torch.set_num_threads(1)
    torch.manual_seed(105056)
    torch.cuda.manual_seed_all(105056)
    assert sha(args.checkpoint)==SOURCE_SHA
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    assert ck['update']==56 and ck['frozen_sha256']==SOURCE_FROZEN
    assert ck['voltage_v']==24 and ck['assist_strength']==.625
    grid=PROFILES
    profiles=grid if args.mode=='search' else [grid[args.profile]]
    if args.search:
        search=read(args.search)
        ranked=search['ranked_for_native']
        if args.rank>=len(ranked):
            result.update(status='SKIPPED_NO_PASSING_PROFILE')
            return
        profiles=[ranked[args.rank]['profile']]
    if args.profile_result:
        prior=read(args.profile_result)
        assert prior['status']=='EVALUATED'
        profiles=prior['profiles']

    env=SlotEnv(45*len(profiles),profiles,height_m=args.height_m,fast_backend=args.mode=='search',proof=True,abort_dir=out/'aborts')
    standing=JumpPolicy(env.device).eval().requires_grad_(False)
    policy=Policy(env.device).eval().requires_grad_(False)
    policy.load_state_dict(ck['model_state_dict'],strict=True)
    trace=out/'traces.npz' if args.mode=='native' else None
    *_,summary=trial(env,standing,FrozenLaunch(env.device),policy,limit,trace_path=trace)
    summary['controller_revision']='encoder-based fore/aft slot tracking with bounded motor offsets and optional precontact retraction'
    write(out/'summary.json',summary)
    result.update(profiles=profiles,source_checkpoint=dict(path=str(args.checkpoint),sha256=SOURCE_SHA,update=56),
        mass_kg=env.mass,terrain=env.terrain_receipt,all_metrics=metrics(summary),evaluation=compact(summary),
        prefix_proof_samples=env.proof_samples,launch_fixed=True,training_updates=0,
        landing_protection=dict(enabled=True,parameters_in_profiles=True,
            capped_at_standing_gains=True,old_collision_gate_unchanged=True),
        proximity_sensor=dict(kind='ideal simulated clearance, causal backward-difference velocity',
            actual_fifo_delay_exposed=False,hardware_qualified=False,noise_tested=False))
    assert env.proof_samples>0
    if args.mode=='search':
        groups=[]
        for i,p in enumerate(profiles):
            cases=summary['cases'][i*45:(i+1)*45]
            m=metrics(dict(cases=cases,reasons=dict(Counter(r['reason'] for r in cases))))
            groups.append(dict(profile=p,metrics=m))
        for g in groups:
            g['admitted']=eligible(g['metrics'],groups[0]['metrics'])
        result.update(status='SEARCH_COMPLETE',candidates=groups,
            chosen=min((g for g in groups if g['admitted']),key=lambda g:g['metrics']['mean_force_n'],default=None),
            ranked_for_native=sorted((g for g in groups[1:] if g['metrics']['passed']==45),
                key=lambda g:g['metrics']['mean_force_n'])[:2])
    else:
        result.update(status='EVALUATED',force_timing=contact_metrics(trace),force_windows=force_windows(trace),
            physical_audit=audit_trace(trace,summary,env.mass),controller_audit=load_controller_audit()(trace,summary),
            schedule_audit=schedule_audit(trace,profiles[0],True),slot_audit=slot_audit(trace,profiles[0]),precontact_analysis=analyze(trace))
        if args.height_m:
            with np.load(trace) as z:
                error=float(np.abs(z['sensor_landing_surface_z_m'][504:]-args.height_m).max())
                supported=(z['sensor_wheel_force_n'][-200:].min(2)>=1).all(0)
                gap=np.abs(z['sensor_clearance_above_surface_m'][-200:]).max(axis=(0,2))
                good=supported & (gap<.004)
                result['surface_support']=dict(worlds=int(good.sum()),actual_plane_error_m=error,max_gap_m=float(gap.max()))
                assert error<1e-7 and (int(good.sum())==45 or summary['passed']!=45)
        if args.baseline:
            result['admitted_vs_native_baseline']=eligible(result['all_metrics'],read(args.baseline)['all_metrics'])
            result['training_seed_qualified']=seed_qualified(result['all_metrics'],read(args.baseline)['all_metrics'])
        result['targets']=dict(mean_peak_le300=result['all_metrics']['mean_force_n']<=300,
            mean_peak_le250=result['all_metrics']['mean_force_n']<=250,all_peaks_le250=result['all_metrics']['max_force_n']<=250,
            all45_pass=result['all_metrics']['passed']==45)
    result['final_frozen_sha256']=verify()
    result['final_source_sha256']=code_hashes()
    assert result['final_frozen_sha256']==SOURCE_FROZEN
    assert result['final_source_sha256']==result['source_sha256']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['search','native'],required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--profile',type=int,choices=range(len(PROFILES)),default=0)
    p.add_argument('--search',type=Path)
    p.add_argument('--profile-result',type=Path)
    p.add_argument('--rank',type=int,choices=[0,1],default=0)
    p.add_argument('--baseline',type=Path)
    p.add_argument('--height-m',type=float,choices=[0.,.01],default=0.)
    
    a=p.parse_args()
    assert __import__('re').fullmatch('[A-Za-z0-9_-]+',a.run_id)
    assert not a.height_m or a.mode=='native'
    grid=PROFILES
    assert a.profile<len(grid)
    assert verify()==SOURCE_FROZEN
    out=HERE/'runs'/a.run_id
    assert not out.exists(),'Use a new run id'
    n=45*len(grid) if a.mode=='search' else 45
    with exclusive(n) as resources:
        out.mkdir(parents=True)
        start=time.monotonic()
        result=dict(status='RUNNING',utc_start=now(),mode=a.mode,num_envs=n,resources=resources,
            voltage_v=24,assist_strength=.625,training_updates=0,source_sha256=code_hashes(),frozen_sha256=SOURCE_FROZEN)
        write(out/'progress.json',result)
        try:
            execute(a,out,result,resource_limit(start,1800))
        except BaseException as e:
            result.update(status='ERROR_STOPPED',error=repr(e),traceback=traceback.format_exc())
            print(result['traceback'],flush=True)
        finally:
            result.update(utc_end=now(),wall_s=time.monotonic()-start)
            write(out/'result.json',result)
            write(out/'progress.json',result)
            print(json.dumps({k:result.get(k) for k in ('status','all_metrics','chosen','targets')}),flush=True)
    return int(result['status']=='ERROR_STOPPED')


if __name__=='__main__':
    raise SystemExit(main())


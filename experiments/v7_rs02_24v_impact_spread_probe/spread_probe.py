"""Bounded parameter screening and unseen +10 mm landing-height perturbation.

No optimizer, policy noise, hardware changes, or writes to frozen source files.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = HERE.parent/'v7_rs02_24v_compliant_landing_v3'
sys.path.insert(0,str(PARENT))
import compliant_paths
from compliant_runtime import verify, sha, write, read, exclusive, now, resource_limit

SOURCE_SHA = 'e8b0e8a3bb80455cb72f1ee18f841883c80c98a72af3e0bb6123f500c49e095d'
SOURCE_FROZEN = '4ef61ca9d5689005391c487fe7bca14bf95b31b11f95db079040d65d5b9b1b8f'
PROFILES = [
    dict(name='selected56',air_delta=0.,time_delta=0.,kd_ratio=1.),
    dict(name='air_minus5mm',air_delta=-.005,time_delta=0.,kd_ratio=1.),
    dict(name='advance8ms',air_delta=0.,time_delta=-.008,kd_ratio=1.),
    dict(name='advance16ms',air_delta=0.,time_delta=-.016,kd_ratio=1.),
    dict(name='damping80',air_delta=0.,time_delta=0.,kd_ratio=.8),
    dict(name='air5_advance8',air_delta=-.005,time_delta=-.008,kd_ratio=1.),
    dict(name='air5_damping80',air_delta=-.005,time_delta=0.,kd_ratio=.8),
    dict(name='advance8_damping80',air_delta=0.,time_delta=-.008,kd_ratio=.8),
    dict(name='advance8_damping120',air_delta=0.,time_delta=-.008,kd_ratio=1.2),
]


def change_parameters(raw, profiles):
    import torch
    from compliant_control import decode, encode, LOW, HIGH
    values,_=decode(raw)
    group=torch.arange(len(raw),device=raw.device)//45
    table=raw.new_tensor([[p['air_delta'],p['time_delta'],p['kd_ratio']] for p in profiles])[group]
    values[:,0]+=table[:,0]
    values[:,1]+=table[:,1]
    values[:,6]*=table[:,2]
    assert bool(((values>=raw.new_tensor(LOW)) & (values<=raw.new_tensor(HIGH))).all()), 'Profile outside original bounds'
    changed=(table[:,0]!=0)|(table[:,1]!=0)|(table[:,2]!=1)
    result=raw.clone()
    result[:,:8]=torch.where(changed[:,None],encode(values,raw.device),raw[:,:8])
    return result


def metrics(summary):
    rows=summary['cases']
    avg=lambda k:sum(r['landing_metrics'][k] for r in rows)/len(rows)
    return dict(passed=sum(r['passed'] for r in rows),worlds=len(rows),
        mean_force_n=avg('peak_force_n'),max_force_n=max(r['landing_metrics']['peak_force_n'] for r in rows),
        mean_com_stroke_m=avg('first_stop_com_drop_m'),
        min_com_stroke_m=min(r['landing_metrics']['first_stop_com_drop_m'] for r in rows),
        mean_wheel_cm=sum(r['wheel_cm'] for r in rows)/len(rows),
        mean_com_cm=sum(r['com_cm'] for r in rows)/len(rows),
        mean_recovery_s=sum(r['success_s']-r['touchdown_s'] for r in rows)/len(rows),
        min_stable_s=min(r['landing_metrics']['best_continuous_stable_s'] for r in rows),
        max_rebound_mps=max(r['landing_metrics']['peak_rebound_vz_mps'] for r in rows),
        reasons=summary.get('reasons',{}))


def admitted(candidate,baseline):
    return (candidate['passed']==45 and candidate['mean_com_stroke_m']>=.05
        and candidate['mean_force_n']<=.97*baseline['mean_force_n']
        and candidate['max_force_n']<=baseline['max_force_n']
        and candidate['mean_wheel_cm']>=.99*baseline['mean_wheel_cm']
        and candidate['mean_com_cm']>=.99*baseline['mean_com_cm']
        and candidate['mean_recovery_s']<=baseline['mean_recovery_s']+.10)


def execute(args,out,result,limit):
    import numpy as np
    import torch
    import warp as wp
    from torch.distributions import Normal
    from compliant_env import CompliantEnv
    from compliant_learning import Policy,FrozenLaunch
    from compliant_rollout import trial,compact
    from standing_policy import JumpPolicy
    from analyze_saved import contact_metrics
    from audit_training import audit_trace
    from audit import controller_trace
    torch.set_num_threads(1)
    torch.manual_seed(104056)
    torch.cuda.manual_seed_all(104056)
    assert sha(args.checkpoint)==SOURCE_SHA
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    assert checkpoint['update']==56 and checkpoint['frozen_sha256']==SOURCE_FROZEN
    assert checkpoint['voltage_v']==24 and checkpoint['assist_strength']==.625
    profiles=PROFILES if args.mode=='search' else [PROFILES[args.profile]]
    if args.search:
        search=read(args.search)
        if search['chosen'] is None:
            result.update(status='SKIPPED_NO_ADMITTED_PROFILE')
            return
        profiles=[search['chosen']['profile']]

    class ProbeEnv(CompliantEnv):
        def __init__(self,n,**kw):
            self.policy_calls=0
            self.terrain_activated=False
            self.terrain_receipt=None
            super().__init__(n,**kw)
            self.ground=self.m.geom('ground').id
            self.floor_positions=wp.to_torch(self.gm.geom_pos)

        def reset(self,mask,**kw):
            if hasattr(self,'floor_positions'):
                assert bool(mask.all()), 'Diagnostic only supports full reset'
                if self.floor_positions.ndim==3:
                    self.floor_positions[:,self.ground,2]=0.
                else:
                    self.floor_positions[self.ground,2]=0.
            self.policy_calls=0
            self.terrain_activated=False
            return super().reset(mask,**kw)

        def step(self,standing_actions,**kw):
            # One global change while every world is well airborne; not an actor input.
            if args.height_m and self.policy_calls==63:
                assert bool(self.task.height_score.apex.all())
                assert float(self.support.max())<=.5
                assert float(self.clearance.min())>args.height_m+.03
                before_q,before_v=self.q.clone(),self.v.clone()
                if self.floor_positions.ndim==3:
                    self.floor_positions[:,self.ground,2]=args.height_m
                elif self.floor_positions.ndim==2:
                    self.floor_positions[self.ground,2]=args.height_m
                else:
                    raise RuntimeError('Unexpected geom_pos shape')
                self.terrain_activated=True
                self.terrain_receipt=dict(at_s=1.26,height_m=args.height_m,
                    minimum_wheel_gap_before_m=float(self.clearance.min()),
                    initial_qv_unchanged=torch.equal(before_q,self.q) and torch.equal(before_v,self.v),
                    input_to_policy=False,mode='airborne landing-height perturbation; not a static stair edge')
            self.policy_calls+=1
            output=super().step(standing_actions,**kw)
            if args.height_m and self.policy_calls==64:
                self.terrain_receipt['first_step_force_max_n']=float(self.support.max())
                assert float(self.support.max())<=.5, 'Terrain placement injected contact'
            return output

        def sensors(self,motor=None):
            x=super().sensors(motor)
            # Preserve original jump height datum; report the new surface clearance separately.
            height=args.height_m if self.terrain_activated else 0.
            x['landing_surface_z_m']=torch.full((self.n,),height,device=self.device)
            x['clearance_above_surface_m']=x['wheel_clearance_m']-height
            return x

    env=ProbeEnv(45*len(profiles),fast_backend=args.mode=='search',proof=True,abort_dir=out/'aborts')
    standing=JumpPolicy(env.device).eval().requires_grad_(False)
    policy=Policy(env.device).eval().requires_grad_(False)
    policy.load_state_dict(checkpoint['model_state_dict'],strict=True)
    class Proposal:
        def distribution(self,obs):
            mean=change_parameters(policy.actor(obs),profiles)
            return Normal(mean,policy.std.expand_as(mean))
    trace=out/'traces.npz' if args.mode=='native' else None
    *_,summary=trial(env,standing,FrozenLaunch(env.device),Proposal(),limit,trace_path=trace)
    write(out/'summary.json',summary)
    result.update(profiles=profiles,source_checkpoint=dict(path=str(args.checkpoint),sha256=SOURCE_SHA,update=56),
        terrain=env.terrain_receipt,all_metrics=metrics(summary),evaluation=compact(summary),
        prefix_proof=dict(samples=env.proof_samples,same_state_parameter_prefix_unchanged=True))
    assert env.proof_samples>0
    if args.mode=='search':
        groups=[]
        for i,profile in enumerate(profiles):
            data=metrics(dict(cases=summary['cases'][45*i:45*(i+1)]))
            groups.append(dict(profile=profile,metrics=data))
        base=groups[0]['metrics']
        for g in groups:
            g['admitted']=admitted(g['metrics'],base)
        result.update(candidates=groups,chosen=min((g for g in groups if g['admitted']),
            key=lambda g:g['metrics']['mean_force_n'],default=None),status='SEARCH_COMPLETE')
    else:
        result['force_timing']=contact_metrics(trace)
        result['physical_audit']=audit_trace(trace,summary,env.mass)
        result['controller_audit']=controller_trace(trace,summary)
        result['status']='EVALUATED'
        if args.height_m:
            with np.load(trace) as z:
                contact=z['sensor_wheel_force_n'][-200:].min(2)>=1
                gap=z['sensor_clearance_above_surface_m'][-200:]
                good=contact.all(0)&(np.abs(gap).max(axis=(0,2))<.004)
                result['surface_support']=dict(worlds=int(good.sum()),expected=45,
                    max_abs_final_gap_m=float(np.abs(gap).max()))
                assert int(good.sum())==45 or summary['passed']!=45, 'Passed without actual elevated support'
        if args.baseline:
            base=read(args.baseline)
            result['admitted_vs_native_flat']=admitted(result['all_metrics'],base['all_metrics'])
            with np.load(trace) as z, np.load(args.baseline.parent/'traces.npz') as old:
                # Absolute native floating-point differences, not a bitwise guarantee.
                result['pre_terrain_comparison']=dict(
                    q_max_abs=float(np.abs(z['q'][:504]-old['q'][:504]).max()),
                    v_max_abs=float(np.abs(z['v'][:504]-old['v'][:504]).max()),
                    same_profile=profiles==base['profiles'])
    result['final_parent_frozen_sha256']=verify()
    assert result['final_parent_frozen_sha256']==SOURCE_FROZEN


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['search','native'],required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--profile',type=int,choices=range(len(PROFILES)),default=0)
    p.add_argument('--search',type=Path)
    p.add_argument('--baseline',type=Path)
    p.add_argument('--height-m',type=float,choices=[0.,.01],default=0.)
    args=p.parse_args()
    assert __import__('re').fullmatch('[A-Za-z0-9_-]+',args.run_id)
    assert not args.height_m or args.mode=='native'
    out=HERE/'runs'/args.run_id
    assert not out.exists(), 'Use fresh run id'
    assert verify()==SOURCE_FROZEN
    n=405 if args.mode=='search' else 45
    with exclusive(n) as resource:
        out.mkdir(parents=True)
        started=time.monotonic()
        result=dict(status='RUNNING',mode=args.mode,num_envs=n,resource=resource,utc_start=now(),
            voltage_v=24,assist_strength=.625,training_updates=0,
            source_sha256={f.name:sha(f) for f in HERE.glob('*.py')},parent_frozen_sha256=SOURCE_FROZEN)
        write(out/'progress.json',result)
        try:
            execute(args,out,result,resource_limit(started,1800))
        except BaseException as e:
            result.update(status='ERROR_STOPPED',error=repr(e),traceback=traceback.format_exc())
            print(result['traceback'],flush=True)
        finally:
            result.update(utc_end=now(),wall_s=time.monotonic()-started)
            write(out/'result.json',result)
            write(out/'progress.json',result)
            print(json.dumps({k:result.get(k) for k in ['status','all_metrics','chosen','surface_support']}),flush=True)
    return int(result['status']=='ERROR_STOPPED')


if __name__=='__main__':
    raise SystemExit(main())

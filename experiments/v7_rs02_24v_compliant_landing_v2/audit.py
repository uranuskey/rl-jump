"""Audit actual exits, independently reconstructed stroke, and all FIFO channels."""
import argparse
from pathlib import Path
import numpy as np
import compliant_paths
from compliant_runtime import read,write,verify,sha
from audit_training import audit_trace


def controller_trace(path,summary):
    with np.load(path) as z:
        requested,arrived=z['requested_payload'],z['arrived_payload']
        assert requested.shape[-1]==arrived.shape[-1]==15
        assert z['landing_action'].shape[-1]==16
        assert np.all(z['landing_action'][~z['landing_control_enabled']]==0)
        results=[]
        for w,row in enumerate(summary['cases']):
            delay=(w%45)//5
            expected=np.zeros_like(arrived[:,w])
            if delay:
                expected[delay:]=requested[:-delay,w]
            else:
                expected[:]=requested[:,w]
            assert np.array_equal(expected,arrived[:,w]), 'Reference or impedance bypassed FIFO'
            active=z['active'][:,w]
            phase=z['phase'][active,w]
            force=z['sensor_wheel_force_n'][active,w]
            vz=z['sensor_com_vz_mps'][active,w]
            height=z['sensor_com_z_m'][active,w]
            touches=np.flatnonzero((np.r_[0,phase[:-1]]==2)&(force.max(-1)>=1))
            if not len(touches):
                assert not row['passed']
                continue
            touch=touches[0]
            stops=np.flatnonzero(vz[touch:touch+101]>=-.05)
            last=touch+stops[0] if len(stops) else min(touch+100,len(height)-1)
            stroke=max(0.,float(height[touch-1]-height[touch:last+1].min()))
            assert abs(stroke-row['landing_metrics']['first_stop_com_drop_m'])<1e-5
            assert bool(len(stops))==bool(row['landing_metrics']['first_stop_observed'])
            results.append(dict(case=w,stroke_m=stroke,kd_before_touch=float(z['impedance_kd'][touch-1,w])))
        return dict(status='PASS',fifo_exact=True,stroke_recomputed=True,rows=results)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--training',type=Path,required=True)
    p.add_argument('--evaluation',type=Path,required=True)
    args=p.parse_args()
    frozen=verify()
    train,ev=read(args.training/'result.json'),read(args.evaluation/'result.json')
    assert train['status']=='BUDGET_COMPLETED' and train['completed_updates']==128 and ev['status']=='EVALUATED'
    assert train['frozen_sha256']==train['final_frozen_sha256']==ev['final_frozen_sha256']==frozen
    for folder,stage in ((args.training,'train'),(args.evaluation,'evaluate')):
        receipt=read(folder/'exit_receipt.json')
        assert receipt['process_exited'] and receipt['exit_code']==0 and receipt['stage']==stage and receipt['frozen_sha256']==frozen
    updates=[read(args.training/f'update_{i:04d}.json') for i in range(1,129)]
    assert all(r['update']==i+1 and r['num_envs']==512 and r['trial']['parameter_plan_unchanged'] and r['trial']['launch_plan_unchanged'] for i,r in enumerate(updates))
    assert sum(r['actor_steps'] for r in updates)==train['actor_steps']>0
    assert sum(r['eligible_samples'] for r in updates)==train['executed_landing_plans']>0
    audits={}
    for name in ('baseline','seed','selected','latest'):
        checkpoint=ev[name]['checkpoint']
        assert sha(checkpoint['path'])==checkpoint['sha256']
        summary=read(args.evaluation/f'{name}.json')
        trace=args.evaluation/f'{name}_traces.npz'
        audits[name]=dict(physics=audit_trace(trace,summary,ev['mass_kg']))
        if name!='baseline':
            audits[name]['controller']=controller_trace(trace,summary)
    result=dict(status='AUDITED',frozen_sha256=frozen,updates=128,num_envs=512,
        selected_update=ev['selected']['checkpoint']['update'],controller_improved=ev['controller_improved'],
        ppo_improved=ev['ppo_improved'],latest_policy_qualified=ev['latest_policy_qualified'],
        models={name:ev[name] for name in ('baseline','seed','selected','latest')},trace_audits=audits,
        voltage_v=24,assist_strength=.625,zero_assistance_qualified=False,hardware_qualified=False)
    write(args.evaluation/'audit_result.json',result)
    print({k:result[k] for k in ('status','selected_update','controller_improved','ppo_improved','latest_policy_qualified')})


if __name__=='__main__':
    main()

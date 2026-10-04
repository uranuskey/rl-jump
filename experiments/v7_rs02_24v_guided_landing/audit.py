"""Audit actual exit and 400 Hz traces; preserve failures of the latest policy."""
import argparse
from pathlib import Path
import numpy as np
import guided_paths
from guided_runtime import read, sha, verify, write
from legacy_audit import audit_trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--training', type=Path, required=True)
    p.add_argument('--evaluation', type=Path, required=True)
    args = p.parse_args()
    frozen = verify()
    train, evaluation = read(args.training/'result.json'), read(args.evaluation/'result.json')
    assert train['status']=='BUDGET_COMPLETED' and train['completed_updates']==128
    assert evaluation['status']=='EVALUATED'
    assert train['frozen_sha256']==train['final_frozen_sha256']==evaluation['final_frozen_sha256']==frozen
    for folder, stage in ((args.training,'train'), (args.evaluation,'evaluate')):
        receipt = read(folder/'exit_receipt.json')
        assert receipt['process_exited'] and receipt['exit_code']==0
        assert receipt['stage']==stage and receipt['frozen_sha256']==frozen
    updates = [read(args.training/f'update_{i:04d}.json') for i in range(1,129)]
    assert all(row['update']==i+1 and row['num_envs']==512 and row['trial']['parameter_plan_unchanged']
               and row['trial']['launch_plan_unchanged'] and row['trial']['pre_apex_actions_zero']
               and row['trial']['parameter_samples_per_episode']==1 for i,row in enumerate(updates))
    assert sum(row['actor_steps'] for row in updates)==train['actor_steps']>0
    assert sum(row['eligible_samples'] for row in updates)==train['executed_landing_plans']>0
    assert train['prefix_proof']['status']=='PASS' and train['frozen_launch_unchanged']
    for entry in train['evaluations']:
        assert sha(entry['checkpoint']['path'])==entry['checkpoint']['sha256']
    reports = {}
    for name in ('baseline','guided','selected','latest'):
        entry = evaluation[name]
        assert sha(entry['checkpoint']['path'])==entry['checkpoint']['sha256']
        trace = args.evaluation/f'{name}_traces.npz'
        reports[name] = audit_trace(trace, read(args.evaluation/f'{name}.json'), evaluation['mass_kg'])
        with np.load(trace) as arrays:
            assert arrays['landing_action'].shape[-1]==13
            assert np.all(arrays['landing_action'][~arrays['landing_control_enabled']]==0)
            # Independently recompute the rewarded stroke only until the first arrest.
            measured=read(args.evaluation/f'{name}.json')['cases']
            for i,row in enumerate(measured):
                valid=arrays['active'][:,i]
                phase=arrays['phase'][valid,i]
                f=arrays['sensor_wheel_force_n'][valid,i]
                vz=arrays['sensor_com_vz_mps'][valid,i]
                z=arrays['sensor_com_z_m'][valid,i]
                touch=np.flatnonzero((np.r_[0,phase[:-1]]==2)&(f.max(-1)>=1))[0]
                stopped=np.flatnonzero(vz[touch:touch+101]>=-.05)
                last=touch+stopped[0] if len(stopped) else min(touch+100,len(z)-1)
                stroke=max(0.,float(z[touch-1]-z[touch:last+1].min()))
                assert abs(stroke-row['landing_metrics']['first_stop_com_drop_m'])<1e-5
                assert bool(len(stopped))==bool(row['landing_metrics']['first_stop_observed'])
    chosen, baseline = evaluation['selected']['evaluation'], evaluation['baseline']['evaluation']
    result = dict(status='AUDITED', frozen_sha256=frozen, num_envs=512, updates=128,
        actor_steps=train['actor_steps'], executed_landing_plans=train['executed_landing_plans'],
        training_result_sha256=sha(args.training/'result.json'), evaluation_result_sha256=sha(args.evaluation/'result.json'),
        selected_update=train['selected']['checkpoint']['update'], selected_checkpoint=train['selected']['checkpoint'],
        trained_policy_promoted=evaluation['trained_policy_promoted'], latest_policy_qualified=evaluation['latest_policy_qualified'],
        guided=evaluation['guided']['evaluation'], ppo_improved_over_guided_seed=evaluation['ppo_improved_over_guided_seed'],
        parent_checkpoint=train['parent_checkpoint'], guidance=train['guide_search'],
        baseline=baseline, selected=chosen, latest=evaluation['latest']['evaluation'], trace_audits=reports,
        force_peak_mean_change_percent=100*(chosen['mean_landing_metrics']['peak_force_n']/baseline['mean_landing_metrics']['peak_force_n']-1),
        zero_assistance_qualified=False, hardware_qualified=False, voltage_v=24, assist_strength=.625)
    write(args.evaluation/'audit_result.json', result)
    print({k:result[k] for k in ('status','selected_update','trained_policy_promoted','latest_policy_qualified')})


if __name__=='__main__':
    main()

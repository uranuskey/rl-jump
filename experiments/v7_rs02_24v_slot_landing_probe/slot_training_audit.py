"""Verify exits, update accounting, fixed launch, forces, actual stroke and FIFO."""
import argparse
from pathlib import Path
from slot_training_runtime import *


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--training', type=Path, required=True)
    p.add_argument('--evaluation', type=Path, required=True)
    args = p.parse_args()
    frozen = verify_training()
    train = checked_result(args.training/'result.json','BUDGET_COMPLETED','train')
    evaluation = checked_result(args.evaluation/'result.json','EVALUATED','evaluate')
    for result in (train,evaluation):
        receipt_identity(result,frozen)
    assert train['completed_updates'] == 128 and train['num_envs'] == 512
    assert evaluation['training_result_sha256'] == sha(args.training/'result.json')
    rows = [read(args.training/f'update_{i:04d}.json') for i in range(1,129)]
    assert all(r['update']==i+1 and r['num_envs']==512 and r['sample_kind']=='stochastic_training'
        and r['trial']['launch_plan_unchanged'] and r['trial']['parameter_plan_unchanged']
        for i,r in enumerate(rows))
    assert sum(r['actor_steps'] for r in rows) == train['actor_steps'] > 0
    assert sum(r['eligible_samples'] for r in rows) == train['executed_landing_plans'] > 0
    assert train['final_checkpoint']['update'] == 128
    assert sha(train['latest_checkpoint']['path']) == train['latest_checkpoint']['sha256']
    from audit_training import audit_trace
    from slot_probe import schedule_audit, force_windows, load_controller_audit
    from slot_audit import slot_audit
    from analyze_saved import contact_metrics
    audits = {}
    for name, model in evaluation['models'].items():
        checkpoint = model['checkpoint']
        assert sha(checkpoint['path']) == checkpoint['sha256']
        summary = read(args.evaluation/f'{name}.json')
        trace = args.evaluation/f'{name}_traces.npz'
        assert metrics(summary) == model['metrics']
        audits[name] = dict(
            physics=audit_trace(trace,summary,evaluation['mass_kg']),
            controller=load_controller_audit()(trace,summary),
            schedule=schedule_audit(trace,model['profile'],True),
            slot=slot_audit(trace,model['profile']),
            force_windows=force_windows(trace), force_timing=contact_metrics(trace))
        for key in ('physics','controller','schedule','slot'):
            assert audits[name][key]['status']=='PASS'
    result = dict(status='AUDITED', frozen_sha256=SOURCE_FROZEN, training_frozen_sha256=frozen,
        completed_updates=128, num_envs=512, models=evaluation['models'], trace_audits=audits,
        selected_update=evaluation['models']['selected']['checkpoint']['update'],
        controller_improved=evaluation['controller_improved'], ppo_improved=evaluation['ppo_improved'],
        latest_policy_qualified=evaluation['latest_policy_qualified'],
        voltage_v=24, assist_strength=.625, zero_assistance_qualified=False, hardware_qualified=False,
        utc_end=now(), final_frozen_sha256=verify(), final_training_frozen_sha256=verify_training())
    receipt_identity(result,frozen)
    write(args.evaluation/'audit_result.json',result)
    print({k:result[k] for k in ('status','selected_update','controller_improved','ppo_improved','latest_policy_qualified')})


if __name__ == '__main__':
    main()

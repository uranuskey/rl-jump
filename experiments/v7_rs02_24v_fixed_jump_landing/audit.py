"""CPU audit of completed training, independent traces, and real process receipts."""
import argparse
import json
from pathlib import Path
import bootstrap
from runtime import sha, verify, write
from legacy_audit import audit_trace


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


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
    for folder, stage in ((args.training, 'train'), (args.evaluation, 'evaluate')):
        receipt = read(folder/'exit_receipt.json')
        assert receipt['process_exited'] and receipt['exit_code']==0
        assert receipt['stage']==stage and receipt['run_id']==folder.name and receipt['frozen_sha256']==frozen
    updates = [read(args.training/f'update_{i:04d}.json') for i in range(1, 129)]
    assert all(x['update']==i+1 and x['trial']['assist_strength']==.625 and
               x['trial']['launch_plan_unchanged'] and x['trial']['pre_apex_actions_zero']
               and x['trial']['dense_reward_identity_max_error']<=.001
               for i, x in enumerate(updates))
    assert sum(x['actor_steps'] for x in updates)==train['actor_steps']>0
    assert sum(x['eligible_samples'] for x in updates)==train['landing_transitions']>0
    assert train['prefix_proof']['status']=='PASS' and train['frozen_launch_unchanged']
    for entry in train['evaluations']:
        assert sha(Path(entry['checkpoint']['path']))==entry['checkpoint']['sha256']
    traces = {}
    for name in ('baseline', 'selected'):
        if name=='selected' and evaluation['selected_is_initial']:
            traces[name] = traces['baseline']
        else:
            traces[name] = audit_trace(args.evaluation/f'{name}_traces.npz',
                read(args.evaluation/f'{name}.json'), evaluation['mass_kg'])
    original, chosen = evaluation['baseline']['evaluation'], evaluation['selected']['evaluation']
    result = dict(status='AUDITED', frozen_sha256=frozen, updates=128, num_envs=train['num_envs'],
        training_result_sha256=sha(args.training/'result.json'), evaluation_result_sha256=sha(args.evaluation/'result.json'),
        actor_steps=train['actor_steps'], landing_transitions=train['landing_transitions'],
        selected_update=train['selected']['checkpoint']['update'], selected_checkpoint=train['selected']['checkpoint'],
        trained_policy_promoted=evaluation['trained_policy_promoted'], baseline=original, selected=chosen,
        force_peak_mean_change_percent=100*(chosen['mean_landing_metrics']['peak_force_n']/original['mean_landing_metrics']['peak_force_n']-1),
        trace_audits=traces, prefix_proof=train['prefix_proof'], assist_strength=.625, voltage_v=24,
        zero_assistance_qualified=False, hardware_qualified=False)
    write(args.evaluation/'audit_result.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('baseline','selected','trace_audits')}))


if __name__=='__main__':
    main()

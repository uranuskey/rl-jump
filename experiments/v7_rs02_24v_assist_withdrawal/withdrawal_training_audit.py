"""Check actual exits, source identity, accepted optimization and saved traces."""
import argparse
from pathlib import Path
from withdrawal_runtime import verify, read, write, sha, now, metrics
from withdrawal_training_runtime import verify_training, checked_result
from withdrawal_contract import admission


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--training', type=Path, required=True)
    p.add_argument('--evaluation', type=Path, required=True)
    args = p.parse_args()
    frozen, training_frozen = verify(), verify_training()
    train = checked_result(args.training/'result.json', 'BUDGET_COMPLETED', 'train')
    evaluated = checked_result(args.evaluation/'result.json', 'EVALUATED', 'evaluate')
    for r in (train, evaluated):
        assert r['training_frozen_sha256'] == r['final_training_frozen_sha256'] == training_frozen
    assert evaluated['training_result_sha256'] == sha(args.training/'result.json')
    assert train['completed_updates'] == 128 and train['num_envs'] == 512
    assert train['frozen_launch_unchanged'] and train['no_further_assistance_reduction']
    level = train['contract']['strength']
    updates = [read(args.training/f'update_{i:04d}.json') for i in range(1, 129)]
    assert len(updates) == 128 and all(r['update'] == i+1 for i, r in enumerate(updates))
    assert sum(r['actor_steps'] for r in updates) == train['actor_steps'] > 0
    assert sum(r['eligible_samples'] for r in updates) == train['executed_landing_plans']
    for row in updates:
        assert row['num_envs'] == 512 and row['sample_kind'] == 'stochastic_training'
        assert abs(row['trial']['assist_strength']-level) < 1e-7
        assert row['trial']['parameter_plan_unchanged'] and row['trial']['launch_plan_unchanged']
        assert row['trial']['pre_apex_actions_zero']
    for entry in train['evaluations']:
        ck = entry['checkpoint']
        assert sha(ck['path']) == ck['sha256']
    from withdrawal_audit import audit
    models = evaluated['models']
    anchor = models['reference625']['metrics']
    for name, row in models.items():
        assert sha(row['checkpoint']['path']) == row['checkpoint']['sha256']
        trace, summary_path = args.evaluation/(name+'_traces.npz'), args.evaluation/(name+'.json')
        assert sha(trace) == row['trace_sha256'] and sha(summary_path) == row['summary_sha256']
        summary = read(summary_path)
        assert metrics(summary) == row['metrics']
        checks = audit(trace, summary, evaluated['mass_kg'], train['contract']['profile'], row['strength'])
        assert checks == row['audits']
        row['admission'] = admission(row['metrics'], anchor)
    selected = models['selected']
    qualified = selected['admission']['passed'] and selected['audits']['status'] == 'PASS'
    ppo_delta = models['seed']['metrics']['mean_force_n']-selected['metrics']['mean_force_n']
    result = dict(status='AUDITED', utc_end=now(), completed_updates=128, num_envs=512,
        actor_steps=train['actor_steps'], executed_landing_plans=train['executed_landing_plans'],
        frozen_sha256=frozen, training_frozen_sha256=training_frozen,
        final_frozen_sha256=verify(), final_training_frozen_sha256=verify_training(),
        models=models, selected_update=selected['checkpoint']['update'], assist_strength=level,
        selected_lower_assistance_qualified=qualified,
        latest_policy_qualified=models['latest']['admission']['passed'] and models['latest']['audits']['status']=='PASS',
        ppo_mean_force_improvement_n=ppo_delta,
        ppo_improved=qualified and selected['checkpoint']['update'] > 0 and ppo_delta > 1.,
        voltage_v=24, zero_assistance_qualified=False, hardware_qualified=False,
        no_automatic_next_stage=True)
    write(args.evaluation/'audit_result.json', result)
    print({k:v for k,v in result.items() if k != 'models'}, flush=True)


if __name__ == '__main__':
    main()

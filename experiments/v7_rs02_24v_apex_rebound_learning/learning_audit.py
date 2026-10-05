"""Check actual exits, source identity, accepted optimization and saved traces."""
import argparse
from pathlib import Path
from learning_runtime import verify, read, write, sha, now, metrics
from learning_runtime import verify_training, checked_result, learning_evidence
from learning_contract import (qualification as admission, learning_admission, evaluation_entries,
                               replay_qualified, selected_better)


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
    prerequisite = train['prerequisites']['preflight']
    assert sha(prerequisite['path']) == prerequisite['sha256']
    prior = checked_result(prerequisite['path'], 'LEARNING_ENTRY_VERIFIED', 'preflight')
    contract, evidence = learning_evidence(train['contract']['probe_result']['path'])
    assert contract == train['contract'] == prior['contract']
    assert evidence == train['prerequisites']['reused_evidence'] == prior['evidence']
    initial = metrics(read(args.training/'baseline.json'))
    assert initial == train['initial_metrics']
    assert learning_admission(initial, train['anchor_metrics']) == train['initial_learning_admission']
    assert train['initial_learning_admission']['passed']
    level = train['contract']['strength']
    updates = [read(args.training/f'update_{i:04d}.json') for i in range(1, 129)]
    assert len(updates) == 128 and all(r['update'] == i+1 for i, r in enumerate(updates))
    assert sum(r['actor_steps'] for r in updates) == train['actor_steps'] > 0
    assert sum(r['eligible_samples'] for r in updates) == train['executed_landing_plans']
    for row in updates:
        assert row['num_envs'] == 512 and row['sample_kind'] == 'stochastic_training'
        assert row['trial']['takeoff_assist_strength'] == .625
        assert abs(row['trial']['after_apex_assist_strength']-level) < 1e-7
        assert row['trial']['assist_schedule'] == 'observed_COM_apex_then_100ms_ramp'
        assert row['trial']['parameter_plan_unchanged'] and row['trial']['launch_plan_unchanged']
        assert row['trial']['pre_apex_actions_zero']
    reconstructed_selection = None
    for entry in train['evaluations']:
        ck = entry['checkpoint']
        assert sha(ck['path']) == ck['sha256']
        assert entry['admission'] == admission(entry['metrics'], train['anchor_metrics'])
        incumbent = None if reconstructed_selection is None else reconstructed_selection['metrics']
        if selected_better(entry['metrics'], incumbent, train['anchor_metrics']):
            reconstructed_selection = entry
    assert reconstructed_selection == train['selected']
    from apex_audit import audit
    models = evaluated['models']
    expected_entries = evaluation_entries(train)
    assert set(models) == {x[0] for x in expected_entries}
    for name, checkpoint, level_expected in expected_entries:
        assert models[name]['checkpoint'] == checkpoint and models[name]['strength'] == level_expected
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
    batch = evaluated['batch_replays']
    assert set(batch) == set(models)
    assert evaluated['batch_anchor_metrics'] == batch['reference625'][0]['metrics']
    for name, replays in batch.items():
        expected_count = 3 if name in ('selected', 'latest') else 1
        assert len(replays) == expected_count
        assert [r['repeat'] for r in replays] == list(range(1, expected_count+1))
        for replay in replays:
            path = args.evaluation/replay['summary']
            assert sha(path) == replay['summary_sha256']
            measured = metrics(read(path))
            assert measured == replay['metrics'] and measured['worlds'] == 512
            check_anchor = train['anchor_metrics'] if name == 'reference625' else evaluated['batch_anchor_metrics']
            assert replay['admission'] == admission(measured, check_anchor)
            assert replay['prefix_proof_samples'] > 0
    reference_ok = replay_qualified(batch['reference625'], 1)
    def qualified_model(name):
        row = models.get(name)
        return bool(row and reference_ok and row['admission']['passed'] and row['audits']['status'] == 'PASS'
                    and replay_qualified(batch[name]))
    selected = models.get('selected')
    qualified = qualified_model('selected')
    ppo_delta = None if selected is None else models['seed']['metrics']['mean_force_n']-selected['metrics']['mean_force_n']
    rebound_delta = None if selected is None else (batch['seed'][0]['metrics']['max_rebound_mps']
        - max(r['metrics']['max_rebound_mps'] for r in batch['selected']))
    result = dict(status='AUDITED', utc_end=now(), completed_updates=128, num_envs=512,
        actor_steps=train['actor_steps'], executed_landing_plans=train['executed_landing_plans'],
        frozen_sha256=frozen, training_frozen_sha256=training_frozen,
        final_frozen_sha256=verify(), final_training_frozen_sha256=verify_training(),
        models=models, batch_replays=batch, reference_batch_qualified=reference_ok,
        selected_update=None if selected is None else selected['checkpoint']['update'],
        takeoff_assist_strength=.625,
        after_apex_assist_strength=level, assist_schedule='observed_COM_apex_then_100ms_ramp',
        selected_lower_assistance_qualified=qualified,
        latest_policy_qualified=qualified_model('latest'),
        ppo_mean_force_improvement_n=ppo_delta,
        ppo_max_rebound_improvement_batch_mps=rebound_delta,
        ppo_force_improved=qualified and ppo_delta > 1.,
        ppo_rebound_improved=qualified and rebound_delta > 1e-6,
        voltage_v=24, zero_assistance_qualified=False, hardware_qualified=False,
        no_automatic_next_stage=True)
    write(args.evaluation/'audit_result.json', result)
    print({k:v for k,v in result.items() if k not in ('models','batch_replays')}, flush=True)


if __name__ == '__main__':
    main()

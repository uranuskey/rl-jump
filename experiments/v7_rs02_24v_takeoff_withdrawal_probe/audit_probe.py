"""Independent saved-evidence audit; completion does not imply a qualified reduction."""
import argparse
from pathlib import Path
from probe_runtime import verify, handoff_contract, read, write, sha, metrics, now
from probe_contract import LEVELS, admission, qualified, deepest_qualified
from probe_audit import audit


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True, type=Path)
    args = p.parse_args()
    result = read(args.run/'result.json')
    receipt = read(args.run/'exit_receipt.json')
    assert receipt['process_exited'] and receipt['exit_code'] == 0
    assert result['status'] == 'PROBE_COMPLETED' and result['training_updates'] == 0
    assert result['frozen_sha256'] == result['final_frozen_sha256'] == verify()
    contract = handoff_contract(result['contract']['handoff']['path'])
    assert contract == result['contract']
    expected = [('reference625', .625, .625)] + list(LEVELS)
    models = result['models']
    assert list(models) == [x[0] for x in expected[:len(models)]]
    previous_ok = True
    for name, before, after in expected[:len(models)]:
        assert previous_ok, 'Reduction proceeded after failed qualification'
        row = models[name]
        assert (row['before'], row['after']) == (before, after)
        expected_ck = contract['original_reference']['source_checkpoint'] if name == 'reference625' else contract['source_checkpoint']
        assert row['checkpoint'] == expected_ck
        assert sha(expected_ck['path']) == expected_ck['sha256']
        native = row['native']
        for key in ('summary', 'trace'):
            assert sha(args.run/native[key]) == native[key+'_sha256']
        summary = read(args.run/native['summary'])
        assert metrics(summary) == native['metrics']
        assert audit(args.run/native['trace'], summary, result['mass_kg'], contract['profile'], before, after) == native['audits']
        anchor = contract['historical_anchor'] if name == 'reference625' else models['reference625']['native']['metrics']
        assert admission(native['metrics'], anchor) == native['admission']
        for replay in row['batch_replays']:
            path = args.run/replay['summary']
            assert sha(path) == replay['summary_sha256']
            assert metrics(read(path)) == replay['metrics']
            anchor = contract['historical_anchor'] if name == 'reference625' else models['reference625']['batch_replays'][0]['metrics']
            assert admission(replay['metrics'], anchor) == replay['admission']
            assert replay['prefix_proof_samples'] > 0
        previous_ok = qualified(row, 1 if name == 'reference625' else 3)
        assert row['qualified'] == previous_ok
    chosen = deepest_qualified(models)
    assert chosen == result['deepest_qualified']
    audited = dict(status='AUDITED', utc=now(), frozen_sha256=verify(),
        probe_result_sha256=sha(args.run/'result.json'), parent_completed_updates=contract['parent_completed_updates'],
        source_checkpoint=contract['source_checkpoint'], models=models, deepest_qualified=chosen,
        further_withdrawal_qualified=chosen not in (None, 'current625_600'),
        training_updates=0, zero_assistance_qualified=False, hardware_qualified=False,
        no_force_improvement_required=True, voltage_v=24, estimated_motor_curve=True,
        qualification_is_not_hardware_validation=True)
    write(args.run/'audit_result.json', audited)
    print({k: v for k, v in audited.items() if k != 'models'}, flush=True)


if __name__ == '__main__':
    main()

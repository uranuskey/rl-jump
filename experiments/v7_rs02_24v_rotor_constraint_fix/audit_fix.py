"""Reconstruct diagnosis and qualification from retained raw evidence."""
import argparse
from pathlib import Path
import numpy as np
from fix_runtime import verify, contract, read, write, sha, metrics, now
from fix_contract import VARIANTS, LEVELS, admission, qualified, prefix_qualified, deepest_qualified
from fix_audit import audit


def check_receipt(folder, file='exit_receipt.json'):
    r = read(folder/file)
    assert r['process_exited'] and r['exit_code'] == 0


def check_physics(receipt, variant):
    expected = VARIANTS[variant]
    assert receipt['variant'] == variant and receipt['applied'] == expected
    assert receipt['structure_unchanged'] and receipt['refsafe_enabled']
    assert receipt['timestep_s'] == .0025
    assert receipt['q_gate_rad'] == .001 and receipt['v_gate_rad_s'] == .01
    assert not receipt['state_projection'] and not receipt['motor_curve_changed'] and not receipt['contact_parameters_changed']
    assert np.allclose(receipt['cpu_eq_solref'], [[expected['rotor_timeconst'], 1.]]*4, atol=0, rtol=0)
    assert np.allclose(receipt['gpu_eq_solref'], receipt['cpu_eq_solref'], atol=1e-9, rtol=1e-7)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--diagnosis', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    frozen, source = verify(), contract()
    d, result = read(args.diagnosis/'result.json'), read(args.run/'result.json')
    check_receipt(args.diagnosis); check_receipt(args.run)
    assert d['status'] == 'DIAGNOSED' and result['status'] == 'PROBE_COMPLETED'
    for r in (d, result):
        assert r['contract'] == source and r['training_updates'] == 0
        assert r['frozen_sha256'] == r['final_frozen_sha256'] == frozen
    assert sha(result['diagnosis']['path']) == result['diagnosis']['sha256']
    for name, variant in d['variants'].items():
        check_physics(variant['physics_receipt'], name)
        assert len(variant['repeats']) == 3
        for saved in variant['repeats']:
            row = read(args.diagnosis/f"{name}_{saved['repeat']}.json")
            assert {k:v for k,v in row.items() if k!='cases'} == saved
            assert row['worlds'] == 512 and len(row['cases']) == 512
            assert sum(c['failed'] for c in row['cases']) == row['failed']
            assert max(c['max_v_rad_s'] for c in row['cases']) == row['max_mimic_v_rad_s']
            assert max(c['max_q_rad'] for c in row['cases']) == row['max_mimic_q_rad']
            path = args.diagnosis/row['trace']
            assert sha(path) == row['trace_sha256']
            with np.load(path) as z:
                live = z['active']
                v64 = np.abs(z['rotor_v'].astype(np.float64)-7.75*z['joint_v'].astype(np.float64)).max(-1)
                assert float(v64[live].max()) == row['float64_recomputed_peak_v']
                assert float(np.abs(z['reported_v']-v64)[live].max()) == row['max_float32_measurement_difference']
                assert float(z['reported_v'][live].max()) <= row['max_mimic_v_rad_s']+1e-9
                assert int(z['solver_niter'][live].max()) == row['max_solver_iterations']
        assert prefix_qualified(variant['repeats']) == variant['prefix_qualified']
    selected = next((name for name in ('solve_strict','rotor10ms','rotor5ms')
        if name in d['variants'] and prefix_qualified(d['variants'][name]['repeats'])), None)
    assert selected == d['selected_variant'] == result['constraint_variant']
    models = result['models']
    expected = [('reference625', .625, .625)] + LEVELS
    assert list(models) == [x[0] for x in expected[:len(models)]]
    previous_ok = True
    for name, before, after in expected[:len(models)]:
        assert previous_ok, 'Continued withdrawal after failed qualification'
        row = models[name]
        assert (row['before'], row['after'], row['variant']) == (before, after, selected)
        check_physics(row['physics_receipt'], selected)
        ck = source['original_reference']['source_checkpoint'] if name=='reference625' else source['source_checkpoint']
        assert row['checkpoint'] == ck and sha(ck['path']) == ck['sha256']
        native = row['native']
        for key in ('summary','trace'):
            assert sha(args.run/native[key]) == native[key+'_sha256']
        summary = read(args.run/native['summary'])
        assert metrics(summary) == native['metrics']
        assert summary['max_mimic_v_rad_s'] == native['max_mimic_v_rad_s']
        assert audit(args.run/native['trace'], summary, result['mass_kg'], source['profile'], before, after, selected) == native['audits']
        anchor = source['historical_anchor'] if name=='reference625' else models['reference625']['native']['metrics']
        assert admission(native['metrics'], anchor) == native['admission']
        assert admission(native['metrics'], source['old_models']['reference625']['native']['metrics']) == native['old_reference_admission']
        old_native = source['old_models'].get(name, {}).get('native', {}).get('metrics')
        assert native['old_physics_comparison'] == (None if old_native is None else admission(native['metrics'], old_native))
        for replay in row['batch_replays']:
            path = args.run/replay['summary']
            assert sha(path) == replay['summary_sha256']
            summary = read(path)
            assert metrics(summary) == replay['metrics']
            assert summary['max_mimic_v_rad_s'] == replay['max_mimic_v_rad_s']
            anchor = source['historical_anchor'] if name=='reference625' else models['reference625']['batch_replays'][0]['metrics']
            assert admission(replay['metrics'], anchor) == replay['admission']
            assert admission(replay['metrics'], source['old_models']['reference625']['batch_replays'][0]['metrics']) == replay['old_reference_admission']
        previous_ok = qualified(row, 1 if name=='reference625' else 3)
        assert row['qualified'] == previous_ok
    chosen = deepest_qualified(models)
    assert chosen == result['deepest_qualified']
    output = dict(status='AUDITED', utc=now(), frozen_sha256=frozen, constraint_variant=selected,
        diagnosis_sha256=sha(args.diagnosis/'result.json'), qualification_sha256=sha(args.run/'result.json'),
        source_checkpoint=source['source_checkpoint'], models=models, diagnosis_variants=d['variants'],
        deepest_qualified=chosen, further_withdrawal_qualified=chosen not in (None,'current625_600'),
        training_updates=0, voltage_v=24, estimated_motor_curve=True,
        zero_assistance_qualified=False, hardware_qualified=False, original_gates_unchanged=True,
        no_force_improvement_required=True, source_asset_unchanged=True,
        physics_revision_explicit=True)
    write(args.run/'audit_result.json', output)
    print({k:v for k,v in output.items() if k not in ('models','diagnosis_variants')}, flush=True)


if __name__ == '__main__':
    main()

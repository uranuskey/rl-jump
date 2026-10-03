"""CPU-only audit of completed training and independent 400 Hz landing traces."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_trace(path, summary, mass):
    with np.load(path) as archive:
        arrays = {k: archive[k] for k in archive.files}
    active = arrays['active']
    force = arrays['sensor_wheel_force_n']
    legs = arrays['sensor_leg_height_m']
    stable = ((force.min(-1) >= 1) & (arrays['sensor_tilt_rad'] <= np.deg2rad(5))
              & (np.linalg.norm(arrays['sensor_body_vxy_mps'], axis=-1) <= .05)
              & (abs(arrays['sensor_body_vz_mps']) <= .05)
              & (arrays['sensor_gyro_norm_rad_s'] <= .5)
              & (legs.min(-1) >= .17) & (legs.max(-1) <= .19))
    assert np.all(arrays['assist_wrench'][..., :3] == 0)
    assert np.all(arrays['assist_wrench'][..., 5] == 0)
    assert np.linalg.norm(arrays['assist_wrench'][active, 3:], axis=-1).max() <= 12.50001
    speeds = abs(arrays['sensor_motor_speed_pre_rad_s'][active])
    cap = np.interp(speeds, np.array([14, 68, 121, 160, 202, 205]) * np.pi / 30,
                    [17, 14, 10, 7, .5, 0], left=17, right=0)
    assert np.max(abs(cap - arrays['sensor_motor_envelope_nm'][active])) < 2e-5
    assert np.all(abs(arrays['sensor_motor_torque_nm'][active]) <= cap + 2e-5)
    reports = []
    for i, row in enumerate(summary['cases']):
        valid = active[:, i]
        phase = arrays['phase'][valid, i]
        f = force[valid, i]
        clear = arrays['sensor_wheel_clearance_m'][valid, i]
        com = arrays['sensor_com_z_m'][valid, i]
        vz = arrays['sensor_com_vz_mps'][valid, i]
        ticks = arrays['ticks'][valid, i]
        assert np.array_equal(ticks, np.arange(1, len(ticks) + 1))
        success = np.flatnonzero(phase == 4)
        free = f.max(-1) <= .5
        flight = (phase == 2) & free
        first_flight = np.flatnonzero(phase == 2)[0]
        release = first_flight
        while release > 0 and free[release - 1]:
            release -= 1
        wheel = float(clear[flight].min(-1).max())
        com_rise = float(com[flight].max() - com[release])
        assert abs(wheel * 100 - row['wheel_cm']) < .0001
        assert abs(com_rise * 100 - row['com_cm']) < .0001
        assert vz[release] >= .15 and np.any(vz[flight] <= 0)
        previous = np.r_[0, phase[:-1]]
        touch = np.flatnonzero((previous == 2) & (f.max(-1) >= 1))[0]
        peak_force = float(f[touch:].sum(-1).max())
        assert abs(peak_force - row['landing_metrics']['peak_force_n']) < .001
        rebound = float(np.maximum(vz[touch:][free[touch:]], 0).max(initial=0))
        assert abs(rebound - row['landing_metrics']['peak_rebound_vz_mps']) < .00001
        if row['passed']:
            assert len(ticks) == 2000 and len(success) > 0
            assert success[0] >= 399 and stable[valid, i][success[0] - 399:].all()
            assert (arrays['reason'][valid, i] == 0).all()
            assert wheel >= .09 and com_rise >= .115
            assert abs(arrays['sensor_motor_speed_rad_s'][valid, i]).max() <= 205 * np.pi / 30
            assert abs(arrays['sensor_wheel_speed_rad_s'][valid, i]).max() <= 20
            assert arrays['sensor_mimic_q_error_rad'][valid, i].max() <= .001
            assert arrays['sensor_mimic_v_error_rad_s'][valid, i].max() <= .01
            assert arrays['sensor_nonwheel_force_n'][valid, i].max() <= 5
            assert not arrays['sensor_self_contact'][valid, i].any()
        reports.append(dict(case=i, passed=row['passed'], ticks=len(ticks), wheel_cm=wheel * 100,
                            com_cm=com_rise * 100, peak_force_n=peak_force,
                            peak_force_bodyweights=peak_force / (mass * 9.81), rebound_mps=rebound))
    return dict(status='PASS', cases=len(reports), passed=sum(x['passed'] for x in reports),
                max_peak_force_n=max(x['peak_force_n'] for x in reports),
                min_wheel_cm=min(x['wheel_cm'] for x in reports),
                min_com_cm=min(x['com_cm'] for x in reports),
                max_rebound_mps=max(x['rebound_mps'] for x in reports), rows=reports,
                trace_sha256=digest(path), checks='400 Hz support, height, stable interval, motor envelope, contacts and assist wrench')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    args = parser.parse_args()
    train = read(args.training / 'result.json')
    evaluation = read(args.evaluation / 'result.json')
    assert train['status'] == 'BUDGET_COMPLETED' and train['completed_updates'] == 128
    assert evaluation['status'] == 'EVALUATED'
    assert train['frozen_sha256'] == train['final_frozen_sha256'] == evaluation['final_frozen_sha256']
    for folder, stage in ((args.training, 'train'), (args.evaluation, 'evaluate')):
        receipt = read(folder / 'exit_receipt.json')
        assert receipt['process_exited'] is True and receipt['exit_code'] == 0
        assert receipt['stage'] == stage and receipt['run_id'] == folder.name
        assert receipt['frozen_sha256'] == train['frozen_sha256']
    updates = [read(args.training / f'update_{i:04d}.json') for i in range(1, 129)]
    assert all(x['update'] == i + 1 and x['trial']['assist_strength'] == .625 for i, x in enumerate(updates))
    assert sum(x['actor_steps'] for x in updates) == train['actor_steps']
    assert sum(x['eligible_samples'] for x in updates) == train['executed_plans'] == 32768
    assert train['frozen_launch_unchanged'] and train['final_training_assist'] == .625
    for entry in train['evaluations']:
        ck = entry['checkpoint']
        assert digest(Path(ck['path'])) == ck['sha256']
    traces = {}
    for name in ('baseline', 'selected'):
        if name == 'selected' and evaluation['selected_is_initial']:
            traces[name] = traces['baseline']
        else:
            traces[name] = audit_trace(args.evaluation / (name + '_traces.npz'),
                                       read(args.evaluation / (name + '.json')), evaluation['mass_kg'])
    original = evaluation['baseline']['evaluation']
    chosen = evaluation['selected']['evaluation']
    result = dict(status='AUDITED', training_result_sha256=digest(args.training / 'result.json'),
                  evaluation_result_sha256=digest(args.evaluation / 'result.json'),
                  audit_source_sha256=digest(Path(__file__)),
                  num_envs=256, updates=128, executed_plans=train['executed_plans'],
                  accepted_actor_steps=train['actor_steps'], assist_strength=.625, voltage_v=24,
                  selected_update=train['selected']['checkpoint']['update'],
                  trained_policy_promoted=evaluation['trained_policy_promoted'],
                  baseline=original, selected=chosen,
                  force_peak_mean_change_percent=100 * (chosen['mean_landing_metrics']['peak_force_n'] /
                                                        original['mean_landing_metrics']['peak_force_n'] - 1),
                  trace_audits=traces, final_checkpoint=train['final_checkpoint'],
                  selected_checkpoint=train['selected']['checkpoint'],
                  zero_assistance_qualified=False, hardware_qualified=False,
                  limitations='Original 45 known simulation conditions; estimated 24 V model; assistance held at62.5%')
    (HERE / 'TRAINING_RESULT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('trace_audits', 'baseline', 'selected')}, ensure_ascii=False))


if __name__ == '__main__':
    main()

"""Read-only, CPU reconstruction of the last 40 ms before first touchdown."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

DT = .0025


def analyze(path):
    with np.load(path) as z:
        keys = ('active', 'phase', 'sensor_wheel_force_n', 'sensor_wheel_clearance_m',
                'sensor_leg_height_m', 'sensor_com_vz_mps', 'requested_payload',
                'arrived_payload', 'sensor_motor_speed_rad_s', 'sensor_motor_torque_nm',
                'sensor_motor_envelope_nm', 'sensor_com_z_m')
        a = {k: z[k] for k in keys}
        rows = []
        for w in range(a['active'].shape[1]):
            x = {k: v[a['active'][:, w], w] for k, v in a.items()}
            hits = np.flatnonzero((np.r_[0, x['phase'][:-1]] == 2)
                                 & (x['sensor_wheel_force_n'].max(1) >= 1))
            if not len(hits):
                rows.append(dict(world=w, touched=False))
                continue
            t = int(hits[0])
            snapshots = []
            for lag in (16, 12, 8, 4, 2, 1):
                k = t-lag
                if k < 2:
                    continue
                clearance = x['sensor_wheel_clearance_m']
                h = x['sensor_leg_height_m']
                snapshots.append(dict(before_touch_ms=lag*DT*1000,
                    min_clearance_m=float(clearance[k].min()),
                    wheel_bottom_vz_mps=((clearance[k]-clearance[k-2])/(2*DT)).tolist(),
                    com_vz_mps=float(x['sensor_com_vz_mps'][k]),
                    mean_leg_height_m=float(h[k].mean()),
                    leg_retraction_v_mps=float(((h[k]-h[k-2])/(2*DT)).mean()),
                    nominal_leg_margin_to105mm_m=float(h[k].min()-.105),
                    requested_height_m=float(.18+.03*x['requested_payload'][k, 6]),
                    arrived_height_m=float(.18+.03*x['arrived_payload'][k, 6]),
                    max_motor_speed_rad_s=float(np.abs(x['sensor_motor_speed_rad_s'][k]).max()),
                    max_motor_torque_nm=float(np.abs(x['sensor_motor_torque_nm'][k]).max()),
                    min_motor_envelope_nm=float(x['sensor_motor_envelope_nm'][k].min())))
            force = x['sensor_wheel_force_n'].sum(1)
            vz = x['sensor_com_vz_mps']
            stop = np.flatnonzero(vz[t:t+101] >= -.05)
            end = t+int(stop[0]) if len(stop) else min(t+100,len(vz)-1)
            rows.append(dict(world=w,delay_ms=((w%45)//5)*DT*1000,touched=True,
                touchdown_s=(t+1)*DT,first_force_n=float(force[t]),
                overall_peak_n=float(force[t:].max()),
                first_stop_com_drop_m=float(x['sensor_com_z_m'][t-1]-x['sensor_com_z_m'][t:end+1].min()),
                snapshots=snapshots))
    valid=[r for r in rows if r['touched']]
    means=[]
    for i in range(6):
        ss=[r['snapshots'][i] for r in valid]
        means.append({k:np.mean([s[k] for s in ss],axis=0).tolist() for k in ss[0]})
    by_delay=[]
    for delay in sorted({r['delay_ms'] for r in valid}):
        rr=[r for r in valid if r['delay_ms']==delay]
        by_delay.append(dict(delay_ms=delay,first_force_n=float(np.mean([r['first_force_n'] for r in rr])),
            wheel_bottom_vz_mps=float(np.mean([r['snapshots'][-1]['wheel_bottom_vz_mps'] for r in rr])),
            leg_height_m=float(np.mean([r['snapshots'][-1]['mean_leg_height_m'] for r in rr])),
            leg_retraction_v_mps=float(np.mean([r['snapshots'][-1]['leg_retraction_v_mps'] for r in rr]))))
    return dict(trace=str(path),trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        samples_per_second=400,velocity_method='causal 5 ms backward difference; mesh-bottom motion',
        margin_note='leg-height minus 105 mm is nominal kinematic margin, not a proven COM stroke',
        means=means,by_delay=by_delay,rows=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert not a.output.exists(),'Preserve previous diagnostics'
    result=analyze(a.trace)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('trace_sha256','means','by_delay')},allow_nan=False))

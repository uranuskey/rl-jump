"""Read-only source diagnostic: trace one failed-search profile, never train it."""
import argparse
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'v7_rs02_24v_compliant_landing'
sys.path.insert(0, str(SOURCE))
import compliant_paths
from compliant_runtime import verify, sha, write, exclusive, resource_limit


def analyze(path):
    import numpy as np
    with np.load(path) as a:
        x = {k:a[k] for k in a.files}
    rows=[]
    for i in range(x['active'].shape[1]):
        valid=x['active'][:,i]
        phase=x['phase'][valid,i]
        f=x['sensor_wheel_force_n'][valid,i].sum(-1)
        touchdown=np.flatnonzero((np.r_[0,phase[:-1]]==2)&(x['sensor_wheel_force_n'][valid,i].max(-1)>=1))[0]
        peak=touchdown+f[touchdown:].argmax()
        h=x['sensor_leg_height_m'][valid,i].mean(-1)
        vz=x['sensor_com_vz_mps'][valid,i]
        r=dict(case=i,force_n=float(f[peak]),peak_ms=float((peak-touchdown)*2.5),
               touch_height_m=float(h[touchdown]),pre_touch_leg_v_mps=float((h[touchdown-1]-h[touchdown-3])/.005),
               pre_touch_com_vz_mps=float(vz[touchdown-1]),height_at_peak_m=float(h[peak]),
               saturated_at_peak=bool(x['sensor_saturated'][valid,i][peak]),
               peak_motor_nm=float(abs(x['sensor_motor_torque_nm'][valid,i][peak]).max()),
               peak_motor_speed_rad_s=float(abs(x['sensor_motor_speed_rad_s'][valid,i][peak]).max()),
               force_after_touch_n=[float(v) for v in f[touchdown:touchdown+16]])
        if 'impedance_kd' in x:
            r.update(kd_pre_touch=float(x['impedance_kd'][valid,i][touchdown-1]),
                     kd_at_peak=float(x['impedance_kd'][valid,i][peak]))
        rows.append(r)
    keys=[k for k,v in rows[0].items() if isinstance(v,(float,bool))]
    return dict(trace_sha256=sha(path),means={k:sum(r[k] for r in rows)/len(rows) for k in keys},rows=rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--parent-trace',type=Path,required=True)
    args=p.parse_args()
    assert __import__('re').fullmatch('[A-Za-z0-9_-]+',args.run_id)
    out=HERE/'runs'/args.run_id
    assert not out.exists()
    frozen=verify()
    with exclusive(45):
        out.mkdir(parents=True)
        started=time.monotonic()
        import torch
        from standing_policy import JumpPolicy
        from compliant_env import CompliantEnv
        from compliant_learning import Policy,FrozenLaunch,initialize
        from compliant_rollout import trial,compact
        torch.set_num_threads(1)
        torch.manual_seed(104051)
        torch.cuda.manual_seed_all(104051)
        env=CompliantEnv(45,fast_backend=False,proof=True,abort_dir=out/'aborts')
        standing=JumpPolicy(env.device).eval().requires_grad_(False)
        policy=Policy(env.device)
        values=(.20,.16,.05,.12,.60,.25,.20,.70)
        initialize(policy,args.checkpoint,values)
        *_,summary=trial(env,standing,FrozenLaunch(env.device),policy,resource_limit(started,1800),trace_path=out/'candidate_traces.npz')
        write(out/'summary.json',summary)
        result=dict(status='DIAGNOSED',source_frozen_sha256=frozen,script_sha256=sha(Path(__file__)),
                    profile=values,evaluation=compact(summary),parent=analyze(args.parent_trace),
                    candidate=analyze(out/'candidate_traces.npz'),wall_s=time.monotonic()-started)
        assert verify()==frozen
        write(out/'result.json',result)
        print(json.dumps(dict(status=result['status'],parent=result['parent']['means'],candidate=result['candidate']['means'])),flush=True)


if __name__=='__main__':
    main()

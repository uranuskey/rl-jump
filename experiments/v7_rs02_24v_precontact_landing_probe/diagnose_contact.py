"""CPU reconstruction of the limiting native baseline contact, never a gate edit."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def diagnose(trace,world):
    m=mujoco.MjModel.from_xml_path(str(ROOT/'experiments/v7_mujoco_training_gate/v7_full_collision.xml'))
    d=mujoco.MjData(m)
    with np.load(trace) as z:
        live=np.flatnonzero(z['active'][:,world])
        bad=np.flatnonzero(z['sensor_self_contact'][:,world] & z['active'][:,world])
        hit=int(bad[0]) if len(bad) else int(live[-1])
        rows=[]
        for t in range(max(0,hit-3),min(hit+1,len(live))):
            d.qpos[:]=z['q'][t,world];d.qvel[:]=z['v'][t,world]
            mujoco.mj_forward(m,d)
            contacts=[]
            for i in range(d.ncon):
                c=d.contact[i];ga,gb=c.geom;ba,bb=m.geom_bodyid[[ga,gb]]
                f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
                if ba and bb:
                    contacts.append(dict(body1=m.body(ba).name,body2=m.body(bb).name,
                        mesh1=m.mesh(m.geom_dataid[ga]).name,mesh2=m.mesh(m.geom_dataid[gb]).name,
                        distance_m=float(c.dist),normal_force_n=float(f[0])))
            rows.append(dict(tick=t+1,time_s=(t+1)*.0025,
                leg_height_m=z['sensor_leg_height_m'][t,world].tolist(),
                reference_m=float(.18+.03*z['arrived_payload'][t,world,6]),
                self_contact=bool(z['sensor_self_contact'][t,world]),
                nonwheel_force_n=float(z['sensor_nonwheel_force_n'][t,world]),
                motor_torque_nm=z['sensor_motor_torque_nm'][t,world].tolist(),
                com_vz_mps=float(z['sensor_com_vz_mps'][t,world]),
                cpu_contacts=contacts))
    return dict(trace=str(trace),world=world,first_self_contact_tick=int(bad[0]+1) if len(bad) else None,
        note='CPU double-precision contact reconstruction; original GPU flags remain authoritative',rows=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--world',type=int,default=0);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    r=diagnose(a.trace,a.world);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(r))

"""CPU geometric counterfactuals. No simulation step, policy edit, or gate change."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
XML=ROOT/'experiments/v7_mujoco_training_gate/v7_full_collision.xml'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ik(h,dx):
    k=np.arccos(np.clip((h*h+dx*dx-.105**2-.145**2)/(2*.105*.145),-1,1))
    a=-np.arctan2(dx,h)-np.arctan2(.145*np.sin(k),.105+.145*np.cos(k))
    return float(a),float(k)


def leg(h,k):
    return dict(height_mm=float(1000*(.105*np.cos(h)+.145*np.cos(h+k))),
        cad_dx_mm=float(-1000*(.105*np.sin(h)+.145*np.sin(h+k))),
        hip_deg=float(np.degrees(h)),knee_deg=float(np.degrees(k)))


def main(a):
    assert not a.output.exists(),'Preserve old results'
    xml_hash=sha(XML)
    cad=json.loads((HERE/'cad_path.json').read_text())
    m=mujoco.MjModel.from_xml_path(str(XML));d=mujoco.MjData(m)
    names=('left_hip_joint','left_knee_joint','right_hip_joint','right_knee_joint')
    qi=[int(m.jnt_qposadr[m.joint(n).id]) for n in names]
    ri=[int(m.jnt_qposadr[m.joint(n.replace('_joint','_rotor_proxy_joint')).id]) for n in names]

    def pose(q):
        d.qpos[:]=q;d.qvel[:]=0;mujoco.mj_forward(m,d)
        contacts=[]
        for c in d.contact:
            ga,gb=int(c.geom1),int(c.geom2)
            ba,bb=m.geom_bodyid[[ga,gb]]
            if ba and bb and c.dist < -1e-5:
                contacts.append(dict(body1=m.body(ba).name,body2=m.body(bb).name,
                    geom1=m.geom(ga).name,geom2=m.geom(gb).name,penetration_mm=float(-1000*c.dist)))
        return dict(left=leg(q[qi[0]],q[qi[1]]),right=leg(q[qi[2]],q[qi[3]]),
            self_penetrations=contacts,count=len(contacts))

    def corrected(q,heights,dxs):
        q=q.copy()
        for side in range(2):
            angles=ik(heights[side],dxs[side])
            for j,angle in enumerate(angles):
                q[qi[side*2+j]]=angle;q[ri[side*2+j]]=7.75*angle
        return q

    nominal=[r for r in cad['samples'] if r['dx_perturbation_mm']==0]
    hs=np.array([r['h_mm'] for r in nominal]);xs=np.array([r['dx_mm'] for r in nominal])
    rows=[]
    with np.load(a.trace) as z:
        failed=[]
        for w in range(z['active'].shape[1]):
            bad=np.flatnonzero(z['active'][:,w] & z['sensor_self_contact'][:,w])
            if len(bad): failed.append((w,int(bad[0])))
        for w,t in failed:
            q=z['q'][t,w].astype(float);actual=pose(q)
            heights=[actual[s]['height_mm']/1000 for s in ('left','right')]
            assert all(.09<=h<=.23 for h in heights),'Outside original sampled CAD path'
            dxs=[float(np.interp(h*1000,hs,xs)/1000) for h in heights]
            fixed=corrected(q,heights,[.052984004467725755]*2)
            aligned=corrected(q,heights,dxs)
            rows.append(dict(world=w,time_s=(t+1)*.0025,gpu_self_contact=True,
                actual=actual,same_height_fixed_x=pose(fixed),same_height_cad_path=pose(aligned),
                cad_dx_mm=[x*1000 for x in dxs],height_change_only_test=False,
                changed_joint_deg=np.degrees(aligned[qi]-q[qi]).tolist()))
    static=[]
    for r in cad['samples']:
        q=m.qpos0.copy();q[:7]=[0,0,r['h_mm']/1000+.10,1,0,0,0]
        q=corrected(q,[r['h_mm']/1000]*2,[r['dx_mm']/1000]*2)
        p=pose(q)
        static.append(dict(h_mm=r['h_mm'],dx_mm=r['dx_mm'],
            perturbation_mm=r['dx_perturbation_mm'],count=p['count'],contacts=p['self_penetrations']))
    scan=[]
    if rows:
        first=rows[0];h=first['actual']['left']['height_mm']/1000
        q=m.qpos0.copy();q[:7]=[0,0,h+.10,1,0,0,0]
        for dx in range(30,66):
            p=pose(corrected(q,[h]*2,[dx/1000]*2))
            scan.append(dict(height_mm=h*1000,dx_mm=dx,count=p['count']))
    result=dict(scope='Static CPU counterfactual only; no proof of dynamic tracking or hardware clearance',
        trace=str(a.trace),trace_sha256=sha(a.trace),xml_sha256=xml_hash,
        script_sha256=sha(Path(__file__)),cad_input_sha256=sha(HERE/'cad_path.json'),
        geometry_unchanged=xml_hash==sha(XML),collision_criterion='nonadjacent robot penetration below -10 micrometres; original filters',
        collider_labels=cad['collider_labels'],failures=rows,static_cad_samples=static,local_x_scan=scan,
        summary=dict(failed_worlds=len(rows),actual_with_penetration=sum(r['actual']['count']>0 for r in rows),
            same_height_fixed_x_clear=sum(r['same_height_fixed_x']['count']==0 for r in rows),
            same_height_cad_clear=sum(r['same_height_cad_path']['count']==0 for r in rows),
            static_cad_clear=sum(r['count']==0 for r in static),static_cad_total=len(static)))
    assert result['geometry_unchanged']
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(result['summary']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())

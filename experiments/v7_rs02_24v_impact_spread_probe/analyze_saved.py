"""CPU-only contact timing and static-step feasibility from saved trajectories."""
import argparse
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DT = .0025


def contact_metrics(trace):
    with np.load(trace) as z:
        rows, curves = [], []
        for w in range(z['active'].shape[1]):
            live = z['active'][:, w]
            phase = z['phase'][live, w]
            force = z['sensor_wheel_force_n'][live, w].sum(1)
            individual = z['sensor_wheel_force_n'][live, w]
            hits = np.flatnonzero((np.r_[0, phase[:-1]] == 2) & (individual.max(1) >= 1))
            if not len(hits):
                rows.append(dict(world=w, touchdown=False))
                continue
            t = int(hits[0])
            peak = t + int(force[t:].argmax())
            h = z['sensor_leg_height_m'][live, w].mean(1)
            rows.append(dict(world=w, touchdown=True, touch_s=(t+1)*DT,
                peak_n=float(force[peak]), peak_after_touch_ms=(peak-t)*DT*1000,
                pre_touch_leg_v_mps=float((h[t-1]-h[t-3])/(2*DT)),
                pre_touch_com_vz_mps=float(z['sensor_com_vz_mps'][live,w][t-1]),
                impulse_20ms_ns=float(force[t:t+8].sum()*DT),
                impulse_80ms_ns=float(force[t:t+32].sum()*DT),
                duration_above_200n_ms=float((force[t:t+80]>200).sum()*DT*1000)))
            curve = np.full(97, np.nan)
            indices = np.arange(t-16, t+81)
            valid = (indices>=0) & (indices<len(force))
            curve[valid] = force[indices[valid]]
            curves.append(curve)
        keys = [k for k,v in rows[0].items() if isinstance(v,float)] if rows[0]['touchdown'] else []
        means = {k:float(np.mean([r[k] for r in rows if r['touchdown']])) for k in keys}
        return dict(means=means, rows=rows, relative_time_ms=(np.arange(-16,81)*2.5).tolist(),
            mean_force_n=np.nanmean(curves,axis=0).tolist() if curves else [],
            max_force_n=np.nanmax(curves,axis=0).tolist() if curves else [])


def step_geometry(trace, summary):
    """Find an edge that avoids all pre-apex wheel geometry and covers touchdown."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(ROOT/'experiments/v7_mujoco_training_gate/v7_full_collision.xml'))
    data = mujoco.MjData(model)
    wheel_bodies = [model.body(s+'_wheel').id for s in ('left','right')]
    meshes = []
    for gid in np.flatnonzero(np.isin(model.geom_bodyid,wheel_bodies)):
        mid = model.geom_dataid[gid]
        adr, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        meshes.append((gid, model.mesh_vert[adr:adr+count].copy()))
    q = np.load(trace)['q']
    rows = []
    for w, case in enumerate(summary['cases']):
        bound = -float('inf')
        for t in range(case['gate_tick']+1):
            data.qpos[:] = q[t,w]
            mujoco.mj_kinematics(model,data)
            for gid, vertices in meshes:
                xyz = vertices @ data.geom_xmat[gid].reshape(3,3).T + data.geom_xpos[gid]
                low = xyz[:,2] <= .013
                if low.any():
                    bound = max(bound, float(xyz[low,0].max()))
        bottom_x = []
        for t in range(case['gate_tick']+1, int(case['touchdown_s']/DT)):
            data.qpos[:] = q[t,w]
            mujoco.mj_kinematics(model,data)
            xyzs = [v @ data.geom_xmat[g].reshape(3,3).T + data.geom_xpos[g] for g,v in meshes]
            if min(x[:,2].min() for x in xyzs) <= .012:
                bottom_x = [float(x[x[:,2]<=x[:,2].min()+.002,0].min()) for x in xyzs]
                break
        rows.append(dict(world=w, pre_apex_max_low_vertex_x=bound,
                         landing_patch_min_x=min(bottom_x) if bottom_x else None))
    lower = max(r['pre_apex_max_low_vertex_x'] for r in rows)+.002
    upper = min(r['landing_patch_min_x'] for r in rows if r['landing_patch_min_x'] is not None)-.002
    return dict(lower_edge_m=lower, upper_edge_m=upper, feasible=lower<upper,
                chosen_edge_m=(lower+upper)/2 if lower<upper else None, rows=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--geometry',action='store_true')
    a=p.parse_args()
    result={n:contact_metrics(a.evaluation/(n+'_traces.npz')) for n in ('baseline','seed','selected','latest')}
    if a.geometry:
        result['step_geometry']=step_geometry(a.evaluation/'selected_traces.npz',json.loads((a.evaluation/'selected.json').read_text()))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:(v['means'] if 'means' in v else {a:b for a,b in v.items() if a!='rows'}) for k,v in result.items()}))

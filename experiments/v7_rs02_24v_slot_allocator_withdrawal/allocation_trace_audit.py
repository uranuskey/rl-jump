"""Independent NumPy allocation reconstruction, physical limits and shared FIFO."""
import json
from pathlib import Path
import numpy as np
from fix_audit import audit_arrays
from fix_contract import checked_variant
def slot_audit(path,profile):
    with np.load(path) as z:
        live=z['active'] & z['landing_control_enabled']
        q=z['sensor_slot_input_q'];v=z['sensor_slot_input_v']
        enabled=z['sensor_slot_enabled'];old=z['sensor_slot_original_action']
        angles=np.cumsum(q[live].reshape(-1,2,2),axis=-1)
        speed=np.cumsum(v[live].reshape(-1,2,2),axis=-1)
        lengths=np.array([.105,.145],dtype=np.float32)
        jx=lengths*np.cos(angles);jh=-lengths*np.sin(angles)
        h=jx.sum(-1);x=(lengths*np.sin(angles)).sum(-1)
        vx=(jx*speed).sum(-1);vh=(jh*speed).sum(-1)
        cad=Path(__file__).resolve().parent.parent/'v7_rs02_24v_geometry_pose_probe/cad_path.json'
        rows=[r for r in json.loads(cad.read_text())['samples'] if r['dx_perturbation_mm']==0]
        hs=np.array([r['h_mm']/1000 for r in rows],dtype=np.float32)
        xs=np.array([-r['dx_mm']/1000 for r in rows],dtype=np.float32)
        hc=np.clip(h,hs[0],hs[-1]);index=np.clip(np.floor((hc-.09)/.002).astype(int),0,len(rows)-2)
        slope=(xs[index+1]-xs[index])/(hs[index+1]-hs[index])
        xt=xs[index]+slope*(hc-hs[index]);vt=slope*vh
        u=np.clip((.20-h)/.04,0,1)
        force=np.clip(profile['slot_kx']*(xt-x)+profile['slot_dx']*(vt-vx),
                      -profile['slot_force_cap_n'],profile['slot_force_cap_n'])*u*u*(3-2*u)
        force=np.where(enabled[live,None],force,0.)
        torque=jx*force[...,None]
        delta=torque/(60*z['sensor_slot_kp'][live,None,None]*.06)
        original=old[live,:4].reshape(-1,2,2);w=profile['slot_action_bound']
        denom=np.where(abs(delta)>1e-10,delta,1.)
        limits=np.where(delta>1e-10,(w-original)/denom,
                 np.where(delta < -1e-10,(-w-original)/denom,1.))
        alpha=np.clip(limits.min(-1),0,1)
        expected=old[live].copy();expected[:,:4]=(original+delta*alpha[...,None]).reshape(-1,4)
        assert np.allclose(expected,z['requested_payload'][...,:6][live],atol=8e-5,rtol=1e-5)
        assert np.allclose(torque*alpha[...,None],z['sensor_slot_motor_increment_nm'][live],atol=2e-4,rtol=1e-5)
        assert np.allclose(force*alpha,z['sensor_slot_allocated_force_n'][live],atol=.002,rtol=1e-5)
        assert np.all(z['sensor_slot_action_bound']==w)
        assert abs(z['requested_payload'][...,:4][live]).max()<=w+2e-6
        assert abs(old[live]).max()<=1.
        assert np.array_equal(z['requested_payload'][...,4:6][live],old[...,4:6][live])
        assert np.all(z['sensor_slot_motor_increment_nm'][~enabled]==0)
        ids=[7,8,10,11];vids=[6,7,9,10]
        assert np.allclose(q[1:][live[1:]],z['q'][:-1,...,ids][live[1:]],atol=1e-7)
        assert np.allclose(v[1:][live[1:]],z['v'][:-1,...,vids][live[1:]],atol=1e-7)
        motor=np.cumsum(z['q'][...,ids].reshape(*live.shape,2,2),axis=-1)
        actual_h=(np.cos(motor)*lengths).sum(-1);actual_x=(np.sin(motor)*lengths).sum(-1)
        tx=np.interp(actual_h,hs,xs)
        compressed=live & (actual_h.min(-1)<.14)
        error=abs(tx-actual_x).max(-1)
        per=[float(error[:,i][compressed[:,i]].max(initial=0)*1000) for i in range(live.shape[1])]
        after=live & (z['sensor_pre_touchdown_s']>0)
        return dict(status='PASS',numpy_feedback_recomputed=True,encoder_causality_verified=True,
            declared_motor_offset_window_rad=.06*w,original_policy_bounds_preserved=True,
            wheel_commands_unchanged=True,external_force_added=False,
            max_allocated_slot_force_n=float(abs(z['sensor_slot_allocated_force_n'][live]).max(initial=0)),
            max_motor_offset_increment_nm=float(abs(z['sensor_slot_motor_increment_nm'][live]).max(initial=0)),
            max_compressed_foreaft_error_mm=max(per),min_actual_leg_height_mm=float(actual_h[after].min(initial=1)*1000),
            per_world_worst_foreaft_error_mm=per)
def audit(path,summary,mass,profile,before,after,variant):
    from audit_training import audit_trace
    from slot_probe import load_controller_audit,schedule_audit
    expected=checked_variant(variant)
    with np.load(path) as z:
        assistance=audit_arrays(z,before,after)
        assert np.allclose(z['sensor_rotor_eq_timeconst_s'],expected['rotor_timeconst'],atol=1e-9,rtol=1e-7)
        assert np.allclose(z['sensor_solver_tolerance'],expected['tolerance'],atol=0,rtol=1e-7)
    assistance.update(runtime_constraint_variant=variant,runtime_constraint_parameters_verified=True)
    slots=slot_audit(path,profile)
    if summary['passed']!=summary['worlds']:
        return dict(status='UNQUALIFIED',assistance=assistance,slot=slots,
            reason='Task failure retained; no qualification')
    assert assistance['observed_apex_worlds']==summary['worlds']
    assert assistance['before_apex']['physical_samples']>0 and assistance['after_ramp']['physical_samples']>0
    return dict(status='PASS',assistance=assistance,physical=audit_trace(path,summary,mass),
        controller=load_controller_audit()(path,summary),schedule=schedule_audit(path,profile,True),slot=slots)

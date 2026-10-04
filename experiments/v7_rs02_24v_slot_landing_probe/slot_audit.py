"""Reconstruct slot feedback from saved encoder samples and bounded payload."""
import numpy as np
import torch
from slot_control import feedback,table,target


def slot_audit(path,profile):
    with np.load(path) as z:
        live=z['active'] & z['landing_control_enabled']
        q=z['sensor_slot_input_q'];v=z['sensor_slot_input_v']
        enabled=z['sensor_slot_enabled'];kp=z['sensor_slot_kp']
        old=z['sensor_slot_original_action']
        c=dict(enabled=torch.from_numpy(enabled[live]),kp=torch.from_numpy(kp[live]),
            correction=torch.atanh(torch.from_numpy(old[live]).clamp(-1+1e-7,1-1e-7)))
        out,d=feedback(c,table([profile],int(live.sum()),'cpu'),torch.from_numpy(q[live]),torch.from_numpy(v[live]))
        assert np.allclose(out.numpy(),z['requested_payload'][...,:6][live],atol=2e-5)
        assert np.allclose(d['slot_motor_increment_nm'].numpy(),z['sensor_slot_motor_increment_nm'][live],atol=2e-5)
        assert np.max(abs(z['requested_payload'][...,:6][live]))<=1.000002
        assert np.array_equal(z['requested_payload'][...,4:6][live],old[...,4:6][live])
        assert np.all(z['sensor_slot_motor_increment_nm'][~enabled]==0)
        # Saved inputs are pre-step encoders, not the post-step trace or future q.
        ids=[7,8,10,11];vids=[6,7,9,10]
        assert np.allclose(q[1:][live[1:]],z['q'][:-1,...,ids][live[1:]],atol=1e-7)
        assert np.allclose(v[1:][live[1:]],z['v'][:-1,...,vids][live[1:]],atol=1e-7)
        actual=z['q'][...,ids].reshape(*z['active'].shape,2,2)
        motor=np.cumsum(actual,axis=-1)
        h=(np.cos(motor)*np.array([.105,.145])).sum(-1)
        x=(np.sin(motor)*np.array([.105,.145])).sum(-1)
        xt=target(torch.from_numpy(h))[0].numpy()
        compressed=live & (h.min(-1)<.14)
        error=abs(xt-x).max(-1)
        worst_by_world=[float(error[:,w][compressed[:,w]].max(initial=0)*1000) for w in range(live.shape[1])]
        after=live & (z['sensor_pre_touchdown_s']>0)
        return dict(status='PASS',feedback_recomputed=True,encoder_causality_verified=True,
            correction_bounds_preserved=True,wheel_commands_unchanged=True,
            max_motor_offset_increment_nm=float(abs(z['sensor_slot_motor_increment_nm'][live]).max(initial=0)),
            max_compressed_foreaft_error_mm=max(worst_by_world),
            mean_world_worst_foreaft_error_mm=float(np.mean(worst_by_world)),
            min_actual_leg_height_mm=float(h[after].min(initial=1)*1000),
            per_world_worst_foreaft_error_mm=worst_by_world)

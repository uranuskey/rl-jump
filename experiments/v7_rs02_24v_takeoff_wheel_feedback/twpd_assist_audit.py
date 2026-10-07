"""Reconstruct both assistance phases from physical traces, not metadata labels."""
import numpy as np
from twpd_gates import checked_levels


def audit_arrays(z, before, after):
    checked_levels(before, after)
    active, ticks = z['active'], z['ticks']
    preticks = ticks-active.astype(np.int64)
    apex = np.full(active.shape[1], -1, dtype=np.int64)
    for w in range(active.shape[1]):
        hits = np.flatnonzero(active[:, w] & (z['phase'][:, w] == 2)
            & (z['sensor_wheel_force_n'][:, w].max(-1) <= .5)
            & (z['sensor_com_vz_mps'][:, w] <= 0))
        if len(hits):
            apex[w] = ticks[hits[0], w]
    post = (apex[None, :] >= 0) & (preticks >= apex[None, :])
    expected_record = np.where(post, apex[None, :], -1)
    assert np.array_equal(z['sensor_assist_apex_tick'][active], expected_record[active]), 'Noncausal apex'
    u = np.clip((preticks-apex[None, :])/40, 0, 1)
    strength = before+(after-before)*np.where(post, u*u*(3-2*u), 0.)
    assert np.allclose(z['assist_strength'][active], strength[active], atol=1e-7, rtol=0)
    entry = np.clip((preticks-200)/40, 0, 1)
    effective = strength*entry*entry*(3-2*entry)
    assert np.allclose(z['assist_effective_strength'][active], effective[active], atol=1e-7, rtol=0)
    up, omega = z['base_rotation_pre'][..., :, 2], z['assist_world_omega_pre']
    desired = np.stack((120*up[..., 1]-8*omega[..., 0],
        -120*up[..., 0]-8*omega[..., 1], np.zeros_like(u)), -1)
    length = np.linalg.norm(desired, axis=-1, keepdims=True)
    expected = desired*np.minimum(1., 20/np.maximum(length, 1e-12))*(effective*active)[..., None]
    wrench = z['assist_wrench']
    assert np.isfinite(wrench).all()
    assert np.all(wrench[..., :3] == 0) and np.all(wrench[..., 5] == 0)
    error = float(np.abs(expected-wrench[..., 3:]).max())
    assert error < 5e-5, ('Assistance wrench mismatch', error)
    torque = np.linalg.norm(wrench[..., 3:], axis=-1)
    pre = active & ~post & (preticks >= 240)
    full = active & post & (u >= 1)
    assert np.max(torque[pre], initial=0) <= 20*before+5e-5
    assert np.max(torque[full], initial=0) <= 20*after+5e-5
    def stats(mask):
        impulse = (torque*mask).sum(0)*.0025
        return dict(peak_torque_nm=float(torque[mask].max(initial=0)),
            mean_angular_impulse_nms=float(impulse.mean()), physical_samples=int(mask.sum()))
    return dict(status='PASS', takeoff_strength=before, after_apex_strength=after,
        takeoff_cap_nm=20*before, after_apex_cap_nm=20*after,
        before_apex=stats(pre), after_ramp=stats(full),
        observed_apex_worlds=int((apex >= 0).sum()), causal_apex_recomputed=True,
        max_formula_error_nm=error, zero_linear_force=True, zero_yaw_torque=True,
        entry_ramp_s=[.5, .6], apex_ramp_s=.1)

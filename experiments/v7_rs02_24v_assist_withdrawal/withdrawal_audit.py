"""Recompute the assistance wrench at every physical step, including the ramp."""
import numpy as np


def assistance_arrays(z, strength):
    active = z['active']
    declared = z['assist_strength']
    assert np.isfinite(declared).all() and np.allclose(declared, strength, atol=1e-7, rtol=0)
    preticks = z['ticks']-active.astype(np.int64)
    u = np.clip((preticks-200)/40, 0, 1)
    effective = strength*u*u*(3-2*u)
    assert np.allclose(effective, z['assist_effective_strength'], atol=1e-7, rtol=0)
    up = z['base_rotation_pre'][..., :, 2]
    omega = z['assist_world_omega_pre']
    desired = np.stack((120*up[..., 1]-8*omega[..., 0],
                        -120*up[..., 0]-8*omega[..., 1], np.zeros_like(u)), -1)
    length = np.linalg.norm(desired, axis=-1, keepdims=True)
    expected = desired*np.minimum(1., 20/np.maximum(length, 1e-12))
    expected *= (effective*active)[..., None]
    wrench = z['assist_wrench']
    assert np.isfinite(wrench).all()
    assert np.all(wrench[..., :3] == 0) and np.all(wrench[..., 5] == 0)
    error = float(np.max(np.abs(expected-wrench[..., 3:])))
    assert error < 5e-5, ('Assistance formula mismatch', error)
    peak = float(np.linalg.norm(wrench[..., 3:], axis=-1).max())
    assert peak <= 20*strength+5e-5
    return dict(status='PASS', strength=strength, peak_torque_nm=peak,
                torque_cap_nm=20*strength, max_formula_error_nm=error,
                physical_samples=int(active.sum()), zero_linear_force=True, zero_yaw_torque=True,
                ramp_start_s=.5, ramp_full_s=.6, uniform_strength_after_ramp=True)


def audit(path, summary, mass, profile, strength):
    from audit_training import audit_trace
    from slot_probe import load_controller_audit, schedule_audit, slot_audit
    with np.load(path) as z:
        assistance = assistance_arrays(z, strength)
    # Some ancestor audits require a completed flight in every world. A failed
    # probe is retained as a qualification failure, never mislabeled as audited.
    if summary['passed'] != summary['worlds']:
        return dict(status='UNQUALIFIED', assistance=assistance,
                    reason='Some worlds failed; full passing-flight audit not applicable')
    return dict(status='PASS', assistance=assistance,
        physical=audit_trace(path, summary, mass), controller=load_controller_audit()(path, summary),
        schedule=schedule_audit(path, profile, True), slot=slot_audit(path, profile))

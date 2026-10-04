"""Variable motor impedance. References and gains share a 15-channel FIFO."""
import compliant_paths
import torch
from launch_contract import (servo as legacy_servo, payload as legacy_payload,
                             force_to_motor, allocate)
from height_contract import to_motor
from voltage_curve import torque_cap


def payload(action, height, velocity, force, kp=None, kd=None, enabled=None):
    one = torch.ones_like(height)
    return torch.cat((legacy_payload(action, height, velocity, force),
        (one if kp is None else kp)[:, None], (one if kd is None else kd)[:, None],
        (torch.zeros_like(height) if enabled is None else enabled.to(height.dtype))[:, None]), 1)


def servo(arrived, q, v, rotation, support, voltage=24):
    assert arrived.shape[1]==15
    out = legacy_servo(arrived[:, :12], q, v, rotation, support, voltage)
    enabled = arrived[:, 14]>.5
    pos_error = out['motor_target_rad']-to_motor(q[:, :4])
    velocity = to_motor(v[:, :4])
    kp, kd = 60*arrived[:, 12:13], 2*arrived[:, 13:14]
    base = kp*pos_error+kd*(arrived[:, 7:11]-velocity)
    delta = force_to_motor(q, rotation, arrived[:, 11])
    paired = support.min(1).values>=1.
    alpha = allocate(base, delta, torque_cap(velocity, voltage), paired)
    applied = delta*alpha[:, None]
    request = base+applied
    pair = request.reshape(-1, 2, 2)
    values = dict(motor_request=request,
        joint_request=torch.stack((pair.sum(-1), pair[..., 1]), -1).reshape(-1, 4),
        motor_base_request_nm=kp*pos_error-kd*velocity,
        motor_velocity_feedforward_nm=kd*arrived[:, 7:11],
        thrust_motor_applied_nm=applied, thrust_allocation_fraction=alpha,
        motor_request_before_thrust_nm=base)
    for key, value in values.items():
        mask = enabled.reshape((-1,)+(1,)*(value.ndim-1))
        out[key] = torch.where(mask, value, out[key])
    out.update(impedance_kp=torch.where(enabled,kp.flatten(),60.),
               impedance_kd=torch.where(enabled,kd.flatten(),2.), compliant_enabled=enabled)
    return out

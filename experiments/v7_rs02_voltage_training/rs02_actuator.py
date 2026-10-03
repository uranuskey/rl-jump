"""Same coupled output-motor mapping, with voltage selected by the matching speed limit."""
import importlib.util
import torch
from bootstrap import ROOT
from voltage_curve import torque_cap, no_load
spec = importlib.util.spec_from_file_location('voltage_prior_actuator', ROOT/'experiments/v7_rl/actuator.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
PhysicalStepFIFO = legacy.PhysicalStepFIFO
POLICY_NAMES = legacy.POLICY_NAMES


def map_and_limit(joint_torque_request_nm, joint_speed_rad_s, wheel_torque_request_nm, wheel_speed_rad_s,
                  *, leg_torque_limit_nm=17., leg_speed_limit_rad_s=no_load(24),
                  wheel_torque_limit_nm=1.5, wheel_speed_limit_rad_s=20., taper_fraction=.1):
    voltage = next((v for v in (24, 48) if abs(leg_speed_limit_rad_s-no_load(v)) < 1e-9), None)
    if voltage is None or leg_torque_limit_nm != 17:
        raise ValueError('Voltage / actuator configuration mismatch')
    req = joint_torque_request_nm.reshape(-1, 2, 2)
    vel = joint_speed_rad_s.reshape(-1, 2, 2)
    mr = torch.stack((req[..., 0]-req[..., 1], req[..., 1]), -1)
    mv = torch.stack((vel[..., 0], vel[..., 0]+vel[..., 1]), -1)
    cap = torque_cap(mv, voltage)
    actual = torch.maximum(torch.minimum(mr, cap), -cap)
    ja = torch.stack((actual[..., 0]+actual[..., 1], actual[..., 1]), -1).reshape(-1, 4)
    wa, ws, we = legacy._speed_taper(wheel_torque_request_nm, wheel_speed_rad_s,
        wheel_torque_limit_nm, wheel_speed_limit_rad_s, taper_fraction)
    return dict(leg_joint_request_nm=joint_torque_request_nm, leg_joint_actual_nm=ja,
        leg_joint_residual_nm=ja-joint_torque_request_nm,
        leg_motor_request_nm=mr.reshape(-1, 4), leg_motor_actual_nm=actual.reshape(-1, 4),
        leg_motor_speed_rad_s=mv.reshape(-1, 4), leg_motor_envelope_nm=cap.reshape(-1, 4),
        leg_motor_torque_saturated=(actual != mr).reshape(-1, 4),
        leg_motor_speed_exceeded=(mv.abs() > no_load(voltage)).reshape(-1, 4),
        wheel_request_nm=wheel_torque_request_nm, wheel_actual_nm=wa,
        wheel_residual_nm=wa-wheel_torque_request_nm, wheel_speed_rad_s=wheel_speed_rad_s,
        wheel_torque_saturated=ws, wheel_speed_exceeded=we)

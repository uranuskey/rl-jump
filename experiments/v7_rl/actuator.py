"""Vectorized V7 output-motor map and physical-step command FIFO.

No simulator import.  All motor limits here are 24 V *research assumptions*,
not measured motor capabilities or a thermal model.  Tensor last dimensions
are policy main-joint order [LH,LK,RH,RK,LW,RW].
"""

from __future__ import annotations

import torch


POLICY_NAMES = (
    "left_hip_joint", "left_knee_joint", "right_hip_joint", "right_knee_joint",
    "left_wheel_joint", "right_wheel_joint",
)
AUX_NAMES = (
    "left_hip_rotor_proxy_joint", "right_hip_rotor_proxy_joint",
    "left_knee_rotor_proxy_joint", "right_knee_rotor_proxy_joint",
)


class PhysicalStepFIFO:
    """Per-env 0..8 physical-step action latency with selective reset.

    A policy action is held during each of its eight 400 Hz physics steps.  The
    FIFO advances once per physics step. Reset clears only selected env rows.
    """

    def __init__(self, num_envs: int, action_dim: int, device: str | torch.device,
                 max_delay_steps: int = 8, delay_steps: int = 0):
        if num_envs < 1 or action_dim < 1 or not 0 <= delay_steps <= max_delay_steps:
            raise ValueError("invalid FIFO dimensions or delay")
        self.max_delay_steps = max_delay_steps
        self.depth = max_delay_steps + 1
        self.history = torch.zeros((num_envs, self.depth, action_dim), device=device)
        self.delay_steps = torch.full((num_envs,), delay_steps, dtype=torch.long, device=device)
        self.rows = torch.arange(num_envs, dtype=torch.long, device=device)
        self.write_index = 0

    def set_delays(self, env_ids: torch.Tensor, delays: torch.Tensor) -> None:
        ids = env_ids.to(device=self.history.device, dtype=torch.long)
        requested = delays.to(device=self.history.device, dtype=torch.long)
        if ids.numel() != requested.numel() or bool(((requested < 0) | (requested > self.max_delay_steps)).any()):
            raise ValueError("one 0..max delay required per selected env")
        self.delay_steps[ids] = requested

    def push(self, request: torch.Tensor) -> torch.Tensor:
        if request.shape != (self.history.shape[0], self.history.shape[2]):
            raise ValueError(f"FIFO request shape mismatch: {tuple(request.shape)}")
        self.history[:, self.write_index, :] = request
        read_index = torch.remainder(self.write_index - self.delay_steps, self.depth)
        arrived = self.history[self.rows, read_index].clone()
        self.write_index = (self.write_index + 1) % self.depth
        return arrived

    def reset(self, env_ids: torch.Tensor) -> None:
        ids = env_ids.to(device=self.history.device, dtype=torch.long)
        self.history[ids] = 0.0


def _speed_taper(torque: torch.Tensor, speed: torch.Tensor, torque_limit: float,
                 speed_limit: float, taper_fraction: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if torque_limit <= 0 or speed_limit <= 0 or not 0 < taper_fraction <= 1:
        raise ValueError("invalid research torque/speed envelope")
    clipped = torque.clamp(-torque_limit, torque_limit)
    width = speed_limit * taper_fraction
    factor = ((speed_limit + width - speed.abs()) / width).clamp(0.0, 1.0)
    propulsive = clipped * speed > 0.0
    actual = torch.where(propulsive, clipped * factor, clipped)
    return actual, clipped != torque, speed.abs() > speed_limit


def map_and_limit(
    joint_torque_request_nm: torch.Tensor,
    joint_speed_rad_s: torch.Tensor,
    wheel_torque_request_nm: torch.Tensor,
    wheel_speed_rad_s: torch.Tensor,
    *, leg_torque_limit_nm: float = 3.0,
    leg_speed_limit_rad_s: float = 6.0,
    wheel_torque_limit_nm: float = 1.5,
    wheel_speed_limit_rad_s: float = 20.0,
    taper_fraction: float = 0.1,
) -> dict[str, torch.Tensor]:
    """τq→A⁻ᵀτq→motor clipping/soft speed taper→Aᵀτm.

    Input leg tensors are [N,4] = [LH,LK,RH,RK]. Wheel tensors are [N,2].
    The knee *motor* speed is q̇h+q̇k, not the relative-knee q̇k.  This function
    never alters joint velocity state and never drives the four auxiliary DOFs.
    """
    if (joint_torque_request_nm.ndim != 2 or joint_torque_request_nm.shape[1] != 4
        or joint_speed_rad_s.shape != joint_torque_request_nm.shape
        or wheel_torque_request_nm.shape != (joint_torque_request_nm.shape[0], 2)
        or wheel_speed_rad_s.shape != wheel_torque_request_nm.shape):
        raise ValueError("expected leg [N,4], speed [N,4], wheel request/speed [N,2]")
    req = joint_torque_request_nm.reshape(-1, 2, 2)
    vel = joint_speed_rad_s.reshape(-1, 2, 2)
    motor_req = torch.stack((req[..., 0] - req[..., 1], req[..., 1]), dim=-1)
    motor_speed = torch.stack((vel[..., 0], vel[..., 0] + vel[..., 1]), dim=-1)
    motor_actual, leg_torque_sat, leg_speed_exceeded = _speed_taper(
        motor_req, motor_speed, leg_torque_limit_nm, leg_speed_limit_rad_s, taper_fraction,
    )
    joint_actual = torch.stack((motor_actual[..., 0] + motor_actual[..., 1],
                                motor_actual[..., 1]), dim=-1).reshape(-1, 4)
    wheel_actual, wheel_torque_sat, wheel_speed_exceeded = _speed_taper(
        wheel_torque_request_nm, wheel_speed_rad_s, wheel_torque_limit_nm,
        wheel_speed_limit_rad_s, taper_fraction,
    )
    return {
        "leg_joint_request_nm": joint_torque_request_nm,
        "leg_joint_actual_nm": joint_actual,
        "leg_joint_residual_nm": joint_actual - joint_torque_request_nm,
        "leg_motor_request_nm": motor_req.reshape(-1, 4),
        "leg_motor_actual_nm": motor_actual.reshape(-1, 4),
        "leg_motor_speed_rad_s": motor_speed.reshape(-1, 4),
        "leg_motor_torque_saturated": leg_torque_sat.reshape(-1, 4),
        "leg_motor_speed_exceeded": leg_speed_exceeded.reshape(-1, 4),
        "wheel_request_nm": wheel_torque_request_nm,
        "wheel_actual_nm": wheel_actual,
        "wheel_residual_nm": wheel_actual - wheel_torque_request_nm,
        "wheel_speed_rad_s": wheel_speed_rad_s,
        "wheel_torque_saturated": wheel_torque_sat,
        "wheel_speed_exceeded": wheel_speed_exceeded,
    }

"""Vectorized Torch implementation of task.JumpTask, with per-environment reset."""
import math
import torch
from settings import JumpConfig

SETTLE, COMPRESS, FLIGHT, RECOVER, SUCCESS, FAILED = range(6)
REASONS = ('none', 'backend_or_overflow', 'mimic_residual', 'illegal_contact', 'actual_motor_torque',
           'motor_speed', 'sustained_saturation', 'tilt', 'horizontal_drift', 'leg_workspace',
           'uncommanded_flight', 'unstable_before_request', 'no_upward_com_takeoff', 'no_crouch',
           'takeoff_timeout', 'height_above_stage_scope', 'flight_timeout', 'flight_too_short',
           'height_target_missed', 'leg_retraction_without_com_jump', 'one_wheel_landing', 'rebound',
           'recovery_timeout', 'lost_balance_after_recovery', 'episode_without_complete_jump')


class TensorTask:
    def __init__(self, n, device, config=None, dtype=torch.float32):
        self.cfg = config or JumpConfig()
        self.n, self.device, self.dtype = n, device, dtype
        self.phase = torch.zeros(n, dtype=torch.long, device=device)
        self.reason = self.phase.clone()
        self.counter_names = ('steps', 'settle_ticks', 'off_ticks', 'recovery_ticks', 'saturation_ticks')
        for name in self.counter_names:
            setattr(self, name, self.phase.clone())
        self.float_names = ('request_time', 'takeoff_time', 'touchdown_time', 'success_time', 'off_start', 'off_com_z',
                            'off_vz', 'takeoff_com_z', 'minimum_leg_height', 'peak_clearance_m', 'com_rise_m', 'phase_start')
        for name in self.float_names:
            setattr(self, name, torch.zeros(n, device=device, dtype=dtype))
        self.paired = torch.zeros(n, dtype=torch.bool, device=device)
        self.start_xy = torch.zeros((n, 2), device=device, dtype=dtype)
        self.initial_leg_height = torch.full((n,), .18, device=device, dtype=dtype)
        self.reset(torch.ones_like(self.paired), self.start_xy, self.initial_leg_height[:, None].expand(-1, 2))

    def reset(self, mask, xy, height):
        self.phase[mask] = SETTLE
        self.reason[mask] = 0
        for name in self.counter_names+self.float_names:
            getattr(self, name)[mask] = 0
        self.minimum_leg_height[mask] = float('inf')
        self.paired[mask] = False
        self.start_xy[mask] = xy[mask]
        self.initial_leg_height[mask] = height[mask].mean(1)

    def fail(self, mask, reason):
        mask = mask & (self.phase != FAILED)
        self.phase[mask] = FAILED
        self.reason[mask] = REASONS.index(reason)

    def update(self, x, active=None):
        c = self.cfg
        if active is None:
            active = torch.ones_like(self.paired)
        before = self.phase.clone()
        old_peak = self.peak_clearance_m.clone()
        self.steps += active.long()
        t = self.steps.to(self.dtype)*c.physics_dt_s
        drift = torch.linalg.vector_norm(x['base_xy_m']-self.start_xy, dim=1)
        min_force, max_force = x['wheel_force_n'].min(1).values, x['wheel_force_n'].max(1).values
        stable = ((min_force >= c.support_force_n) & (x['tilt_rad'] <= math.radians(c.recovery_tilt_deg))
                  & (torch.linalg.vector_norm(x['body_vxy_mps'], dim=1) <= c.recovery_speed_mps)
                  & (abs(x['body_vz_mps']) <= c.recovery_speed_mps) & (x['gyro_norm_rad_s'] <= c.recovery_gyro_rad_s)
                  & (drift <= c.recovery_drift_m) & (x['leg_height_m'].min(1).values >= c.recovery_leg_height_low_m)
                  & (x['leg_height_m'].max(1).values <= c.recovery_leg_height_high_m))
        clear = (max_force <= c.airborne_force_n) & (x['wheel_clearance_m'].min(1).values >= c.detection_clearance_m)
        self.saturation_ticks = torch.where(active, torch.where(x['saturated'], self.saturation_ticks+1, 0), self.saturation_ticks)
        checks = (
            (~x['backend_ok'] | x['overflow'], 'backend_or_overflow'),
            ((x['mimic_q_error_rad'] > c.mimic_q_limit_rad) | (x['mimic_v_error_rad_s'] > c.mimic_v_limit_rad_s), 'mimic_residual'),
            ((x['nonwheel_force_n'] > c.nonwheel_force_limit_n) | x['self_contact'], 'illegal_contact'),
            ((x['motor_torque_nm'].abs().max(1).values > c.leg_motor_torque_limit_nm+1e-6)
             | (x['wheel_torque_nm'].abs().max(1).values > c.wheel_torque_limit_nm+1e-6), 'actual_motor_torque'),
            ((x['motor_speed_rad_s'].abs().max(1).values > c.leg_motor_speed_limit_rad_s)
             | (x['wheel_speed_rad_s'].abs().max(1).values > c.wheel_speed_limit_rad_s), 'motor_speed'),
            (self.saturation_ticks >= c.ticks(c.saturation_stop_s), 'sustained_saturation'),
            (x['tilt_rad'] > math.radians(c.maximum_tilt_deg), 'tilt'),
            (drift > c.maximum_drift_m, 'horizontal_drift'),
            ((x['leg_height_m'].min(1).values < c.leg_height_low_m) | (x['leg_height_m'].max(1).values > c.leg_height_high_m), 'leg_workspace'))
        for mask, reason in checks:
            self.fail(mask & active, reason)
        scope = (self.phase == SETTLE) & active
        self.settle_ticks = torch.where(scope, torch.where(stable, self.settle_ticks+1, 0), self.settle_ticks)
        self.fail(scope & clear, 'uncommanded_flight')
        requested = scope & ~clear & x['requested']
        self.fail(requested & (self.settle_ticks < c.ticks(c.settle_s)), 'unstable_before_request')
        start = requested & (self.phase != FAILED)
        self.phase[start] = COMPRESS
        self.request_time[start] = t[start]
        scope = (self.phase == COMPRESS) & active
        self.minimum_leg_height = torch.where(scope, torch.minimum(self.minimum_leg_height, x['leg_height_m'].mean(1)), self.minimum_leg_height)
        beginning = scope & clear & (self.off_ticks == 0)
        self.off_start[beginning] = t[beginning]
        self.off_com_z[beginning] = x['com_z_m'][beginning]
        self.off_vz[beginning] = x['com_vz_mps'][beginning]
        self.off_ticks = torch.where(scope, torch.where(clear, self.off_ticks+1, 0), self.off_ticks)
        takeoff = scope & (self.off_ticks >= c.ticks(c.takeoff_debounce_s))
        self.fail(takeoff & (self.off_vz < c.minimum_takeoff_com_vz_mps), 'no_upward_com_takeoff')
        self.fail(takeoff & (self.initial_leg_height-self.minimum_leg_height < c.minimum_crouch_m-1e-9), 'no_crouch')
        takeoff &= self.phase != FAILED
        self.phase[takeoff] = FLIGHT
        self.takeoff_time[takeoff] = self.off_start[takeoff]
        self.takeoff_com_z[takeoff] = self.off_com_z[takeoff]
        self.off_ticks[takeoff] = 0
        self.fail(scope & (t-self.request_time > c.takeoff_deadline_s), 'takeoff_timeout')
        scope = (self.phase == FLIGHT) & active
        unsupported = scope & (max_force <= c.airborne_force_n)
        self.peak_clearance_m = torch.where(unsupported, torch.maximum(self.peak_clearance_m, x['wheel_clearance_m'].min(1).values), self.peak_clearance_m)
        self.com_rise_m = torch.where(unsupported, torch.maximum(self.com_rise_m, x['com_z_m']-self.takeoff_com_z), self.com_rise_m)
        flight_s = t-self.takeoff_time
        self.fail(scope & (self.peak_clearance_m > c.maximum_height_m+1e-9), 'height_above_stage_scope')
        self.fail(scope & (flight_s > c.maximum_flight_s), 'flight_timeout')
        touchdown = scope & (self.phase != FAILED) & (max_force >= c.support_force_n)
        self.touchdown_time[touchdown] = t[touchdown]
        self.fail(touchdown & (flight_s < c.minimum_flight_s), 'flight_too_short')
        lo, hi = c.height_bounds
        self.fail(touchdown & ((self.peak_clearance_m < lo-1e-9) | (self.peak_clearance_m > hi+1e-9)), 'height_target_missed')
        self.fail(touchdown & (self.com_rise_m < c.minimum_com_rise_m-1e-9), 'leg_retraction_without_com_jump')
        self.phase[touchdown & (self.phase != FAILED)] = RECOVER
        scope = (self.phase == RECOVER) & active
        self.paired |= scope & (min_force >= c.support_force_n)
        self.fail(scope & ~self.paired & (t-self.touchdown_time > c.touchdown_pair_window_s), 'one_wheel_landing')
        self.off_ticks = torch.where(scope, torch.where(clear, self.off_ticks+1, 0), self.off_ticks)
        self.fail(scope & (self.off_ticks >= c.ticks(c.takeoff_debounce_s)), 'rebound')
        self.recovery_ticks = torch.where(scope, torch.where(stable, self.recovery_ticks+1, 0), self.recovery_ticks)
        recovered = scope & (self.phase != FAILED) & (self.recovery_ticks >= c.ticks(c.recovery_hold_s))
        self.phase[recovered] = SUCCESS
        self.success_time[recovered] = t[recovered]
        self.fail(scope & ~recovered & (t-self.touchdown_time > c.recovery_deadline_s), 'recovery_timeout')
        self.fail((before == SUCCESS) & active & ~stable, 'lost_balance_after_recovery')
        self.fail(active & (self.steps >= c.ticks(c.episode_s)) & (self.phase != SUCCESS), 'episode_without_complete_jump')
        power = (x['motor_torque_nm']*x['motor_speed_rad_s']).abs().sum(1)+(x['wheel_torque_nm']*x['wheel_speed_rad_s']).abs().sum(1)
        cost = .25*(x['tilt_rad']/math.radians(c.maximum_tilt_deg)).square()+.2*(drift/c.maximum_drift_m).square()+.001*power+.1*x['saturated']
        reward = -c.physics_dt_s*cost
        crouch_error = (x['leg_height_m'].mean(1)-.15)/.03
        reward += (self.phase == COMPRESS)*c.physics_dt_s*.2*torch.exp(-crouch_error.square())
        reward += (self.phase == FLIGHT)*2*((self.peak_clearance_m/c.target_height_m).clamp(max=1)-(old_peak/c.target_height_m).clamp(max=1))
        reward += ((self.phase == FLIGHT) & (before == COMPRESS)).to(self.dtype)
        reward += 8*((self.phase == SUCCESS) & (before != SUCCESS))
        reward -= 15*(self.phase == FAILED)
        reward = torch.where(active & (before != FAILED), reward, 0)
        changed = active & (self.phase != before)
        self.phase_start[changed] = t[changed]
        return reward

    def privileged(self, x):
        age = self.steps.to(self.dtype)*self.cfg.physics_dt_s-self.phase_start
        rise = torch.where(self.takeoff_time > 0, x['com_z_m']-self.takeoff_com_z, 0)
        return torch.cat((x['wheel_force_n'], x['wheel_clearance_m'], rise[:, None], x['com_vz_mps'][:, None],
                          x['leg_height_m'], torch.nn.functional.one_hot(self.phase, 6).to(self.dtype),
                          age[:, None], (self.recovery_ticks*self.cfg.physics_dt_s)[:, None]), 1)

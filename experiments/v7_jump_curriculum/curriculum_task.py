"""Learn support and height tracking before enabling the unchanged jump events."""
import math
import torch
import shared
from tensor_task import TensorTask, REASONS as JUMP_REASONS, SETTLE, COMPRESS, FLIGHT, RECOVER, SUCCESS, FAILED
from contract import leg_reference

REASONS = JUMP_REASONS+('ground_support_loss', 'ground_horizon_unstable')
STAGES = ('balance', 'crouch', 'jump')


class CurriculumTask(TensorTask):
    def __init__(self, n, device, config=None, stage='balance', dtype=torch.float32):
        self.stage = stage
        super().__init__(n, device, config, dtype)
        self.ground_good = torch.zeros_like(self.steps)
        self.ground_total = torch.zeros_like(self.steps)

    def reset(self, mask, xy, height):
        super().reset(mask, xy, height)
        if hasattr(self, 'ground_good'):
            self.ground_good[mask] = 0
            self.ground_total[mask] = 0

    def fail(self, mask, reason):
        mask = mask & (self.phase != FAILED)
        self.phase[mask] = FAILED
        self.reason[mask] = REASONS.index(reason)

    def ground_stable(self, x, t):
        c = self.cfg
        drift = torch.linalg.vector_norm(x['base_xy_m']-self.start_xy, dim=1)
        height_error = (x['leg_height_m']-leg_reference(t, self.stage)[:, None]).abs().max(1).values
        return ((x['wheel_force_n'].min(1).values >= c.support_force_n)
            & (x['tilt_rad'] <= math.radians(c.recovery_tilt_deg))
            & (torch.linalg.vector_norm(x['body_vxy_mps'], dim=1) <= c.recovery_speed_mps)
            & (x['body_vz_mps'].abs() <= c.recovery_speed_mps)
            & (x['gyro_norm_rad_s'] <= c.recovery_gyro_rad_s)
            & (drift <= c.recovery_drift_m) & (height_error <= .005))

    def update(self, x, active=None):
        if active is None:
            active = torch.ones_like(self.paired)
        if self.stage == 'jump':
            before = self.phase.clone()
            reward = super().update(x, active)
            # Keep learned support useful before request and after touchdown.
            supported = (self.phase == SETTLE) | (self.phase == RECOVER) | (self.phase == SUCCESS)
            reward += active*(before != FAILED)*supported*self.cfg.physics_dt_s*torch.exp(-(x['tilt_rad']/math.radians(5.)).square())
            return reward
        c = self.cfg
        before = self.phase.clone()
        self.steps += active.long()
        t = self.steps.to(self.dtype)*c.physics_dt_s
        drift = torch.linalg.vector_norm(x['base_xy_m']-self.start_xy, dim=1)
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
        off = x['wheel_force_n'].min(1).values < c.support_force_n
        self.off_ticks = torch.where(active, torch.where(off, self.off_ticks+1, 0), self.off_ticks)
        self.fail(active & (self.off_ticks >= c.ticks(.025)), 'ground_support_loss')
        live = active & (self.phase != FAILED)
        desired = leg_reference(t, self.stage)
        self.phase[live] = torch.where(desired[live] < .179, COMPRESS, SETTLE)
        stable = self.ground_stable(x, t)
        self.recovery_ticks = torch.where(live, torch.where(stable, self.recovery_ticks+1, 0), self.recovery_ticks)
        observed = live & (self.steps > c.ticks(1.))
        self.ground_total += observed.long()
        self.ground_good += (observed & stable).long()
        self.minimum_leg_height = torch.where(live, torch.minimum(self.minimum_leg_height, x['leg_height_m'].mean(1)), self.minimum_leg_height)
        qualified = (self.recovery_ticks >= c.ticks(1.)) & (self.ground_good >= .95*self.ground_total)
        if self.stage == 'crouch':
            qualified &= self.minimum_leg_height <= .165
        end = live & (self.steps >= c.ticks(c.episode_s))
        # Ground tasks use a true time-limit truncation. Qualification remains
        # separate from timeout: an unstable survivor never becomes SUCCESS.
        self.phase[end & qualified] = SUCCESS
        power = (x['motor_torque_nm']*x['motor_speed_rad_s']).abs().sum(1)+(x['wheel_torque_nm']*x['wheel_speed_rad_s']).abs().sum(1)
        height_error = (x['leg_height_m']-desired[:, None]).square().mean(1)
        rate = (1.+1.5*torch.exp(-(x['tilt_rad']/math.radians(5.)).square())
            +.75*torch.exp(-x['body_vxy_mps'].square().sum(1)/.01-x['body_vz_mps'].square()/.01)
            +.5*torch.exp(-x['gyro_norm_rad_s'].square())+1.5*torch.exp(-height_error/.0001)
            -.25*(drift/c.maximum_drift_m).square()-.002*power-.1*x['saturated'])
        reward = c.physics_dt_s*rate-15*(self.phase == FAILED)
        changed = live & (self.phase != before)
        self.phase_start[changed] = t[changed]
        return torch.where(active & (before != FAILED), reward, 0)

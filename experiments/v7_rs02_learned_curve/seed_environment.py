"""Trajectory-guided learning at 24 V and matched voltage evaluation."""
import bootstrap
import math
from dataclasses import dataclass, asdict
import torch
from settings import JumpConfig
from rs02_env import HeightEnv
from jump_task import ProbeTask, SETTLE, COMPRESS, FLIGHT, RECOVER, SUCCESS, FAILED, REASONS
from profiles import height, push_height_velocity
from velocity_contract import reference_motor_velocity
from voltage_curve import no_load


@dataclass(frozen=True)
class ComparisonConfig(JumpConfig):
    episode_s: float = 10.
    target_height_m: float = .03
    takeoff_deadline_s: float = 5.
    maximum_height_m: float = .15
    leg_height_low_m: float = .09
    leg_height_high_m: float = .23
    leg_motor_torque_limit_nm: float = 17.
    leg_motor_speed_limit_rad_s: float = no_load(24)

    def __post_init__(self):
        assert all(math.isfinite(v) and v > 0 for v in asdict(self).values())
        assert self.maximum_height_m == .15 and self.minimum_visible_height_m == .01
        assert self.maximum_tilt_deg == 10 and self.saturation_stop_s == .02
        assert self.target_height_m == .03 and self.episode_s == 10

    @property
    def height_bounds(self):
        # Voltage performance comparison accepts visible 1..15 cm jumps. Tracking
        # the 3 cm learning target is reported separately, never silently relabeled.
        return (.01, .15)


class LearningTask(ProbeTask):
    def update(self, x, active=None):
        if active is None:
            active = torch.ones_like(self.paired)
        before, old_peak = self.phase.clone(), self.peak_clearance_m.clone()
        super().update(x, active)
        live = active & (before != FAILED)
        t = self.steps*self.cfg.physics_dt_s
        index = torch.full_like(self.steps, 2)  # Archived quintic 120 -> 220 mm candidate.
        desired = height(t, index)
        support = (self.phase == SETTLE) | (self.phase == COMPRESS)
        landed = (self.phase == RECOVER) | (self.phase == SUCCESS)
        power = (x['motor_torque_nm']*x['motor_speed_rad_s']).abs().sum(1)
        rate = (2*torch.exp(-(x['tilt_rad']/math.radians(5)).square())
                +.5*torch.exp(-x['gyro_norm_rad_s'].square())
                +support*.5*torch.exp(-(x['leg_height_m'].mean(1)-desired).square()/.0001)
                +landed*torch.exp(-x['body_vxy_mps'].square().sum(1)/.01-x['body_vz_mps'].square()/.01)
                -.001*power-.2*x['saturated'])
        reward = self.cfg.physics_dt_s*rate
        reward += 2*((self.phase == FLIGHT) & (before == COMPRESS))
        reward += 3*((self.peak_clearance_m/.03).clamp(max=1)-(old_peak/.03).clamp(max=1))
        reward += 3*((self.phase == RECOVER) & (before == FLIGHT))
        reward += 12*((self.phase == SUCCESS) & (before != SUCCESS))
        reward -= 15*(self.phase == FAILED)
        return torch.where(live, reward, 0.)


class JumpEnv(HeightEnv):
    def __init__(self, n, voltage=24, **kwargs):
        cfg = ComparisonConfig(leg_motor_speed_limit_rad_s=no_load(voltage))
        super().__init__(n, config=cfg, stage='jump', **kwargs)
        self.task = LearningTask(self.n, self.device, self.cfg)
        self.reset(torch.ones_like(self.paused))

    def reset(self, mask, **kwargs):
        if kwargs.get('poses') is None and hasattr(self, 'training_poses'):
            kwargs.update(poses=self.training_poses, velocities=self.training_velocities)
        return super().reset(mask, **kwargs)

    def command(self):
        index = torch.full_like(self.ticks, 2)
        request = (self.ticks >= 200) & self.request_enabled
        age = ((self.ticks-200).clamp_min(0)*.0025/10).clamp(max=1)
        ref = height(self.ticks*.0025, index)
        return torch.stack((torch.full_like(age, .03/.05), request.float(), age, ref/.25), 1)

    def push_motor_velocity(self, requested_height):
        index = torch.full_like(self.ticks, 2)
        hdot = push_height_velocity(self.ticks*.0025, index)
        return .34*reference_motor_velocity(requested_height, hdot)


def reset_cases(env):
    """Identical 9 delays x 5 physical initial conditions, duplicates explicitly labeled."""
    ids = torch.arange(env.n, device=env.device)
    condition, delay = ids%5, (ids%45)//5
    poses = env.nomq[None].expand(env.n, -1).clone()
    pitch = torch.where(condition == 1, math.pi/180, torch.where(condition == 2, -math.pi/180, 0.))
    c, s = pitch.cos(), pitch.sin()
    px, pz = -.052984004467725755, .06
    dx, dz = poses[:, 0]-px, poses[:, 2]-pz
    poses[:, 0], poses[:, 2] = px+c*dx+s*dz, pz-s*dx+c*dz
    poses[:, 3:7] = 0
    poses[:, 3], poses[:, 5] = (pitch/2).cos(), (pitch/2).sin()
    velocities = torch.zeros_like(env.v)
    vx = torch.where(condition == 3, .02, torch.where(condition == 4, -.02, 0.))
    velocities[:, 0] = vx
    velocities[:, env.main[4:]] = (vx/.06)[:, None]
    mask = torch.ones_like(env.paused)
    env.reset(mask, poses=poses, velocities=velocities, fixed_delay=0)
    poses[:, 2] -= env.clearance.min(1).values
    env.reset(mask, poses=poses, velocities=velocities, fixed_delay=0)
    env.fifo.set_delays(ids, delay)
    env.training_poses, env.training_velocities = poses.clone(), velocities.clone()
    return condition, delay

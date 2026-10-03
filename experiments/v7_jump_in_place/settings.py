"""CPU-only preparation contract. These are research targets, not admission evidence."""
from dataclasses import asdict, dataclass
import math

VERSION = "V7_JUMP_IN_PLACE_PREP_V1"


@dataclass(frozen=True)
class JumpConfig:
    physics_dt_s: float = 0.0025
    policy_dt_s: float = 0.02
    episode_s: float = 6.0
    target_height_m: float = 0.02
    height_tolerance_m: float = 0.005
    minimum_visible_height_m: float = 0.01
    maximum_height_m: float = 0.05
    minimum_com_rise_m: float = 0.01
    settle_s: float = 0.30
    request_s: float = 0.50
    takeoff_deadline_s: float = 2.0  # measured from request
    minimum_crouch_m: float = 0.01
    airborne_force_n: float = 0.5
    support_force_n: float = 1.0
    detection_clearance_m: float = 0.003  # sensor hysteresis, NEVER a jump target
    takeoff_debounce_s: float = 0.01
    minimum_takeoff_com_vz_mps: float = 0.15
    minimum_flight_s: float = 0.06
    maximum_flight_s: float = 0.8
    touchdown_pair_window_s: float = 0.06
    recovery_deadline_s: float = 2.5
    recovery_hold_s: float = 1.0
    recovery_tilt_deg: float = 5.0
    maximum_tilt_deg: float = 10.0
    recovery_speed_mps: float = 0.05
    recovery_gyro_rad_s: float = 0.5
    recovery_drift_m: float = 0.05
    maximum_drift_m: float = 0.10
    recovery_leg_height_low_m: float = 0.17
    recovery_leg_height_high_m: float = 0.19
    leg_height_low_m: float = 0.14
    leg_height_high_m: float = 0.22
    leg_motor_torque_limit_nm: float = 6.0
    wheel_torque_limit_nm: float = 1.5
    leg_motor_speed_limit_rad_s: float = 6.0
    wheel_speed_limit_rad_s: float = 20.0
    mimic_q_limit_rad: float = 0.001
    mimic_v_limit_rad_s: float = 0.01
    nonwheel_force_limit_n: float = 5.0
    saturation_stop_s: float = 0.02

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in asdict(self).values()):
            raise ValueError("All numerical settings must be finite and positive")
        if not 0.01 <= self.target_height_m <= 0.05:
            raise ValueError("User scope is 1--5 cm; millimetre targets are prohibited")
        if self.minimum_visible_height_m < 0.01 or self.maximum_height_m > 0.05:
            raise ValueError("Do not weaken the user height bounds")
        if self.height_tolerance_m >= self.target_height_m:
            raise ValueError("Invalid height tolerance")
        if self.support_force_n <= self.airborne_force_n:
            raise ValueError("Contact hysteresis must have separated thresholds")
        for name in ("policy_dt_s", "episode_s", "settle_s", "request_s",
                     "takeoff_debounce_s", "recovery_hold_s", "saturation_stop_s"):
            ticks = getattr(self, name) / self.physics_dt_s
            if abs(ticks - round(ticks)) > 1e-8:
                raise ValueError(f"{name} must align to physics ticks")

    @property
    def height_bounds(self):
        return (max(self.minimum_visible_height_m, self.target_height_m - self.height_tolerance_m),
                min(self.maximum_height_m, self.target_height_m + self.height_tolerance_m))

    def ticks(self, seconds):
        return int(round(seconds / self.physics_dt_s))

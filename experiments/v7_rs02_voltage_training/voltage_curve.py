"""Published 48 V output curve and the archived, uncalibrated 24 V estimate."""
import math
import torch
from motor_model import TORQUE_KNOTS_NM, OFFICIAL_48_RPM, MODEL


def no_load(voltage):
    if voltage not in (24, 48):
        raise ValueError('Only the prospectively declared 24/48 V comparison is supported')
    return (410 if voltage == 48 else 205)*math.pi/30


def torque_cap(speed, voltage):
    no_load(voltage)
    shift = 205 if voltage == 24 else 0
    speeds = [(n-shift)*math.pi/30 for n in reversed(OFFICIAL_48_RPM)]
    torques = tuple(reversed(TORQUE_KNOTS_NM))
    x = speed.abs()
    out = torch.full_like(x, 17.)
    for i in range(len(speeds)-1):
        lo, hi = speeds[i:i+2]
        v = torques[i]+(torques[i+1]-torques[i])*(x-lo)/(hi-lo)
        out = torch.where((x >= lo) & (x < hi), v, out)
    return torch.where(x >= no_load(voltage), 0., out).clamp(0, 17)


def metadata(voltage):
    return dict(voltage_v=voltage, output_torque_knots_nm=TORQUE_KNOTS_NM,
        output_rpm_knots=[n-(205 if voltage == 24 else 0) for n in OFFICIAL_48_RPM],
        source_pdf=MODEL['source_pdf'], source_pdf_sha256=MODEL['source_pdf_sha256'],
        status='uncalibrated voltage-shift estimate' if voltage == 24 else 'published 48 V points, interpolated',
        assumptions=MODEL['assumptions'], peak_torque_nm=17, no_load_rad_s=no_load(voltage),
        field_weakening=False, wheel_limits='unchanged surrogate 1.5 Nm / 20 rad/s', hardware_used=False)

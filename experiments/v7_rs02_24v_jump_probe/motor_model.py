"""Explicit uncalibrated24V estimate derived from the published48V curve."""
import math
import torch

VOLTAGE=24.
RATED_VOLTAGE=48.
NOLOAD_48_RPM=410.
TORQUE_KNOTS_NM=(0.,.5,7.,10.,14.,17.)
OFFICIAL_48_RPM=(410.,407.,365.,326.,273.,219.)
ESTIMATED_24_RPM=tuple(max(0.,n-(1-VOLTAGE/RATED_VOLTAGE)*NOLOAD_48_RPM) for n in OFFICIAL_48_RPM)
PEAK_TORQUE_NM=17.
NOLOAD_RAD_S=ESTIMATED_24_RPM[0]*math.pi/30
PEAK_EXPOSURE_THRESHOLD_NM=6.
PEAK_EXPOSURE_LIMIT_S=.5

MODEL=dict(
    version='RS02_24V_EMPIRICAL_VOLTAGE_SHIFT_V1',voltage_v=VOLTAGE,
    source_voltage_v=RATED_VOLTAGE,source_pdf='mechanical/serial_robot_v7_deep_crouch/reference/RS_series_official_20260917.pdf',
    source_pdf_sha256='76c85c3c11f15bf3adc1676d6a4b8c931ffd9c18cec41221f8ea5a8b54e22ea1',
    source_pages=[11,12,13],
    source_url='https://github.com/RobStride/Product_Information/blob/main/灵足时代RS系列产品规格介绍(2026.09.17).pdf',
    official_48v_points=[dict(torque_nm=t,rpm=n) for t,n in zip(TORQUE_KNOTS_NM,OFFICIAL_48_RPM)],
    estimated_24v_points=[dict(torque_nm=t,rpm=n,rad_s=n*math.pi/30) for t,n in zip(TORQUE_KNOTS_NM,ESTIMATED_24_RPM)],
    formula='n24(T)=max(0,n48(T)-410*(1-24/48))rpm;piecewise linear inverse torque versus absolute output speed',
    assumptions=[
        'No-load speed is proportional to bus voltage; nominal24V no-load205rpm,not a manufacturer24V measurement.',
        'At fixed output torque, an effective voltage loss fitted to48V data is held constant when lowering voltage. Speed-dependent electrical and mechanical losses may violate this approximation.',
        'Below the highest-torque data point,17Nm is an assumed short-time current cap; original rotor inertia and transmission geometry are unchanged.',
        'Apply the same speed-dependent torque cap to drive and braking. Regeneration,bus overvoltage and braking electronics are not identified.',
        'Nominal24V is fixed; battery sag,driver current dynamics,temperature and real mounted cooling are not modeled.'
    ],
    uncertainty='Research approximation,not certified24V performance and not guaranteed conservative.',
    peak_torque_nm=PEAK_TORQUE_NM,no_load_speed_rad_s=NOLOAD_RAD_S,
    official_overload_reference='Published17Nm:rotation9s,stall6s under stated cooling;not a24V mounted-robot guarantee.',
    research_burst_guard=dict(per_motor_accumulated_above_nm=PEAK_EXPOSURE_THRESHOLD_NM,limit_s=PEAK_EXPOSURE_LIMIT_S,
                             window='One10sfirstlife,initiallycoldassumption,nocoolingcredit',
                             status='Chosen bounded-test guard,not a validated thermal model'),
    wheels='Unchanged M3508 surrogate1.5Nm/20rad_s;not powered by a48V substitution',
    hardware_used=False)


def torque_cap(speed):
    """Output-side absolute torque cap, with continuous interpolation and no state clipping."""
    x=speed.abs()
    speeds=tuple(n*math.pi/30 for n in reversed(ESTIMATED_24_RPM))
    torques=tuple(reversed(TORQUE_KNOTS_NM))
    cap=torch.full_like(x,PEAK_TORQUE_NM)
    for i in range(len(speeds)-1):
        lo,hi=speeds[i:i+2]
        value=torques[i]+(torques[i+1]-torques[i])*(x-lo)/(hi-lo)
        cap=torch.where((x>=lo)&(x<hi),value,cap)
    return torch.where(x>=NOLOAD_RAD_S,torch.zeros_like(cap),cap).clamp(0,PEAK_TORQUE_NM)

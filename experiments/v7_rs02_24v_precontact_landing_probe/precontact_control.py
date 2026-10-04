"""Causal proximity/closing-speed feedback, with bounded precontact retraction.

The distance input is an ideal simulated proximity measurement in this study.
This is not yet a deployable sensor estimator. Every command still uses the
original motor FIFO and torque/speed limits. Postcontact control is inherited.
"""
import torch

DT = .0025
MIN_REFERENCE_M = .160
FIELDS = ('enabled', 'horizon_s', 'closing_target_mps', 'extra_limit_m',
          'gain', 'max_extra_speed_mps', 'max_extra_accel_mps2',
          'brake_start_m', 'brake_full_m', 'brake_kp_extra', 'brake_kd_extra')


def profile(name, horizon=.045, target=.45, budget=.006, gain=1., speed=.65,
            accel=25., enabled=True, brake_start=.140, brake_full=.115,
            brake_kp=.20, brake_kd=.10):
    return dict(name=name, enabled=enabled, horizon_s=horizon,
        closing_target_mps=target, extra_limit_m=budget, gain=gain,
        max_extra_speed_mps=speed, max_extra_accel_mps2=accel,
        brake_start_m=brake_start,brake_full_m=brake_full,
        brake_kp_extra=brake_kp,brake_kd_extra=brake_kd)


PROFILES = [
    profile('air5_current', enabled=False),
    profile('close65_h45_b6', target=.65, speed=.45),
    profile('close45_h45_b6'),
    profile('close25_h45_b9', target=.25, budget=.009, speed=.8),
    profile('close45_h30_b6', horizon=.030),
    profile('close45_h60_b6', horizon=.060),
    profile('close45_h45_b3', budget=.003),
    profile('close45_h45_b9', budget=.009),
    profile('close65_h45_b9', target=.65, budget=.009, speed=.45),
    profile('close25_h60_b9', horizon=.060, target=.25, budget=.009, speed=.8),
    profile('close65_h30_b3', horizon=.030, target=.65, budget=.003, speed=.45),
]

# Predeclared bounded follow-up: two postcontact-only controls distinguish the
# effect of earlier braking from that of precontact wheel-speed preparation.
def coupled_profile(name, *, strong=False, **kw):
    return profile(name,brake_start=.160,brake_full=.125,
        brake_kp=.60 if strong else .35,brake_kd=.30 if strong else .15,**kw)


COUPLED_PROFILES = [
    profile('protected_current',enabled=False),
    coupled_profile('earlier_brake_only',enabled=False),
    coupled_profile('stronger_brake_only',enabled=False,strong=True),
    coupled_profile('late3_earlier',target=.65,horizon=.030,budget=.003,speed=.45),
    coupled_profile('late3_stronger',target=.65,horizon=.030,budget=.003,speed=.45,strong=True),
    coupled_profile('mid6_earlier',target=.65,speed=.45),
    coupled_profile('mid6_stronger',target=.65,speed=.45,strong=True),
    coupled_profile('late6_earlier',target=.45,horizon=.030),
    coupled_profile('late6_stronger',target=.45,horizon=.030,strong=True),
]


def table(profiles, device):
    return torch.tensor([[float(p[k]) for k in FIELDS] for p in profiles],
                        device=device, dtype=torch.float32).repeat_interleave(45, 0)


def initial(n, device):
    return dict(clearance1=torch.zeros(n, 2, device=device),
        clearance2=torch.zeros(n, 2, device=device),
        count=torch.zeros(n, dtype=torch.long, device=device),
        offset=torch.zeros(n, device=device), speed=torch.zeros(n, device=device))


def smooth(x):
    x=x.clamp(0.,1.)
    return x*x*(3-2*x)


def modulate(c, config, previous, *, clearance, surface_z, com_vz, touchdown,
             leg_height, protect_landing=False):
    enabled,horizon,target,budget,gain,vmax,amax,brake_start,brake_full,brake_kp,brake_kd=config.unbind(1)
    valid=previous['count']>=2
    wheel_vz=(clearance-previous['clearance2'])/(2*DT)
    wheel_vz=torch.where(valid[:,None],wheel_vz,torch.zeros_like(wheel_vz))
    closing=(-wheel_vz).clamp_min(0).max(1).values
    gap=(clearance-surface_z[:,None]).min(1).values
    ttc=gap.clamp_min(0)/closing.clamp_min(.05)
    # A shared 30-60 ms horizon anticipates the retained 0-20 ms motor delay.
    # Never read the actual per-case FIFO delay, future touchdown, or future state.
    proximity=smooth((horizon-ttc)/(horizon*.5))
    active=c['enabled'] & (enabled>.5) & (touchdown<=0) & (com_vz<-.15) & valid & (c['height']>=MIN_REFERENCE_M)
    old_offset=torch.where(active,previous['offset'],torch.zeros_like(gap))
    remaining=(budget+old_offset).clamp_min(0)
    budget_blend=smooth(remaining/.003)
    desired=(gain*(closing-target).clamp_min(0)).clamp_max(vmax)*proximity*budget_blend
    proposed=torch.minimum(torch.maximum(desired,previous['speed']-amax*DT),
                           previous['speed']+amax*DT).clamp_min(0)
    proposed=torch.where(active,proposed,torch.zeros_like(proposed))
    offset=torch.maximum(old_offset-proposed*DT,-budget)
    # Reserve nominal leg travel without calling it an exact COM stroke bound.
    floor=(MIN_REFERENCE_M-c['height']).clamp_max(0)
    offset=torch.maximum(offset,floor)
    offset=torch.where(active,offset,torch.zeros_like(offset))
    delta_v=torch.where(active,(offset-old_offset)/DT,torch.zeros_like(offset))
    out=dict(c)
    out['height']=c['height']+offset
    out['velocity']=c['velocity']+delta_v
    # Optional, separately baselined protection against the observed dynamic
    # lower-link/body contact near 101 mm. This is motor impedance, not an
    # external force or a relaxation of the original collision gate.
    post=c['enabled'] & (touchdown>0) & protect_landing
    depth=torch.where(post,smooth((brake_start-leg_height.min(1).values)/(brake_start-brake_full)),torch.zeros_like(gap))
    out['kp']=(c['kp']*(1+brake_kp*depth)).clamp_max(1.)
    out['kd']=(c['kd']*(1+brake_kd*depth)).clamp_max(1.)
    state=dict(clearance1=clearance.clone(),clearance2=previous['clearance1'].clone(),
        count=(previous['count']+1).clamp_max(2),offset=offset,
        speed=torch.where(active,(-delta_v).clamp_min(0),torch.zeros_like(delta_v)))
    diag=dict(pre_active=active,pre_gap_m=gap,pre_wheel_vz_mps=wheel_vz,
        pre_ttc_s=ttc,pre_weight=proximity,pre_offset_m=offset,
        pre_offset_velocity_mps=delta_v,pre_original_height_m=c['height'],
        pre_original_velocity_mps=c['velocity'],pre_requested_height_m=out['height'],
        pre_requested_velocity_mps=out['velocity'],pre_original_kp=c['kp'],
        pre_original_kd=c['kd'],pre_original_force=c['force'],
        pre_requested_kp=out['kp'],pre_requested_kd=out['kd'],
        pre_landing_brake=depth,pre_leg_height_m=leg_height.clone(),
        pre_surface_z_m=surface_z,pre_absolute_clearance_m=clearance.clone(),
        pre_com_vz_mps=com_vz.clone(),pre_touchdown_s=touchdown.clone())
    return out,diag,state

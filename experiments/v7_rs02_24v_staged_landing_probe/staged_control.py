"""Causal staged impedance; keep air velocity tracking and the existing FIFO."""
import torch
from compliant_control import smooth

FIELDS=('enabled','early_kp','late_kp','early_kd','late_kd',
        'early_support','late_support','ramp_m')


def profile(name,early_kp=1.,late_kp=1.,early_kd=1.,late_kd=1.,
            early_support=1.,late_support=1.,ramp_m=.015,enabled=True):
    return dict(name=name,enabled=enabled,early_kp=early_kp,late_kp=late_kp,
                early_kd=early_kd,late_kd=late_kd,early_support=early_support,
                late_support=late_support,ramp_m=ramp_m)


PROFILES=[
    profile('air5_current',enabled=False),
    profile('soft85',early_kp=.85),
    profile('soft60',early_kp=.60),
    profile('soft35',early_kp=.35),
    profile('soft60_ramp8',early_kp=.60,ramp_m=.008),
    profile('soft60_ramp25',early_kp=.60,ramp_m=.025),
    profile('soft60_contact_damp80',early_kp=.60,early_kd=.80),
    profile('soft60_late125',early_kp=.60,late_kp=1.25,late_kd=1.20),
    profile('soft60_support50',early_kp=.60,early_support=.50),
    profile('soft35_late125',early_kp=.35,late_kp=1.25,late_kd=1.20),
    profile('soft60_support0',early_kp=.60,early_support=0.),
]


def table(profiles,device):
    return torch.tensor([[float(p[k]) for k in FIELDS] for p in profiles],
                        device=device,dtype=torch.float32).repeat_interleave(45,0)


def modulate(c,p,config,previous,*,ticks,touchdown,touch_com,com_z,com_vz,mass):
    """Only gain/support scheduling changes; the original reference is retained.

    The stopping-force estimate is an urgency signal, not a fake force clamp.
    All requested gain/support changes subsequently traverse the real FIFO.
    """
    enabled,ekp,lkp,ekd,lkd,es,ls,ramp=config.unbind(1)
    active=c['enabled'] & (enabled>.5)
    seen=touchdown>0
    drop=torch.where(seen,(touch_com-com_z).clamp_min(0),torch.zeros_like(com_z))
    remaining=(p[:,2]-drop).clamp_min(.005)
    need=mass*(9.81+com_vz.clamp_max(0).square()/(2*remaining))
    distance=smooth((drop/ramp).clamp(0,1))
    urgency=smooth(((need-250.)/50.).clamp(0,1))
    progress=torch.where(active & seen,torch.maximum(previous,torch.maximum(distance,urgency)),
                         torch.zeros_like(previous))
    age=(ticks*.0025-touchdown).clamp_min(0)
    recovery=smooth(((age-p[:,3])/p[:,4]).clamp(0,1))
    recovery=torch.where(seen,recovery,torch.zeros_like(recovery))
    air_age=(ticks-c['state']['gate_tick'])*.0025
    ready=smooth((air_age/.04).clamp(0,1))
    amplitude=ready*(1-recovery)
    kpr=torch.where(seen,ekp+(lkp-ekp)*progress,ekp)
    # The air Kd is unchanged, including its velocity-reference contribution.
    kdr=torch.where(seen,ekd+(lkd-ekd)*progress,torch.ones_like(ekd))
    sr=es+(ls-es)*progress
    original={k:c[k] for k in ('kp','kd','force')}
    changed=dict(kp=(c['kp']*(1+(kpr-1)*amplitude)).clamp(.05,1.),
                 kd=(c['kd']*(1+(kdr-1)*amplitude)).clamp(.05,1.),
                 force=c['force']*(1+(sr-1)*(1-recovery)))
    out=dict(c)
    for k in changed:
        out[k]=torch.where(active,changed[k],original[k])
    diagnostics=dict(stage_progress=progress,stage_drop_m=drop,stage_remaining_m=remaining,
        stage_required_force_n=need,stage_touch_seen_pre=seen,
        stage_requested_kp=out['kp'],stage_requested_kd=out['kd'],stage_requested_force=out['force'],
        stage_original_kp=original['kp'],stage_original_kd=original['kd'],stage_original_force=original['force'])
    return out,diagnostics,progress

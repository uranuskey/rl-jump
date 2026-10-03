"""Causal bounded curve state, evaluated identically by CPU checks and native rollout."""
import torch
from profiles import height as archived_height

ACTOR_DIM,CRITIC_DIM,ACTION_DIM=145,177,9
DT=.02
OFFSET_LIMIT=.05
HEIGHT_MIN,HEIGHT_MAX=.095,.225
VELOCITY_SCALE=1.5
FORCE_SCALE=300.
BASE_FF=.34
FF_SCALE=.34


def base_reference(ticks):
    t=ticks*.0025
    old=archived_height(t,torch.full_like(ticks,2))
    old_u=((t-4.3)/.12).clamp(0,1)
    old_curve=10*old_u**3-15*old_u**4+6*old_u**5
    offset=ticks-1720
    u=(offset.to(t.dtype)/40).clamp(0,1)
    h=old+.10*(u*u-old_curve)
    v=torch.where((offset>=0)&(offset<40),2.*.1/.1*u,0.)
    return h,v


def initial(n,device='cpu',dtype=torch.float32):
    state=torch.zeros(n,4,device=device,dtype=dtype)
    state[:,3]=BASE_FF
    return state


def features(state):
    return torch.stack((state[:,0]/OFFSET_LIMIT,state[:,1]/VELOCITY_SCALE,
                        state[:,2]/FORCE_SCALE,(state[:,3]-BASE_FF)/FF_SCALE),1)


def advance(state,raw,ticks,active):
    """Offset velocity changes the reference, signed force and FF are independent actions.

    Offset is integrated once per control step. Bound projection removes windup;
    outward reference velocity is also suppressed at the current height bound.
    No physical q/v or time is overwritten.
    """
    if raw.shape!=(len(state),3):
        raise ValueError('Three raw curve actions required')
    base,_=base_reference(ticks)
    enabled=active & (ticks>=1600)
    bounded=torch.tanh(raw)
    lo=torch.maximum(torch.full_like(base,-OFFSET_LIMIT),HEIGHT_MIN-base)
    hi=torch.minimum(torch.full_like(base,OFFSET_LIMIT),HEIGHT_MAX-base)
    wanted=state[:,0]+DT*VELOCITY_SCALE*bounded[:,0]
    offset=torch.maximum(torch.minimum(wanted,hi),lo)
    velocity=(offset-state[:,0])/DT
    candidate=torch.stack((offset,velocity,FORCE_SCALE*bounded[:,1],
                           BASE_FF+FF_SCALE*bounded[:,2]),1)
    return torch.where(enabled[:,None],candidate,state)


def command_reference(state,ticks):
    base,velocity=base_reference(ticks)
    height=(base+state[:,0]).clamp(HEIGHT_MIN,HEIGHT_MAX)
    velocity=velocity+state[:,1]
    velocity=torch.where(((height<=HEIGHT_MIN+1e-7)&(velocity<0))|
                         ((height>=HEIGHT_MAX-1e-7)&(velocity>0)),0.,velocity)
    return height,velocity


def compose(raw,commands,private,state,strength):
    from height_contract import compose as old_compose
    actor,critic=old_compose(raw,commands,private)
    extra=torch.cat((features(state),strength[:,None]),1)
    return torch.cat((actor,extra),1),torch.cat((critic,extra),1)


def seed_input(obs):
    """Frozen posture seed retains its legacy3cm conditioning; new head sees5cm."""
    x=obs[:,:140].clone()
    x[:,[31,66,101,136]]=.6
    return x


def height_progress(new,old,target=.15):
    return 6*(new.clamp(0,target)-old.clamp(0,target))/.05

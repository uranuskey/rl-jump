"""A sampled whole-jump plan; no fixed 4.3 s launch and no old action residual."""
import torch
HANDOFF=.6
NAMES=('crouch_m','crouch_s','hold_s','extend_m','push_s','push_shape','early_force_n','late_force_n','velocity_ff')
LOW=(.095,.18,0.,.190,.060,1.,0.,0.,0.)
HIGH=(.170,.65,.12,.225,.250,3.,300.,300.,1.)
INITIAL=(.120,.35,.03,.220,.100,2.,100.,80.,.34)

def decode(raw):
    low=torch.tensor(LOW,device=raw.device,dtype=raw.dtype)
    return low+(torch.tensor(HIGH,device=raw.device,dtype=raw.dtype)-low)*raw.sigmoid()

def initial_raw(device='cpu'):
    x=(torch.tensor(INITIAL,device=device)-torch.tensor(LOW,device=device))/(torch.tensor(HIGH,device=device)-torch.tensor(LOW,device=device))
    return torch.logit(x)

def reference(plan,t):
    crouch,down,hold,end,up,power,early,late,ff=plan.unbind(-1)
    age=t-HANDOFF
    d=(age/down).clamp(0,1)
    h=.18+(crouch-.18)*(.5-.5*torch.cos(torch.pi*d))
    v=torch.where((age>=0)&(age<down),(crouch-.18)*.5*torch.pi/down*torch.sin(torch.pi*d),0.)
    start=down+hold
    u=((age-start)/up).clamp(0,1)
    extending=age>=start
    ph=crouch+(end-crouch)*u.pow(power)
    pv=torch.where((age>=start)&(age<start+up),(end-crouch)/up*power*u.clamp_min(1e-9).pow(power-1),0.)
    h=torch.where(extending,ph,h)
    v=torch.where(extending,pv,v)
    force=torch.where((age>=start)&(age<start+up),early+(late-early)*u,0.)
    return torch.stack((h-.18,v,force,ff),1)

"""Position and analytic velocity share one prospectively frozen profile specification."""
import json
import math
from pathlib import Path
import torch

SPEC = json.loads((Path(__file__).parent/'SPEC.json').read_text(encoding='utf-8'))
PROFILES = SPEC['profiles']
SHAPES = ('cosine','early_beta23','late_beta32','quintic')


def parameters(t,index):
    if index is None:
        index=torch.zeros_like(t,dtype=torch.long)
    vals=torch.tensor([[p['crouch'],p['extend'],p['up_s'],SHAPES.index(p['shape'])] for p in PROFILES],device=t.device,dtype=t.dtype)[index]
    return vals.unbind(-1)


def smooth(x):
    return .5*(1-torch.cos(math.pi*x.clamp(0,1)))


def curve(u,shape):
    u=u.clamp(0,1)
    values=(smooth(u),6*u.square()-8*u.pow(3)+3*u.pow(4),4*u.pow(3)-3*u.pow(4),10*u.pow(3)-15*u.pow(4)+6*u.pow(5))
    out=values[0]
    for i in range(1,len(values)):
        out=torch.where(shape==i,values[i],out)
    return out


def height(t,index=None):
    crouch,end,duration,shape=parameters(t,index)
    return .18-(.18-crouch)*smooth((t-1.)/3.)+(end-crouch)*curve((t-4.3)/duration,shape)-(end-.18)*smooth((t-5.)/.6)


def push_height_velocity(t,index=None):
    crouch,end,duration,shape=parameters(t,index)
    # Python subtraction matches the archived cosine derivative's float32 arithmetic.
    if index is None:
        index=torch.zeros_like(t,dtype=torch.long)
    amplitude=torch.tensor([p['extend']-p['crouch'] for p in PROFILES],device=t.device,dtype=t.dtype)[index]
    p=(t-4.3)/duration
    u=p.clamp(0,1)
    out=torch.zeros_like(t)
    for i,profile in enumerate(PROFILES):
        archived_order=(profile['extend']-profile['crouch'])*math.pi/(2*duration)*torch.sin(math.pi*p)
        out=torch.where(index==i,archived_order,out)
    derivatives=(None,12*u*(1-u).square(),12*u.square()*(1-u),30*u.square()*(1-u).square())
    for i in range(1,len(derivatives)):
        out=torch.where(shape==i,amplitude/duration*derivatives[i],out)
    return torch.where((p>0)&(p<1),out,0.)


def fraction(index,dtype):
    return torch.tensor([p['velocity_fraction'] for p in PROFILES],device=index.device,dtype=dtype)[index]

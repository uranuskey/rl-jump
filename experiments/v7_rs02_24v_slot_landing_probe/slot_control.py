"""Encoder-based fore/aft impedance routed through the existing motor FIFO.

The virtual Cartesian spring is implemented as a bounded motor target offset,
not an external wrench. Per-leg common allocation preserves its Jx direction.
"""
import json
from functools import lru_cache
from pathlib import Path
import sys
import torch

HERE=Path(__file__).resolve().parent
PRE=HERE.parent/'v7_rs02_24v_precontact_landing_probe'
sys.path.insert(0,str(PRE))
from precontact_control import profile
CAD=HERE.parent/'v7_rs02_24v_geometry_pose_probe/cad_path.json'
NOMINAL=[r for r in json.loads(CAD.read_text())['samples'] if r['dx_perturbation_mm']==0]
FIELDS=('slot_kx','slot_dx','slot_force_cap_n')


def make(name,k=0.,d=0.,cap=0.,**kw):
    return dict(profile(name,**kw),slot_kx=k,slot_dx=d,slot_force_cap_n=cap)


PROFILES=[
    make('protected_baseline',enabled=False),
    make('slot_only',1500.,16.,25.,enabled=False),
    make('mid6_slot_low',600.,8.,15.,target=.65,speed=.45),
    make('mid6_slot_mid',1500.,16.,25.,target=.65,speed=.45),
    make('mid6_slot_high',3000.,32.,40.,target=.65,speed=.45),
    make('late3_slot_mid',1500.,16.,25.,target=.65,horizon=.030,budget=.003,speed=.45),
    make('late3_slot_high',3000.,32.,40.,target=.65,horizon=.030,budget=.003,speed=.45),
    make('late6_slot_mid',1500.,16.,25.,target=.45,horizon=.030),
    make('late6_slot_high',3000.,32.,40.,target=.45,horizon=.030),
]


def table(profiles,n,device):
    t=torch.tensor([[p[k] for k in FIELDS] for p in profiles],device=device,dtype=torch.float32)
    if len(profiles)==1:return t.expand(n,-1).clone()
    assert n==45*len(profiles)
    return t.repeat_interleave(45,0)


def kinematics(q,v):
    angle=q.reshape(-1,2,2).cumsum(-1)
    speed=v.reshape(-1,2,2).cumsum(-1)
    length=q.new_tensor([.105,.145])
    jx=length*angle.cos();jh=-length*angle.sin()
    return dict(x=(length*angle.sin()).sum(-1),height=jx.sum(-1),
        vx=(jx*speed).sum(-1),vh=(jh*speed).sum(-1),jx=jx)


@lru_cache(maxsize=8)
def path_tensors(device,dtype):
    return (torch.tensor([r['h_mm']/1000 for r in NOMINAL],device=device,dtype=dtype),
            torch.tensor([-r['dx_mm']/1000 for r in NOMINAL],device=device,dtype=dtype))


def target(height):
    hs,xs=path_tensors(str(height.device),height.dtype)
    h=height.clamp(float(NOMINAL[0]['h_mm']/1000),float(NOMINAL[-1]['h_mm']/1000))
    index=((h-.09)/.002).floor().long().clamp(0,len(NOMINAL)-2)
    slope=(xs[index+1]-xs[index])/(hs[index+1]-hs[index])
    return xs[index]+slope*(h-hs[index]),slope


def feedback(c,config,q,v):
    k,d,cap=config.unbind(1)
    state=kinematics(q[:,:4],v[:,:4])
    xt,slope=target(state['height']);vt=slope*state['vh']
    # Full correction during compression; gently introduce it on descent.
    u=((.20-state['height'])/.04).clamp(0,1);blend=u*u*(3-2*u)
    active=c['enabled'][:,None] & (k[:,None]>0)
    force=(k[:,None]*(xt-state['x'])+d[:,None]*(vt-state['vx'])).clamp(-cap[:,None],cap[:,None])
    force=torch.where(active,force*blend,torch.zeros_like(force))
    torque=state['jx']*force[:,:,None]
    original=torch.tanh(c['correction'])
    old=original[:,:4].reshape(-1,2,2)
    delta=torque/(60*c['kp'][:,None,None]*.06)
    denom=torch.where(delta.abs()>1e-10,delta,torch.ones_like(delta))
    bound=torch.where(delta>1e-10,(1-old)/denom,
        torch.where(delta< -1e-10,(-1-old)/denom,torch.ones_like(delta)))
    alpha=bound.min(-1).values.clamp(0,1)
    addition=delta*alpha[:,:,None]
    result=original.clone();result[:,:4]=(old+addition).reshape(-1,4)
    diag=dict(slot_input_q=q[:,:4].clone(),slot_input_v=v[:,:4].clone(),
        slot_x_m=state['x'],slot_height_m=state['height'],slot_vx_mps=state['vx'],
        slot_target_x_m=xt,slot_target_vx_mps=vt,slot_error_m=xt-state['x'],
        slot_requested_force_n=force,slot_allocated_force_n=force*alpha,
        slot_motor_increment_nm=torque*alpha[:,:,None],slot_allocation=alpha,
        slot_original_action=original,slot_action=result,slot_enabled=c['enabled'].clone(),
        slot_kp=c['kp'].clone())
    return result,diag

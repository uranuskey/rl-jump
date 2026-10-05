"""Slot residual shares a bounded motor-offset window; actual motor limits remain."""
import torch
from slot_control import kinematics,target
FIELDS=('slot_kx','slot_dx','slot_force_cap_n','slot_action_bound')
def table(profiles,n,device):
    t=torch.tensor([[p[k] for k in FIELDS] for p in profiles],device=device,dtype=torch.float32)
    if len(profiles)==1:return t.expand(n,-1).clone()
    assert n==45*len(profiles)
    return t.repeat_interleave(45,0)
def feedback(c,config,q,v):
    k,d,cap,window=config.unbind(1)
    state=kinematics(q[:,:4],v[:,:4])
    xt,slope=target(state['height']);vt=slope*state['vh']
    u=((.20-state['height'])/.04).clamp(0,1);blend=u*u*(3-2*u)
    active=c['enabled'][:,None] & (k[:,None]>0)
    force=(k[:,None]*(xt-state['x'])+d[:,None]*(vt-state['vx'])).clamp(-cap[:,None],cap[:,None])
    force=torch.where(active,force*blend,torch.zeros_like(force))
    torque=state['jx']*force[:,:,None]
    original=torch.tanh(c['correction'])
    old=original[:,:4].reshape(-1,2,2)
    delta=torque/(60*c['kp'][:,None,None]*.06)
    denom=torch.where(delta.abs()>1e-10,delta,torch.ones_like(delta))
    w=window[:,None,None]
    bound=torch.where(delta>1e-10,(w-old)/denom,
        torch.where(delta < -1e-10,(-w-old)/denom,torch.ones_like(delta)))
    alpha=bound.min(-1).values.clamp(0,1)
    result=original.clone()
    result[:,:4]=(old+delta*alpha[:,:,None]).reshape(-1,4)
    diag=dict(slot_input_q=q[:,:4].clone(),slot_input_v=v[:,:4].clone(),
        slot_x_m=state['x'],slot_height_m=state['height'],slot_vx_mps=state['vx'],
        slot_target_x_m=xt,slot_target_vx_mps=vt,slot_error_m=xt-state['x'],
        slot_requested_force_n=force,slot_allocated_force_n=force*alpha,
        slot_motor_increment_nm=torque*alpha[:,:,None],slot_allocation=alpha,
        slot_original_action=original,slot_action=result,slot_enabled=c['enabled'].clone(),
        slot_kp=c['kp'].clone(),slot_action_bound=window.clone())
    return result,diag

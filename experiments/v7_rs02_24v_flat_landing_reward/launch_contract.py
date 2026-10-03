"""Add a feasible world-up support-force increment in coupled motor coordinates.

Keep the archived PD and velocity feedforward. Add only the increment that fits
inside 99% of the instantaneous motor envelope, using one common scale for all
four leg motors. Base-controller excess still reaches the original limiter and
20 ms stop. The desired force is a planning request, not measured ground force.
"""
import torch
from velocity_contract import servo as previous_servo, payload as previous_payload
from height_contract import to_motor, UPPER, LOWER
from voltage_curve import torque_cap
SPEC={'force_allocator_max_cap_fraction':.99}


def payload(action,height,velocity,force):
    if force.shape != height.shape:
        raise ValueError('One total force increment per world required')
    return torch.cat((previous_payload(action,height,velocity),force[:,None]),1)


def force_to_motor(q,rotation,force):
    # Reaction on robot is upward in world coordinates; project into each XZ leg plane.
    # Active motor coordinates are hip angle and absolute lower-link angle.
    angles=to_motor(q[:,:4]).reshape(-1,2,2)
    fx=rotation[:,2,0,None,None]*force[:,None,None]/2
    fz=rotation[:,2,2,None,None]*force[:,None,None]/2
    lengths=torch.tensor([UPPER,LOWER],device=q.device,dtype=q.dtype)
    return (-lengths*(torch.cos(angles)*fx+torch.sin(angles)*fz)).reshape(-1,4)


def allocate(base,delta,cap,paired):
    available=SPEC['force_allocator_max_cap_fraction']*cap
    denom=torch.where(delta.abs()>1e-10,delta,torch.ones_like(delta))
    bound=torch.where(delta>1e-10,(available-base)/denom,
        torch.where(delta < -1e-10,(-available-base)/denom,torch.ones_like(delta)))
    alpha=bound.min(1).values.clamp(0,1)
    # Do not hide or repair a base-controller limit violation in the new allocator.
    allowed=paired & (base.abs()<=cap).all(1) & (delta.abs().max(1).values>0)
    return torch.where(allowed,alpha,torch.zeros_like(alpha))


def servo(arrived,q,v,rotation,support,voltage=24):
    if arrived.shape[1]!=12:
        raise ValueError('All 12 reference / action / force channels must share the FIFO')
    out=previous_servo(arrived[:,:11],q,v)
    base=out['motor_request']
    delta=force_to_motor(q,rotation,arrived[:,11])
    paired=support.min(1).values>=1.
    cap=torque_cap(to_motor(v[:,:4]),voltage)
    alpha=allocate(base,delta,cap,paired)
    applied=delta*alpha[:,None]
    request=base+applied
    pairs=request.reshape(-1,2,2)
    out.update(motor_request=request,joint_request=torch.stack((pairs.sum(-1),pairs[...,1]),-1).reshape(-1,4),
        thrust_requested_force_n=arrived[:,11],thrust_motor_unlimited_nm=delta,thrust_motor_applied_nm=applied,
        thrust_allocation_fraction=alpha,thrust_pair_contact=paired,motor_request_before_thrust_nm=base,
        base_rotation_pre=rotation.clone())
    return out

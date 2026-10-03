"""World-frame tilt PD, with identically zero linear force and yaw torque."""
import torch
KP=120.0
KD=8.0
TORQUE_CAP=20.0
LEVELS=(1.0,.75,.5,.25,.125,0.0)


def wrench(rotation,body_omega,strength,active):
    world_omega=(rotation@body_omega[...,None]).squeeze(-1)
    up=rotation[:,:,2]
    restoring=torch.stack((up[:,1],-up[:,0],torch.zeros_like(up[:,0])),1)
    damping=world_omega.clone()
    damping[:,2]=0
    requested=KP*restoring-KD*damping
    norm=torch.linalg.vector_norm(requested,dim=1).clamp_min(1e-12)
    torque=requested*(TORQUE_CAP/norm).clamp(max=1)[:,None]
    torque=torque*strength[:,None]*active[:,None]
    force_torque=torch.cat((torch.zeros_like(torque),torque),1)
    return force_torque,world_omega


def next_level(index,evaluation):
    # One successful validation reduces one level; no time-only withdrawal.
    ready=(evaluation['passed_cases']>=36 and evaluation['cases_at_least_1cm']>=40)
    return min(index+1,len(LEVELS)-1) if ready else index

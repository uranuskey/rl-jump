"""Additional measurements; the only controller revision is a declared slot profile."""
import diagnostic_runtime
import torch
from fix_env import ConstraintEnv
from slot_control import kinematics,target

class ClearanceEnv(ConstraintEnv):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self.min_height=torch.ones(self.n,device=self.device)
        self.min_height_q=torch.zeros_like(self.q)
        self.worst_slot_error=torch.zeros(self.n,device=self.device)
    def reset(self,mask,**kw):
        ret=super().reset(mask,**kw)
        if hasattr(self,'min_height'):
            self.min_height[mask]=1.;self.min_height_q[mask]=0;self.worst_slot_error[mask]=0
        return ret
    def sensors(self,motor=None):
        x=super().sensors(motor)
        if hasattr(self,'min_height'):
            state=kinematics(self.q[:,self.qids[:4]],self.v[:,self.main[:4]])
            xt,_=target(state['height'])
            h=state['height'].min(1).values
            live=(self.task.touchdown_time>0)&~self.paused
            lower=live&(h<self.min_height)
            self.min_height=torch.where(lower,h,self.min_height)
            self.min_height_q=torch.where(lower[:,None],self.q,self.min_height_q)
            error=(xt-state['x']).abs().max(1).values
            self.worst_slot_error=torch.maximum(self.worst_slot_error,torch.where(live&(h<.14),error,0.))
        return x

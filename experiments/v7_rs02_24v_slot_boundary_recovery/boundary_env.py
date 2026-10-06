"""Same controller and physics; expanded declared assistance levels and force evidence."""
import zero_runtime
import torch
from allocation_env_v2 import AllocationEnvV2
from boundary_contract import checked_levels

class BalancedAssistEnv(AllocationEnvV2):
    def __init__(self,n,profiles,*,before,after,**kwargs):
        checked_levels(before,after)
        # Ancestor's immutable constructor accepts its historical levels only.
        # No rollout is admitted until trial resets at the new requested level.
        super().__init__(n,profiles,before=.5,after=.5,**kwargs)
        self.before,self.after,self.assist_target=before,after,after
        self.assist_strength.fill_(before)
        self.external_peak=torch.zeros((n,3),device=self.device)
        self.external_sample_ticks=torch.zeros(n,dtype=torch.long,device=self.device)

    def reset(self,mask,**kwargs):
        if hasattr(self,'external_peak'):
            self.external_peak[mask]=0
            self.external_sample_ticks[mask]=0
        return super().reset(mask,**kwargs)

    def sensors(self,motor=None):
        x=super().sensors(motor)
        if motor is not None and hasattr(self,'external_peak'):
            force=self.external[:,:,:3].abs().amax((1,2))
            torque=self.external[:,:,3:].abs().amax((1,2))
            root=self.force[:,:6].abs().amax(1)
            self.external_peak.copy_(torch.maximum(self.external_peak,torch.stack((force,torque,root),1)))
            self.external_sample_ticks.add_((~self.paused).long())
            if self.record:
                x['external_body_wrench']=self.external.clone()
                x['external_root_generalized_force']=self.force[:,:6].clone()
                x['external_base_body_id']=torch.full_like(self.ticks,self.base)
        return x

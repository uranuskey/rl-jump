"""Restore the inherited causal apex bookkeeping before slot control."""
from allocation_env import AllocationEnv
import torch
class AllocationEnvV2(AllocationEnv):
    def control_tick(self,held):
        if hasattr(self,'assist_apex_tick'):
            first=(self.assist_apex_tick<0) & self.task.height_score.apex & ~self.terminal_mask()
            self.assist_apex_tick.copy_(torch.where(first,self.ticks,self.assist_apex_tick))
            u=((self.ticks-self.assist_apex_tick).float()/40).clamp(0,1)
            blend=torch.where(self.assist_apex_tick>=0,u*u*(3-2*u),0.)
            self.assist_strength.copy_(self.before+(self.after-self.before)*blend)
        return super().control_tick(held)

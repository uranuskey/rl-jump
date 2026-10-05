"""Vary attitude assistance during crouch/takeoff as well as after the apex."""
import probe_runtime
import torch
from apex_env import ApexEnv
from slot_env import SlotEnv
from probe_contract import checked_levels


class WithdrawalEnv(ApexEnv):
    def __init__(self, n, profiles, *, before, after, **kwargs):
        self.before, self.after = checked_levels(before, after)
        super().__init__(n, profiles, target=.60, **kwargs)
        self.assist_target = after
        self.assist_strength.fill_(before)

    def reset(self, mask, **kwargs):
        if hasattr(self, 'assist_apex_tick'):
            self.assist_apex_tick[mask] = -1
            for value in (self.max_pre_mimic_q, self.max_pre_mimic_v,
                          self.max_all_mimic_q, self.max_all_mimic_v):
                value[mask] = 0
        if hasattr(self, 'assist_strength'):
            self.assist_strength[mask] = self.before
        # The reset observation must also encode the lower starting assistance.
        return SlotEnv.reset(self, mask, **kwargs)

    def control_tick(self, held):
        if hasattr(self, 'assist_apex_tick'):
            first = (self.assist_apex_tick < 0) & self.task.height_score.apex & ~self.terminal_mask()
            self.assist_apex_tick.copy_(torch.where(first, self.ticks, self.assist_apex_tick))
            u = ((self.ticks-self.assist_apex_tick).float()/40).clamp(0, 1)
            blend = torch.where(self.assist_apex_tick >= 0, u*u*(3-2*u), 0.)
            self.assist_strength.copy_(self.before+(self.after-self.before)*blend)
        # Deliberately skip ApexEnv.control_tick: it hard-codes 62.5% takeoff.
        return SlotEnv.control_tick(self, held)

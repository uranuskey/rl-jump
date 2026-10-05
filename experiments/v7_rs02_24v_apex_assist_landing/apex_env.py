"""Keep takeoff assistance fixed; reduce it only after the observed COM apex."""
import apex_runtime
import torch
from slot_env import SlotEnv
from apex_runtime import SOURCE_LEVEL, RAMP_TICKS


class ApexEnv(SlotEnv):
    def __init__(self, n, profiles, *, target=.60, **kwargs):
        assert target in (.625, .6125, .60)
        self.assist_target = target
        super().__init__(n, profiles, **kwargs)
        self.assist_apex_tick = torch.full_like(self.ticks, -1)
        self.max_pre_mimic_q = torch.zeros(n, device=self.device)
        self.max_pre_mimic_v = torch.zeros(n, device=self.device)
        self.max_all_mimic_q = torch.zeros(n, device=self.device)
        self.max_all_mimic_v = torch.zeros(n, device=self.device)

    def reset(self, mask, **kwargs):
        if hasattr(self, 'assist_apex_tick'):
            self.assist_apex_tick[mask] = -1
            for value in (self.max_pre_mimic_q, self.max_pre_mimic_v,
                          self.max_all_mimic_q, self.max_all_mimic_v):
                value[mask] = 0
        if hasattr(self, 'assist_strength'):
            self.assist_strength[mask] = SOURCE_LEVEL
        return super().reset(mask, **kwargs)

    def control_tick(self, held):
        if hasattr(self, 'assist_apex_tick'):
            first = (self.assist_apex_tick < 0) & self.task.height_score.apex & ~self.terminal_mask()
            self.assist_apex_tick.copy_(torch.where(first, self.ticks, self.assist_apex_tick))
            u = ((self.ticks-self.assist_apex_tick).float()/RAMP_TICKS).clamp(0, 1)
            blend = torch.where(self.assist_apex_tick >= 0, u*u*(3-2*u), 0.)
            self.assist_strength.copy_(SOURCE_LEVEL+(self.assist_target-SOURCE_LEVEL)*blend)
        return super().control_tick(held)

    def sensors(self, motor=None):
        x = super().sensors(motor)
        if hasattr(self, 'assist_apex_tick'):
            live = ~self.paused
            before = live & (self.assist_apex_tick < 0)
            q, v = x['mimic_q_error_rad'], x['mimic_v_error_rad_s']
            self.max_pre_mimic_q.copy_(torch.maximum(self.max_pre_mimic_q, torch.where(before, q, 0.)))
            self.max_pre_mimic_v.copy_(torch.maximum(self.max_pre_mimic_v, torch.where(before, v, 0.)))
            self.max_all_mimic_q.copy_(torch.maximum(self.max_all_mimic_q, torch.where(live, q, 0.)))
            self.max_all_mimic_v.copy_(torch.maximum(self.max_all_mimic_v, torch.where(live, v, 0.)))
            x['assist_apex_tick'] = self.assist_apex_tick.clone()
        return x

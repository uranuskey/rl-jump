"""Fixed launch followed by a 50 Hz continuous residual landing controller."""
import bootstrap
from dataclasses import replace
import torch
from environment import JumpEnv
from controlled_env import HeightEnv
from plan_contract import reference as launch_reference, HANDOFF
from jump_task import FAILED
from control import advance, reference, ACTION_DIM


class LandingEnv(JumpEnv):
    def __init__(self, n, **kw):
        super().__init__(n, voltage=24, **kw)
        self.cfg = replace(self.cfg, episode_s=5.)
        self.task.cfg = self.cfg
        self.landing_start = torch.zeros(n, device=self.device)
        self.height_offset = torch.zeros(n, device=self.device)
        self.previous_policy_action = torch.zeros(n, ACTION_DIM, device=self.device)
        self.gate_tick = torch.full((n,), -1, device=self.device, dtype=torch.long)
        self.plan_locked = torch.zeros(n, device=self.device, dtype=torch.bool)
        self.locked_plan = self.plan.clone()
        self.effective_action = self.previous_policy_action.clone()
        self.control_gate = self.plan_locked.clone()

    def reset(self, mask, **kw):
        if hasattr(self, 'landing_start'):
            for name in ('landing_start', 'height_offset', 'previous_policy_action',
                         'effective_action', 'control_gate', 'plan_locked'):
                getattr(self, name)[mask] = 0
            self.gate_tick[mask] = -1
        return super().reset(mask, **kw)

    def terminal_mask(self):
        return self.task.phase == FAILED

    def lock_launch(self, plan):
        if bool(self.plan_locked.any()):
            raise RuntimeError('Launch is already locked for this episode')
        self.plan.copy_(plan)
        self.locked_plan.copy_(plan)
        self.plan_locked.fill_(True)

    def assert_launch_fixed(self):
        if not torch.equal(self.plan[self.plan_locked], self.locked_plan[self.plan_locked]):
            raise RuntimeError('Frozen jump plan changed')

    def step(self, standing_actions, *, policy_action=None, auto_reset=False):
        self.assert_launch_fixed()
        time = self.ticks*.0025
        state = launch_reference(self.plan, time)
        apex = self.task.height_score.apex
        enabled = apex & ~self.terminal_mask() & (self.ticks < 2000)
        new = enabled & (self.gate_tick < 0)
        self.gate_tick[new] = self.ticks[new]
        self.landing_start[new] = time[new]
        height, velocity = reference(self.plan, time, self.landing_start, self.task.touchdown_time)
        raw = torch.zeros(self.n, ACTION_DIM, device=self.device) if policy_action is None else policy_action
        self.height_offset, height, velocity, corrections, effective = advance(
            raw, enabled, self.height_offset, height, velocity)
        landing = torch.stack((height-.18, velocity, torch.zeros_like(height), torch.ones_like(height)), 1)
        state = torch.where(apex[:, None], landing, state)
        self.curve_state = torch.where((time>=HANDOFF)[:, None], state, self.curve_state)
        actions = torch.where((time<HANDOFF)[:, None], standing_actions[:, :6], corrections)
        self.control_gate.copy_(enabled)
        self.effective_action.copy_(effective)
        if bool(corrections[~apex].any()):
            raise RuntimeError('Landing controller acted before COM apex')
        out = HeightEnv.step(self, actions, auto_reset=auto_reset)
        self.previous_policy_action.copy_(effective)
        self.assert_launch_fixed()
        return out

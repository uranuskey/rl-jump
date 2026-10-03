"""Fixed launch followed by a 50 Hz continuous residual landing controller."""
import bootstrap
from dataclasses import replace
import torch
from environment import JumpEnv
from controlled_env import HeightEnv
from plan_contract import reference as launch_reference, HANDOFF
from jump_task import FAILED
from control import prepare, ACTION_DIM


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

    def prepare_commands(self, standing_actions, policy_action=None):
        self.assert_launch_fixed()
        raw = torch.zeros(self.n, ACTION_DIM, device=self.device) if policy_action is None else policy_action
        return prepare(self.plan, self.ticks, self.task.height_score.apex, self.terminal_mask(),
            self.gate_tick, self.landing_start, self.height_offset, self.curve_state, standing_actions, raw,
            launch_reference(self.plan, self.ticks*.0025), self.task.touchdown_time, HANDOFF)

    def step(self, standing_actions, *, policy_action=None, auto_reset=False):
        commands = self.prepare_commands(standing_actions, policy_action)
        for name, value in commands.items():
            if name!='actions':
                getattr(self, name).copy_(value)
        if bool(self.effective_action[~self.task.height_score.apex].any()):
            raise RuntimeError('Landing controller acted before COM apex')
        out = HeightEnv.step(self, commands['actions'], auto_reset=auto_reset)
        self.previous_policy_action.copy_(self.effective_action)
        self.assert_launch_fixed()
        return out

"""Parameter control over the unchanged 400 Hz fast/original physics backends."""
import path_setup
import torch
from fast_env import FastLandingEnv
from controlled_env import HeightEnv
from plan_contract import reference as launch_reference
from param_control import ACTION_DIM, initial_raw, prepare


class ParameterEnv(FastLandingEnv):
    def __init__(self, n, *, fast_backend=True, **kwargs):
        self.fast_backend = fast_backend
        super().__init__(n, **kwargs)
        self.parameter_action = initial_raw(self.device).expand(n, -1).clone()
        self.locked_parameters = self.parameter_action.clone()
        self.parameters_locked = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.effective_action = torch.zeros(n, ACTION_DIM, device=self.device)
        self.previous_policy_action = self.effective_action.clone()

    def reset(self, mask, **kwargs):
        if hasattr(self, 'parameter_action'):
            self.parameter_action[mask] = initial_raw(self.device)
            self.parameters_locked[mask] = False
        return super().reset(mask, **kwargs)

    def lock_parameters(self, action):
        if action.shape!=(self.n, ACTION_DIM) or not bool(torch.isfinite(action).all()):
            raise ValueError('Expected one finite 13D landing plan per world')
        if bool(self.parameters_locked.any()):
            raise RuntimeError('Landing parameters already sampled for this episode')
        self.parameter_action.copy_(action)
        self.locked_parameters.copy_(action)
        self.parameters_locked.fill_(True)

    def assert_parameters_fixed(self):
        if bool(((self.parameter_action!=self.locked_parameters)&self.parameters_locked[:, None]).any()):
            raise RuntimeError('Sampled landing parameters changed within an episode')

    def prepare_commands(self, standing_actions, policy_action=None):
        self.assert_launch_fixed()
        self.assert_parameters_fixed()
        raw = self.parameter_action if policy_action is None else policy_action
        if raw.shape!=(self.n, ACTION_DIM):
            raise ValueError('Landing plan must have exactly 13 parameters')
        _, velocity, gyro, linear, gravity, _, _ = self.state()
        return prepare(self.plan, raw, self.ticks, self.task.height_score.apex, self.terminal_mask(),
            self.gate_tick, self.landing_start, self.curve_state, standing_actions,
            launch_reference(self.plan, self.ticks*.0025), self.task.touchdown_time,
            gravity, gyro, linear, velocity[:, 4:])

    def step(self, standing_actions, *, auto_reset=False):
        commands = self.prepare_commands(standing_actions)
        for name, value in commands.items():
            if name!='actions':
                getattr(self, name).copy_(value)
        if bool(((self.effective_action!=0)&~self.task.height_score.apex[:, None]).any()):
            raise RuntimeError('Landing parameters acted before COM apex')
        # The independent evaluator directly uses the original physics method.
        if self.fast_backend:
            out = self._step_controlled(commands['actions'], auto_reset=auto_reset)
        else:
            out = HeightEnv.step(self, commands['actions'], auto_reset=auto_reset)
        self.previous_policy_action.copy_(self.effective_action)
        self.assert_launch_fixed()
        self.assert_parameters_fixed()
        return out

    def sensors(self, motor=None):
        return super().sensors(motor) if self.fast_backend else HeightEnv.sensors(self, motor)

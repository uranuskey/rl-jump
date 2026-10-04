"""Frozen launch; per-physics-step landing reference; unchanged physical guards."""
import compliant_paths
import torch
from fast_env import FastLandingEnv
from rs02_actuator import PhysicalStepFIFO
from plan_contract import reference as launch_reference
from velocity_contract import reference_motor_velocity
from guided_reward import GuidedReward
from compliant_control import ACTION_DIM, initial_raw, tick
from compliant_servo import payload
from compliant_backends import BackendSteps
from controlled_env import HeightEnv


class CompliantEnv(BackendSteps, FastLandingEnv):
    def __init__(self, n, *, fast_backend=True, proof=False, **kw):
        self.fast_backend = fast_backend
        self.proof = proof
        self.proof_required = proof
        self.proof_samples = 0
        self.proof_mixed_ticks = 0
        super().__init__(n, **kw)
        self.mass = float(self.m.body_subtreemass[self.base])
        self.task.landing_reward = GuidedReward(n, self.device, .0025, self.mass)
        self.fifo = PhysicalStepFIFO(n, 15, self.device)
        self.shadow_fifo = PhysicalStepFIFO(n, 12, self.device)
        self.parameter_action = initial_raw(self.device).expand(n, -1).clone()
        self.locked_parameters = self.parameter_action.clone()
        self.parameters_locked = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.effective_action = torch.zeros(n, ACTION_DIM, device=self.device)
        self.previous_policy_action = self.effective_action.clone()
        self.control_state = dict(gate_tick=torch.full((n,), -1, dtype=torch.long, device=self.device),
            air_start=torch.zeros(n, device=self.device), touch_seen=torch.zeros(n, dtype=torch.bool, device=self.device),
            touch_height=torch.zeros(n, device=self.device), touch_velocity=torch.zeros(n, device=self.device))

    def reset(self, mask, **kw):
        if hasattr(self, 'control_state'):
            self.proof = self.proof_required
            for key, value in self.control_state.items():
                value[mask] = -1 if key=='gate_tick' else 0
            self.parameter_action[mask] = initial_raw(self.device)
            self.parameters_locked[mask] = False
            self.shadow_fifo.reset(mask.nonzero().flatten())
        return super().reset(mask, **kw)

    def sensors(self, motor=None):
        return super().sensors(motor) if self.fast_backend else HeightEnv.sensors(self, motor)

    def lock_parameters(self, action):
        assert action.shape==(self.n, ACTION_DIM) and bool(torch.isfinite(action).all())
        assert not bool(self.parameters_locked.any())
        self.parameter_action.copy_(action)
        self.locked_parameters.copy_(action)
        self.parameters_locked.fill_(True)

    def assert_parameters_fixed(self):
        assert not bool(((self.parameter_action!=self.locked_parameters)&self.parameters_locked[:, None]).any())

    def control_tick(self, held):
        _, vel, gyro, linear, gravity, height, _ = self.state()
        args = dict(ticks=self.ticks, apex=self.task.height_score.apex, terminal=self.terminal_mask(),
            launch_height=.18+.03*held[:, 6], touchdown=self.task.touchdown_time, leg_height=height,
            incident_vz=self.task.landing_reward.pre_touch_vz, gravity=gravity, gyro=gyro,
            linear=linear, wheel_speed=vel[:, 4:], mass=self.mass)
        def request(raw):
            c = tick(raw, self.control_state, **args)
            motor_v = reference_motor_velocity(c['height'], c['velocity'])
            new = payload(torch.tanh(c['correction']), c['height'], motor_v, c['force'],
                          c['kp'], c['kd'], c['enabled'])
            return c, torch.where(c['enabled'][:, None], new, held)
        c, requested = request(self.parameter_action)
        if self.proof:
            _, alternate = request(self.parameter_action+1.1)
            before = ~self.task.height_score.apex
            assert torch.equal(requested[before], alternate[before])
            assert torch.equal(requested[before, :12], held[before, :12])
            self.proof_samples += int(before.sum())
            self.proof_mixed_ticks += int(bool(before.any()) and not bool(before.all()))
            self.shadow_fifo.delay_steps.copy_(self.fifo.delay_steps)
            self.expected_prefix = self.shadow_fifo.push(held[:, :12]).clone()
            self.prefix_mask = before.clone()
            if not bool(before.any()):
                self.proof = False
        self.control_state = c['state']
        self.gate_tick.copy_(self.control_state['gate_tick'])
        self.control_gate.copy_(c['enabled'])
        self.effective_action.copy_(torch.where(c['enabled'][:, None], self.parameter_action,
                                                torch.zeros_like(self.parameter_action)))
        state = torch.stack((c['height']-.18, c['velocity'], c['force'], torch.ones_like(c['height'])), 1)
        self.curve_state.copy_(torch.where(c['enabled'][:, None], state, self.curve_state))
        return requested

    def check_arrived(self, arrived):
        if self.proof:
            assert torch.equal(arrived[self.prefix_mask, :12], self.expected_prefix[self.prefix_mask])
            assert not bool((arrived[self.prefix_mask, 14]>.5).any())

    def step(self, standing_actions, *, auto_reset=False):
        self.assert_launch_fixed()
        self.assert_parameters_fixed()
        active_launch = (self.ticks*.0025>=.6) & ~self.task.height_score.apex
        self.curve_state.copy_(torch.where(active_launch[:, None], launch_reference(self.plan, self.ticks*.0025), self.curve_state))
        action = torch.where((self.ticks*.0025<.6)[:, None], standing_actions[:, :6], torch.zeros_like(standing_actions[:, :6]))
        out = (self._step_fast if self.fast_backend else self._step_native)(action, auto_reset=auto_reset)
        self.previous_policy_action.copy_(self.effective_action)
        self.assert_launch_fixed()
        self.assert_parameters_fixed()
        return out

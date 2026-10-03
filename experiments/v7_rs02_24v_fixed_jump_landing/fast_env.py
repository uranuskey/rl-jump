"""Same frozen physics and guards, with batched host synchronization."""
import torch
import warp as wp
from controlled_env import (math, actuator, payload, servo, wrench,
                            PEAK_EXPOSURE_THRESHOLD_NM, REASONS, compose)
from landing import LandingEnv
from fast_checks import external_fault, post_step_faults


class FastLandingEnv(LandingEnv):
    def assert_launch_fixed(self):
        if bool(((self.plan!=self.locked_plan)&self.plan_locked[:, None]).any()):
            raise RuntimeError('Frozen jump plan changed')

    def step(self, standing_actions, *, policy_action=None, auto_reset=False):
        commands = self.prepare_commands(standing_actions, policy_action)
        for name, value in commands.items():
            if name!='actions':
                getattr(self, name).copy_(value)
        if bool(((self.effective_action!=0)&~self.task.height_score.apex[:, None]).any()):
            raise RuntimeError('Landing controller acted before COM apex')
        out = self._step_controlled(commands['actions'], auto_reset=auto_reset)
        self.previous_policy_action.copy_(self.effective_action)
        self.assert_launch_fixed()
        return out

    def sensors(self, motor=None):
        q, v, gyro, linear, gravity, height, tilt = self.state()
        zeros4 = torch.zeros((self.n, 4), device=self.device)
        zeros2 = zeros4[:, :2]
        return dict(wheel_force_n=self.support.clone(), wheel_clearance_m=self.clearance.clone(), com_z_m=self.com[:, 2].clone(),
            com_vz_mps=self.com_v[:, 2].clone(), base_xy_m=self.q[:, :2].clone(), body_vxy_mps=linear[:, :2].clone(),
            body_vz_mps=linear[:, 2].clone(), tilt_rad=tilt, gyro_norm_rad_s=torch.linalg.vector_norm(gyro, dim=1),
            leg_height_m=height, motor_torque_nm=zeros4 if motor is None else motor['leg_motor_actual_nm'].clone(),
            motor_speed_rad_s=torch.stack((v[:, :4].reshape(self.n, 2, 2)[:, :, 0], v[:, :4].reshape(self.n, 2, 2).sum(-1)), -1).reshape(self.n, 4),
            wheel_torque_nm=zeros2 if motor is None else motor['wheel_actual_nm'].clone(), wheel_speed_rad_s=v[:, 4:].clone(),
            mimic_q_error_rad=(self.q[:, self.aq]-7.75*q[:, :4]).abs().max(1).values,
            mimic_v_error_rad_s=(self.v[:, self.aux]-7.75*v[:, :4]).abs().max(1).values,
            nonwheel_force_n=self.bad[:, 0].clone(), self_contact=self.bad[:, 1] > 0,
            saturated=torch.zeros_like(self.paused) if motor is None else (motor['leg_motor_torque_saturated'].any(1) | motor['wheel_torque_saturated'].any(1)),
            backend_ok=torch.ones_like(self.paused), overflow=wp.to_torch(self.gd.overflow).any().expand_as(self.paused),
            requested=(self.ticks >= self.cfg.ticks(self.cfg.request_s)) & self.request_enabled,
            motor_request_nm=zeros4 if motor is None else motor['leg_motor_request_nm'].clone(),
            motor_speed_pre_rad_s=zeros4 if motor is None else motor['leg_motor_speed_rad_s'].clone(),
            motor_envelope_nm=zeros4 if motor is None else motor['leg_motor_envelope_nm'].clone(),
            motor_peak_exposure_s=self.peak_exposure_s.clone())

    def _step_controlled(self, actions, *, auto_reset=True):
        if actions.shape != (self.n, 6) or not bool(torch.isfinite(actions).all()):
            self.abort('invalid_action')
        req = torch.tanh(actions)
        requested_height = self.command()[:, 3]*.25
        requested_velocity = self.push_motor_velocity(requested_height)
        requested_thrust = self.push_force()
        requested_payload = payload(req, requested_height, requested_velocity, requested_thrust)
        self.previous.copy_(req)
        # A deterministic first-life evaluation may call step without autoreset.
        # Failed/finished rows remain frozen across policy ticks.
        self.paused.copy_(self.terminal_mask() | (self.ticks >= self.cfg.ticks(self.cfg.episode_s)))
        reward = torch.zeros(self.n, device=self.device)
        elapsed = reward.clone()
        traces = []
        for _ in range(8):
            active = ~self.paused
            arrived = self.fifo.push(requested_payload)
            command = servo(arrived, self.q[:, self.qids], self.v[:, self.main], self.R, self.support, voltage=self.voltage)
            motor = actuator.map_and_limit(command['joint_request'], self.v[:, self.main[:4]], command['wheel_request'],
                self.v[:, self.main[4:]], leg_torque_limit_nm=self.cfg.leg_motor_torque_limit_nm,
                leg_speed_limit_rad_s=self.cfg.leg_motor_speed_limit_rad_s,
                wheel_torque_limit_nm=self.cfg.wheel_torque_limit_nm,wheel_speed_limit_rad_s=self.cfg.wheel_speed_limit_rad_s)
            effort = torch.cat((motor['leg_joint_actual_nm'], motor['wheel_actual_nm']), 1)
            self.force.zero_()
            self.force[:, self.main] = effort
            u=((self.ticks.to(self.R.dtype)-200)/40).clamp(0,1)
            effective_strength=self.assist_strength*(u*u*(3-2*u))
            assist,omega_pre=wrench(self.R,self.v[:,3:6],effective_strength,active)
            self.external.zero_()
            self.external[:,self.base]=assist
            with wp.ScopedStream(self.stream):
                wp.capture_launch(self.physics_graph)
            omega_post=(self.R@self.v[:,3:6,None]).squeeze(-1)
            assist_power=(assist[:,3:]*(omega_pre+omega_post)*.5).sum(1)
            self.assist_work[:,0]+=assist_power*self.cfg.physics_dt_s
            self.assist_work[:,1]+=assist_power.clamp_min(0)*self.cfg.physics_dt_s
            self.assist_peak=torch.maximum(self.assist_peak,torch.linalg.vector_norm(assist[:,3:],dim=1))
            if bool(external_fault(self.external, self.base, assist)):
                self.abort('external_assistance_force_path')
            self.ticks += active.long()
            self.peak_exposure_s += active[:, None]*(motor['leg_motor_actual_nm'].abs()>PEAK_EXPOSURE_THRESHOLD_NM)*self.cfg.physics_dt_s
            x = self.sensors(motor)
            faults = post_step_faults(x['overflow'], self.q, self.v, self.acc, active,
                self.force, self.main, effort, self.aux, wp.to_torch(self.gd.qfrc_actuator))
            # All guards still run at every physics tick; only one host read on success.
            if bool(faults.any()):
                numeric, force_path = faults.cpu().tolist()
                if numeric:
                    self.abort('numeric_or_illegal_contact_or_overflow', x)
                if force_path:
                    self.abort('backend_force_path')
            phase_before = self.task.phase.clone()
            tick_reward = self.task.update(x, active)
            self.capture_terminal(x, active & self.terminal_mask(), phase_before)
            reward += tick_reward
            elapsed += active*self.cfg.physics_dt_s
            if self.mode == 'admission':
                # Scripted drops intentionally bypass the task's uncommanded-flight
                # failure, but must NEVER bypass the physical protection envelope.
                # Stop the case at the first limit, not later at a body impact.
                self.admission_saturation_ticks = torch.where(active, torch.where(x['saturated'], self.admission_saturation_ticks+1, 0), self.admission_saturation_ticks)
                guards = (
                    ((x['motor_speed_rad_s'].abs().max(1).values > self.cfg.leg_motor_speed_limit_rad_s)
                     | (x['wheel_speed_rad_s'].abs().max(1).values > self.cfg.wheel_speed_limit_rad_s), 'motor_speed'),
                    (self.admission_saturation_ticks >= self.cfg.ticks(self.cfg.saturation_stop_s), 'sustained_saturation'),
                    (x['tilt_rad'] > math.radians(self.cfg.maximum_tilt_deg), 'tilt'),
                    (torch.linalg.vector_norm(x['base_xy_m']-self.task.start_xy, dim=1) > self.cfg.maximum_drift_m, 'horizontal_drift'),
                    ((x['leg_height_m'].min(1).values < self.cfg.leg_height_low_m)
                     | (x['leg_height_m'].max(1).values > self.cfg.leg_height_high_m), 'leg_workspace'))
                for mask, reason in guards:
                    stopped = mask & active & (self.admission_reason == 0)
                    self.admission_reason[stopped] = REASONS.index(reason)
                self.paused |= self.admission_reason != 0
            if self.record:
                traces.append(dict(sample={k: v.clone() for k, v in x.items()}, active=active.clone(), ticks=self.ticks.clone(),
                    native_time=self.time.clone(), q=self.q.clone(), v=self.v.clone(), phase=self.task.phase.clone(),
                    assist_wrench=assist.clone(),assist_strength=self.assist_strength.clone(),
                    assist_effective_strength=effective_strength.clone(),
                    assist_world_omega_pre=omega_pre.clone(),assist_world_omega_post=omega_post.clone(),
                    assist_power_w=assist_power.clone(),assist_work_j=self.assist_work.clone(),
                    height_only_reward=tick_reward.clone(),
                    reward_flight_height=self.task.height_score.peak.clone(),
                    reward_release_z=self.task.height_score.release_z.clone(),
                    reward_release_time=self.task.height_score.release_time.clone(),
                    reason=self.task.reason.clone(), motor_target=command['motor_target_rad'].clone(),
                    motor_position_pre=command['motor_position_pre_rad'].clone(),
                    arrived_motor_velocity=command['motor_velocity_reference_rad_s'].clone(),
                    motor_base_request=command['motor_base_request_nm'].clone(),
                    motor_velocity_feedforward=command['motor_velocity_feedforward_nm'].clone(),
                    requested_motor_velocity=requested_velocity.clone(),
                    requested_thrust_force=requested_thrust.clone(),
                    requested_height=requested_height.clone(),
                    curve_state=self.curve_state.clone(),curve_action=self.curve_action.clone(),
                    thrust_requested_force=command['thrust_requested_force_n'].clone(),
                    thrust_motor_unlimited=command['thrust_motor_unlimited_nm'].clone(),
                    thrust_motor_applied=command['thrust_motor_applied_nm'].clone(),
                    thrust_allocation_fraction=command['thrust_allocation_fraction'].clone(),
                    thrust_pair_contact=command['thrust_pair_contact'].clone(),
                    motor_request_before_thrust=command['motor_request_before_thrust_nm'].clone(),
                    base_rotation_pre=command['base_rotation_pre'].clone(),
                    arrived_reference_height=command['reference_height_m'].clone(),
                    arrived_corrections=arrived[:, :6].clone(), episode_id=self.episode_id.clone()))
            if self.mode == 'policy':
                self.paused |= self.terminal_mask()
            self.sq.copy_(self.q); self.sv.copy_(self.v); self.sw.copy_(self.warm); self.st.copy_(self.time)
        done = self.paused.clone()
        timeout = (self.ticks >= self.cfg.ticks(self.cfg.episode_s)) & ~done
        self.raw_history[:, :-1] = self.raw_history[:, 1:].clone()
        self.raw_history[:, -1] = self.raw()
        self.command_history[:, :-1] = self.command_history[:, 1:].clone()
        self.command_history[:, -1] = self.command()
        terminal_actor, terminal_critic = compose(self.raw_history, self.command_history, self.task.privileged(x),self.curve_state,self.assist_strength)
        self.last = dict(sample=x, elapsed=elapsed, traces=traces, phase=self.task.phase.clone(), reason=self.task.reason.clone(),
                         episode_ticks=self.ticks.clone(),
                         terminated=done, time_outs=timeout, terminal_actor=terminal_actor, terminal_critic=terminal_critic,
                         admission_reason=self.admission_reason.clone())
        self.obs, self.critic_obs = terminal_actor, terminal_critic
        if auto_reset and bool((done | timeout).any()):
            self.reset(done | timeout)
        return self.obs, self.critic_obs, reward, done | timeout, self.last

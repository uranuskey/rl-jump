"""Original full physics with an explicit,uncalibrated RS02-24V motor envelope."""
import bindings
import importlib.util
import json
import math
from pathlib import Path
import numpy as np
import torch
import mujoco
import mujoco_warp as mjw
import warp as wp
from native_kernels import restore_paused, reset_pose, contact_metrics, wheel_bottom
from settings import JumpConfig
from height_contract import compose
from velocity_contract import servo,payload
from contract import leg_reference
from curriculum_task import CurriculumTask, FAILED, REASONS, STAGES

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL_PATH = ROOT/'experiments/v7_mujoco_training_gate/v7_full_collision.xml'
import rs02_actuator as actuator
from motor_model import PEAK_EXPOSURE_THRESHOLD_NM
MAIN = actuator.POLICY_NAMES
AUX = [s+'_'+j+'_rotor_proxy_joint' for s in ('left', 'right') for j in ('hip', 'knee')]


class HeightEnv:
    def __init__(self, n=4, *, config=None, mode='policy', record=False, abort_dir=None, fixed_delay=None, stage='balance'):
        if mode not in ('policy', 'admission'):
            raise ValueError('Unknown execution mode')
        self.cfg = config or JumpConfig()
        if stage not in STAGES:
            raise ValueError(stage)
        self.stage = stage
        self.n, self.mode, self.record = n, mode, record
        self.device = 'cuda:0'
        self.abort_dir = Path(abort_dir or HERE/'native_aborts')
        self.fixed_delay = fixed_delay
        wp.init()
        wp.set_device(self.device)
        self.torch_stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.torch_stream)
        self.stream = wp.stream_from_torch(self.torch_stream)
        self.m = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        if abs(self.m.opt.timestep-self.cfg.physics_dt_s) > 1e-12:
            raise RuntimeError('Frozen timestep mismatch')
        pose = json.loads((ROOT/'experiments/v7_exact_rotor_mjcf/targets.json').read_text())['poses']['180mm']
        cpu = mujoco.MjData(self.m)
        cpu.qpos[:7] = pose['root_pos_quat_wxyz']
        for name, value in pose['q_by_name'].items():
            cpu.qpos[self.m.joint(name).qposadr[0]] = value
        mujoco.mj_forward(self.m, cpu)
        self.main_np = np.array([self.m.joint(k).dofadr[0] for k in MAIN])
        self.qids_np = np.array([self.m.joint(k).qposadr[0] for k in MAIN])
        self.aux_np = np.array([self.m.joint(k).dofadr[0] for k in AUX])
        self.aq_np = np.array([self.m.joint(k).qposadr[0] for k in AUX])
        self.main, self.qids, self.aux, self.aq = [torch.tensor(x, device=self.device) for x in (self.main_np, self.qids_np, self.aux_np, self.aq_np)]
        self.base = self.m.body('base_link').id
        self.wheels = [self.m.body(s+'_wheel').id for s in ('left', 'right')]
        self.nomq = torch.tensor(cpu.qpos, dtype=torch.float32, device=self.device)
        self.legnom = self.nomq[self.qids[:4]].clone()
        self.poses = self.nomq[None, :].repeat(n, 1)
        self.ipos = torch.tensor(self.m.body_ipos[self.base], dtype=torch.float32, device=self.device)
        with wp.ScopedStream(self.stream):
            self.gm = mjw.put_model(self.m)
            self.gd = mjw.put_data(self.m, cpu, nworld=n, nconmax=128, njmax=1024, nccdmax=128)
        for name, field in [('q', 'qpos'), ('v', 'qvel'), ('warm', 'qacc_warmstart'), ('time', 'time'),
                            ('force', 'qfrc_applied'), ('acc', 'qacc')]:
            setattr(self, name, wp.to_torch(getattr(self.gd, field)))
        self.R = wp.to_torch(self.gd.xmat)[:, self.base]
        self.com = wp.to_torch(self.gd.subtree_com)[:, self.base]
        self.com_v = wp.to_torch(self.gd.subtree_linvel)[:, self.base]
        self.paused = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.resetmask = self.paused.clone()
        self.sq, self.sv, self.sw, self.st = self.q.clone(), self.v.clone(), self.warm.clone(), self.time.clone()
        self.forces_wp = wp.zeros(self.gd.naconmax, dtype=wp.spatial_vector, device=self.device)
        self.contactids = wp.array(np.arange(self.gd.naconmax, dtype=np.int32), dtype=int, device=self.device)
        self.support_wp = wp.zeros((n, 2), dtype=float, device=self.device)
        self.bad_wp = wp.zeros((n, 2), dtype=float, device=self.device)
        self.clearance_wp = wp.zeros((n, 2), dtype=float, device=self.device)
        self.support, self.bad, self.clearance = [wp.to_torch(a) for a in (self.support_wp, self.bad_wp, self.clearance_wp)]
        vertices, geomids, wheelids = [], [], []
        self.wheel_geoms = []
        for wheel, body in enumerate(self.wheels):
            ids = np.flatnonzero(self.m.geom_bodyid == body)
            self.wheel_geoms.append(ids.tolist())
            for gid in ids:
                if self.m.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
                    raise RuntimeError('Wheel collider is no longer a mesh')
                mesh = self.m.geom_dataid[gid]
                adr, count = self.m.mesh_vertadr[mesh], self.m.mesh_vertnum[mesh]
                vertices.extend(self.m.mesh_vert[adr:adr+count].tolist())
                geomids.extend([int(gid)]*count)
                wheelids.extend([wheel]*count)
        self.vertices = wp.array(np.asarray(vertices, np.float32), dtype=wp.vec3, device=self.device)
        self.geomids = wp.array(np.asarray(geomids, np.int32), dtype=int, device=self.device)
        self.wheelids = wp.array(np.asarray(wheelids, np.int32), dtype=int, device=self.device)
        self.vertex_count = len(vertices)
        self.fifo = actuator.PhysicalStepFIFO(n, 11, self.device)
        self.rng = torch.Generator(device=self.device).manual_seed(7)
        self.previous = torch.zeros((n, 6), device=self.device)
        self.peak_exposure_s = torch.zeros((n, 4), device=self.device)
        self.raw_history = torch.zeros((n, 4, 35), device=self.device)
        self.command_history = torch.zeros((n, 4, 4), device=self.device)
        self.task = CurriculumTask(n, self.device, self.cfg, stage=stage)
        self.episode_id = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.ticks = torch.zeros(n, dtype=torch.long, device=self.device)
        self.last = {}
        self.admission_reason = torch.zeros(n, dtype=torch.long, device=self.device)
        self.admission_saturation_ticks = torch.zeros_like(self.admission_reason)
        self.reset_native_graph = self.physics_graph = None
        self.request_enabled = mode == 'policy' and stage == 'jump'
        with wp.ScopedStream(self.stream):
            self._physics()  # compile; rows are reset before any admitted sampling
            with wp.ScopedCapture() as capture:
                self._physics()
            self.physics_graph = capture.graph
            self._reset_native()
            with wp.ScopedCapture() as capture:
                self._reset_native()
            self.reset_native_graph = capture.graph
        self.reset(torch.ones_like(self.paused))

    def _sensors(self):
        # Full forward at q(t+dt),v(t+dt): contacts, constraints, forces, geometry
        # and COM all describe the same state. Extra solve is an explicit new
        # backend contract and is tested in J1; it is not the old kinematics-only refresh.
        mjw.forward(self.gm, self.gd)
        mjw.subtree_vel(self.gm, self.gd)
        self.support_wp.zero_()
        self.bad_wp.zero_()
        mjw.contact_force(self.gm, self.gd, self.contactids, False, self.forces_wp)
        wp.launch(contact_metrics, dim=self.gd.naconmax, inputs=[self.gd.nacon, self.gd.contact.geom, self.gd.contact.worldid,
                  self.gm.geom_bodyid, self.gd.contact.frame, self.gd.contact.dist, self.forces_wp, *self.wheels,
                  self.support_wp, self.bad_wp])
        self.clearance_wp.fill_(float('inf'))
        wp.launch(wheel_bottom, dim=(self.n, self.vertex_count), inputs=[self.vertices, self.geomids, self.wheelids,
                  self.gd.geom_xpos, self.gd.geom_xmat, self.clearance_wp])

    def _physics(self):
        mjw.step(self.gm, self.gd)
        wp.launch(restore_paused, dim=self.n, inputs=[wp.from_torch(self.paused), wp.from_torch(self.sq), wp.from_torch(self.sv),
                  wp.from_torch(self.sw), wp.from_torch(self.st), self.gd.qpos, self.gd.qvel, self.gd.qacc_warmstart, self.gd.time])
        self._sensors()

    def _reset_native(self):
        mjw.reset_data(self.gm, self.gd, wp.from_torch(self.resetmask))
        wp.launch(reset_pose, dim=(self.n, self.m.nq), inputs=[wp.from_torch(self.resetmask), wp.from_torch(self.poses), self.gd.qpos])
        # Do not recompute constraints/forces for untouched worlds during a selective
        # reset: preserve their warm-start and contact state until the next step.
        mjw.kinematics(self.gm, self.gd)
        mjw.com_pos(self.gm, self.gd)
        mjw.com_vel(self.gm, self.gd)
        mjw.subtree_vel(self.gm, self.gd)

    def state(self):
        q, v = self.q[:, self.qids], self.v[:, self.main]
        gyro = self.v[:, 3:6]
        linear = (self.R.transpose(1, 2) @ self.v[:, :3, None]).squeeze(-1)+torch.cross(gyro, self.ipos.expand_as(gyro), dim=1)
        leg = q[:, :4].reshape(self.n, 2, 2)
        h = .105*torch.cos(leg[:, :, 0])+.145*torch.cos(leg.sum(-1))
        return q, v, gyro, linear, -self.R[:, 2, :], h, torch.acos(self.R[:, 2, 2].clamp(-1, 1))

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
            backend_ok=torch.ones_like(self.paused), overflow=torch.full_like(self.paused, bool(wp.to_torch(self.gd.overflow).any())),
            requested=(self.ticks >= self.cfg.ticks(self.cfg.request_s)) & self.request_enabled,
            motor_request_nm=zeros4 if motor is None else motor['leg_motor_request_nm'].clone(),
            motor_speed_pre_rad_s=zeros4 if motor is None else motor['leg_motor_speed_rad_s'].clone(),
            motor_envelope_nm=zeros4 if motor is None else motor['leg_motor_envelope_nm'].clone(),
            motor_peak_exposure_s=self.peak_exposure_s.clone())

    def raw(self):
        q, v, gyro, linear, g, height, tilt = self.state()
        # Legacy yaw slots are constants and excluded from actor. No true height/contact enters raw35.
        slots = torch.tensor([0., 0., 1., 1.], device=self.device).expand(self.n, -1)
        cmd = torch.zeros((self.n, 2), device=self.device)
        odom = torch.cat((linear, torch.zeros((self.n, 1), device=self.device), torch.ones((self.n, 1), device=self.device),
                          torch.zeros((self.n, 1), device=self.device), torch.ones((self.n, 1), device=self.device)), 1)
        return torch.cat((g, .2*gyro, q[:, :4]-self.legnom, .1*v[:, :4], slots, .05*v[:, 4:], self.previous, cmd, odom), 1)

    def command(self):
        request = (self.ticks >= self.cfg.ticks(self.cfg.request_s)) & self.request_enabled
        age = ((self.ticks-self.cfg.ticks(self.cfg.request_s)).clamp_min(0)*self.cfg.physics_dt_s/self.cfg.episode_s).clamp(max=1)
        height = leg_reference(self.ticks*self.cfg.physics_dt_s, self.stage)/.25
        return torch.stack((torch.full_like(age, self.cfg.target_height_m/.05), request.float(), age, height), 1)

    def set_stage(self, stage):
        if stage not in STAGES:
            raise ValueError(stage)
        self.stage = self.task.stage = stage
        self.request_enabled = self.mode == 'policy' and stage == 'jump'
        self.reset(torch.ones_like(self.paused))

    def push_motor_velocity(self,requested_height):
        return torch.zeros((self.n,4),device=self.device,dtype=requested_height.dtype)

    def reset(self, mask, *, fixed_delay=None, poses=None, velocities=None):
        if poses is not None:
            if poses.shape != self.poses.shape:
                raise ValueError('Full reset pose tensor required')
            self.poses[mask] = poses[mask]
        else:
            self.poses[mask] = self.nomq
        self.resetmask.copy_(mask)
        with wp.ScopedStream(self.stream):
            wp.capture_launch(self.reset_native_graph)
            if velocities is not None:
                self.v[mask] = velocities[mask]
                mjw.com_vel(self.gm, self.gd)
                mjw.subtree_vel(self.gm, self.gd)
        ids = mask.nonzero().flatten()
        self.fifo.reset(ids)
        delay = self.fixed_delay if fixed_delay is None else fixed_delay
        ds = torch.randint(0, 9, (len(ids),), generator=self.rng, device=self.device) if delay is None else torch.full((len(ids),), delay, device=self.device, dtype=torch.long)
        self.fifo.set_delays(ids, ds)
        self.previous[mask] = 0
        self.peak_exposure_s[mask] = 0
        self.ticks[mask] = 0
        self.episode_id[mask] += 1
        self.paused[mask] = False
        self.admission_reason[mask] = 0
        self.admission_saturation_ticks[mask] = 0
        self.support[mask] = 0
        self.bad[mask] = 0
        # Recompute mesh clearance after reset; no contact-derived truth reaches actor.
        with wp.ScopedStream(self.stream):
            self.clearance_wp.fill_(float('inf'))
            wp.launch(wheel_bottom, dim=(self.n, self.vertex_count), inputs=[self.vertices, self.geomids, self.wheelids,
                      self.gd.geom_xpos, self.gd.geom_xmat, self.clearance_wp])
        self.task.reset(mask, self.q[:, :2], self.state()[5])
        self.raw_history[mask] = self.raw()[mask, None, :]
        self.command_history[mask] = self.command()[mask, None, :]
        self.sq.copy_(self.q); self.sv.copy_(self.v); self.sw.copy_(self.warm); self.st.copy_(self.time)
        self.obs, self.critic_obs = compose(self.raw_history, self.command_history, self.task.privileged(self.sensors()))

    def abort(self, reason, sample=None):
        self.abort_dir.mkdir(parents=True, exist_ok=True)
        torch.save(dict(reason=reason, q=self.q.clone(), v=self.v.clone(), acc=self.acc.clone(), sample=sample,
                        ticks=self.ticks.clone(), force=self.force.clone(), time=self.time.clone()), self.abort_dir/'failure.pt')
        raise RuntimeError(reason)

    def step(self, actions, *, auto_reset=True):
        if actions.shape != (self.n, 6) or not bool(torch.isfinite(actions).all()):
            self.abort('invalid_action')
        req = torch.tanh(actions)
        requested_height = self.command()[:, 3]*.25
        requested_velocity = self.push_motor_velocity(requested_height)
        requested_payload = payload(req, requested_height, requested_velocity)
        self.previous.copy_(req)
        # A deterministic first-life evaluation may call step without autoreset.
        # Failed/finished rows remain frozen across policy ticks.
        self.paused.copy_((self.task.phase == FAILED) | (self.ticks >= self.cfg.ticks(self.cfg.episode_s)))
        reward = torch.zeros(self.n, device=self.device)
        elapsed = reward.clone()
        traces = []
        for _ in range(8):
            active = ~self.paused
            arrived = self.fifo.push(requested_payload)
            command = servo(arrived, self.q[:, self.qids], self.v[:, self.main])
            motor = actuator.map_and_limit(command['joint_request'], self.v[:, self.main[:4]], command['wheel_request'],
                self.v[:, self.main[4:]], leg_torque_limit_nm=self.cfg.leg_motor_torque_limit_nm,
                leg_speed_limit_rad_s=self.cfg.leg_motor_speed_limit_rad_s,
                wheel_torque_limit_nm=self.cfg.wheel_torque_limit_nm,wheel_speed_limit_rad_s=self.cfg.wheel_speed_limit_rad_s)
            effort = torch.cat((motor['leg_joint_actual_nm'], motor['wheel_actual_nm']), 1)
            self.force.zero_()
            self.force[:, self.main] = effort
            with wp.ScopedStream(self.stream):
                wp.capture_launch(self.physics_graph)
            self.ticks += active.long()
            self.peak_exposure_s += active[:, None]*(motor['leg_motor_actual_nm'].abs()>PEAK_EXPOSURE_THRESHOLD_NM)*self.cfg.physics_dt_s
            x = self.sensors(motor)
            bad = ((x['mimic_q_error_rad'] > self.cfg.mimic_q_limit_rad) | (x['mimic_v_error_rad_s'] > self.cfg.mimic_v_limit_rad_s)
                   | (x['nonwheel_force_n'] > self.cfg.nonwheel_force_limit_n) | x['self_contact'] | x['overflow']
                   | ~torch.isfinite(self.q).all(1) | ~torch.isfinite(self.v).all(1) | ~torch.isfinite(self.acc).all(1)) & active
            if bool(bad.any()):
                self.abort('numeric_or_illegal_contact_or_overflow', x)
            if bool((self.force[:, self.main]-effort).abs().max() > 1e-6) or bool(self.force[:, self.aux].abs().max() > 0) or bool(wp.to_torch(self.gd.qfrc_actuator).abs().max() > 0):
                self.abort('backend_force_path')
            reward += self.task.update(x, active)
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
                    reason=self.task.reason.clone(), motor_target=command['motor_target_rad'].clone(),
                    motor_position_pre=command['motor_position_pre_rad'].clone(),
                    arrived_motor_velocity=command['motor_velocity_reference_rad_s'].clone(),
                    motor_base_request=command['motor_base_request_nm'].clone(),
                    motor_velocity_feedforward=command['motor_velocity_feedforward_nm'].clone(),
                    requested_motor_velocity=requested_velocity.clone(),
                    arrived_reference_height=command['reference_height_m'].clone(),
                    arrived_corrections=arrived[:, :6].clone(), episode_id=self.episode_id.clone()))
            if self.mode == 'policy':
                self.paused |= self.task.phase == FAILED
            self.sq.copy_(self.q); self.sv.copy_(self.v); self.sw.copy_(self.warm); self.st.copy_(self.time)
        done = self.paused.clone()
        timeout = (self.ticks >= self.cfg.ticks(self.cfg.episode_s)) & ~done
        self.raw_history[:, :-1] = self.raw_history[:, 1:].clone()
        self.raw_history[:, -1] = self.raw()
        self.command_history[:, :-1] = self.command_history[:, 1:].clone()
        self.command_history[:, -1] = self.command()
        terminal_actor, terminal_critic = compose(self.raw_history, self.command_history, self.task.privileged(x))
        self.last = dict(sample=x, elapsed=elapsed, traces=traces, phase=self.task.phase.clone(), reason=self.task.reason.clone(),
                         episode_ticks=self.ticks.clone(),
                         terminated=done, time_outs=timeout, terminal_actor=terminal_actor, terminal_critic=terminal_critic,
                         admission_reason=self.admission_reason.clone())
        self.obs, self.critic_obs = terminal_actor, terminal_critic
        if auto_reset and bool((done | timeout).any()):
            self.reset(done | timeout)
        return self.obs, self.critic_obs, reward, done | timeout, self.last

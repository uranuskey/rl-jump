"""Configure only solver precision or the four rotor coupling time constants."""
import fix_runtime
import hashlib
import numpy as np
import mujoco
import torch
import warp as wp
from probe_env import WithdrawalEnv
from fix_contract import checked_variant, checked_levels


def structural_hash(m):
    h = hashlib.sha256()
    for key in ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'jnt_pos', 'jnt_axis',
                'dof_armature', 'dof_damping', 'geom_friction', 'geom_solref', 'geom_solimp',
                'eq_type', 'eq_obj1id', 'eq_obj2id', 'eq_data', 'eq_solimp'):
        h.update(key.encode('ascii')); h.update(getattr(m, key).tobytes())
    return h.hexdigest()


class ConstraintEnv(WithdrawalEnv):
    def __init__(self, n, profiles, *, before, after, variant, **kwargs):
        checked_levels(before, after)
        self.variant_name = variant
        self.variant = checked_variant(variant)
        self.physics_receipt = None
        self.capture_prefix = False
        self.prefix_rows = []
        # Real sampling always calls reset_cases after setting these levels.
        super().__init__(n, profiles, before=.625, after=.625, **kwargs)
        self.before, self.after, self.assist_target = before, after, after
        self.assist_strength.fill_(before)

    def _physics(self):
        if self.physics_receipt is None:
            self.configure_constraint()
        return super()._physics()

    def configure_constraint(self):
        names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_EQUALITY, i) for i in range(self.m.neq)]
        expected = ['eq_'+s+'_'+j+'_rotor_proxy_joint' for s in ('left', 'right') for j in ('hip', 'knee')]
        assert names == expected
        assert np.array_equal(self.m.eq_solref, np.tile([.020, 1.], (4, 1)))
        assert np.all(self.m.eq_data[:, 1] == 7.75)
        assert self.m.opt.timestep == .0025 and self.m.opt.disableflags == 0
        structural = structural_hash(self.m)
        original = dict(eq_solref=self.m.eq_solref.tolist(),
            **{k:getattr(self.m.opt, k) for k in ('iterations', 'tolerance', 'ls_iterations', 'ls_tolerance')})
        for key in ('iterations', 'ls_iterations'):
            setattr(self.m.opt, key, self.variant[key])
            setattr(self.gm.opt, key, self.variant[key])
        for key in ('tolerance', 'ls_tolerance'):
            setattr(self.m.opt, key, self.variant[key])
            wp.to_torch(getattr(self.gm.opt, key)).fill_(self.variant[key])
        self.m.eq_solref[:, 0] = self.variant['rotor_timeconst']
        wp.to_torch(self.gm.eq_solref)[..., 0] = self.variant['rotor_timeconst']
        gpu = wp.to_torch(self.gm.eq_solref).cpu().numpy()
        assert np.allclose(gpu, self.m.eq_solref, atol=1e-9, rtol=1e-7)
        assert structural_hash(self.m) == structural
        assert self.cfg.mimic_q_limit_rad == .001 and self.cfg.mimic_v_limit_rad_s == .01
        self.physics_receipt = dict(variant=self.variant_name, applied=self.variant,
            original=original, cpu_eq_solref=self.m.eq_solref.tolist(), gpu_eq_solref=gpu.tolist(),
            structure_sha256=structural, structure_unchanged=True, timestep_s=.0025,
            refsafe_enabled=True, q_gate_rad=.001, v_gate_rad_s=.01,
            only_four_rotor_solref_and_solver_options_changed=True,
            state_projection=False, motor_curve_changed=False, contact_parameters_changed=False)
        self.step_solver_niter = wp.zeros(self.n, dtype=wp.int32, device=self.device)

    def _sensors(self):
        # Preserve the integration solve's iteration count before the existing
        # post-step forward solve overwrites Data.solver_niter.
        wp.copy(self.step_solver_niter, self.gd.solver_niter)
        return super()._sensors()

    def sensors(self, motor=None):
        x = super().sensors(motor)
        if self.physics_receipt is not None:
            x['rotor_eq_timeconst_s'] = wp.to_torch(self.gm.eq_solref)[..., 0].expand(self.n, -1).clone()
            x['solver_tolerance'] = wp.to_torch(self.gm.opt.tolerance).expand(self.n).clone()
        if self.capture_prefix:
            self.prefix_calls += 1
            if 360 <= self.prefix_calls <= 480:
                self.prefix_rows.append(dict(ticks=self.ticks.clone(), active=(~self.paused).clone(),
                    joint_q=self.q[:, self.qids[:4]].clone(), rotor_q=self.q[:, self.aq].clone(),
                    joint_v=self.v[:, self.main[:4]].clone(), rotor_v=self.v[:, self.aux].clone(),
                    joint_acc=self.acc[:, self.main[:4]].clone(), rotor_acc=self.acc[:, self.aux].clone(),
                    solver_niter=wp.to_torch(self.step_solver_niter).clone(),
                    post_forward_solver_niter=wp.to_torch(self.gd.solver_niter).clone(),
                    constraint_force=wp.to_torch(self.gd.qfrc_constraint)[:, self.aux].clone(),
                    reported_q=x['mimic_q_error_rad'].clone(), reported_v=x['mimic_v_error_rad_s'].clone(),
                    com_vz=x['com_vz_mps'].clone(), tilt=x['tilt_rad'].clone()))
        return x

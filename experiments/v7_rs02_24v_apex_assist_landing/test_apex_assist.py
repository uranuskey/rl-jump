"""Reject hidden pre-apex reductions and noncausal or abrupt assistance ramps."""
import unittest
import numpy as np
from apex_audit import audit_arrays


class ScheduleAuditTests(unittest.TestCase):
    def trace(self):
        levels = np.array([.625, .625, .6125, .6])[:, None]
        omega = np.zeros((4, 1, 3)); omega[..., 1] = -10.
        wrench = np.zeros((4, 1, 6)); wrench[..., 4] = 20*levels
        return dict(active=np.ones((4, 1), dtype=bool), ticks=np.array([[500], [501], [521], [541]]),
            phase=np.full((4, 1), 2), sensor_wheel_force_n=np.zeros((4, 1, 2)),
            sensor_com_vz_mps=np.array([[0.], [-.02], [-.3], [-.7]]),
            sensor_assist_apex_tick=np.array([[-1], [500], [500], [500]]),
            assist_strength=levels.copy(), assist_effective_strength=levels.copy(),
            base_rotation_pre=np.tile(np.eye(3), (4, 1, 1, 1)),
            assist_world_omega_pre=omega, assist_wrench=wrench)

    def test_causal_smooth_reduction(self):
        r = audit_arrays(self.trace(), .6)
        self.assertEqual(r['after_ramp_peak_torque_nm'], 12.)
        self.assertEqual(r['total_peak_torque_nm'], 12.5)

    def test_uniform_lower_assistance_is_not_accepted(self):
        z = self.trace(); z['assist_strength'][0] = .6
        with self.assertRaises(AssertionError): audit_arrays(z, .6)

    def test_future_event_not_used_in_current_step(self):
        z = self.trace(); z['sensor_assist_apex_tick'][0] = 500
        with self.assertRaises(AssertionError): audit_arrays(z, .6)

    def test_instant_step_instead_of_ramp_rejected(self):
        z = self.trace(); z['assist_strength'][1] = .6
        with self.assertRaises(AssertionError): audit_arrays(z, .6)

    def test_reported_apex_needs_physical_downward_motion(self):
        z = self.trace(); z['sensor_com_vz_mps'][:] = .1
        with self.assertRaises(AssertionError): audit_arrays(z, .6)

    def test_contact_cannot_trigger_flight_apex(self):
        z = self.trace(); z['sensor_wheel_force_n'][:] = 20.
        with self.assertRaises(AssertionError): audit_arrays(z, .6)

    def test_hidden_linear_lift_rejected(self):
        z = self.trace(); z['assist_wrench'][2, 0, 2] = .01
        with self.assertRaises(AssertionError): audit_arrays(z, .6)

    def test_unreduced_force_behind_reduced_label_rejected(self):
        z = self.trace(); z['assist_wrench'][3, 0, 4] = 12.5
        with self.assertRaises(AssertionError): audit_arrays(z, .6)


if __name__ == '__main__':
    unittest.main()

"""Admission and physical-assistance regression tests; no simulator required."""
import unittest
import numpy as np
from withdrawal_contract import admission
from withdrawal_audit import assistance_arrays


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.anchor = dict(worlds=45, passed=45, mean_force_n=306., max_force_n=331.,
            mean_com_stroke_m=.05285, mean_wheel_cm=9.25, mean_com_cm=11.96,
            mean_recovery_s=1.64, min_stable_s=1., max_rebound_mps=0.)

    def test_retained_quality(self):
        self.assertTrue(admission(dict(self.anchor), self.anchor)['passed'])

    def test_failure_cannot_hide_in_mean(self):
        self.assertFalse(admission(dict(self.anchor, passed=44), self.anchor)['passed'])

    def test_no_height_for_assistance_trade(self):
        for key in ('mean_wheel_cm', 'mean_com_cm'):
            with self.subTest(key=key):
                c = dict(self.anchor); c[key] *= .98
                self.assertFalse(admission(c, self.anchor)['passed'])

    def test_peak_and_stroke_limits(self):
        for key, value in [('mean_force_n', 313.), ('max_force_n', 342.),
                           ('mean_com_stroke_m', .0505), ('min_stable_s', .99),
                           ('max_rebound_mps', .02), ('mean_recovery_s', 1.75)]:
            with self.subTest(key=key):
                c = dict(self.anchor); c[key] = value
                self.assertFalse(admission(c, self.anchor)['passed'])

    def test_nonfinite_is_rejected(self):
        for value in (float('nan'), float('inf')):
            self.assertFalse(admission(dict(self.anchor, mean_force_n=value), self.anchor)['passed'])


class AssistanceTests(unittest.TestCase):
    def trace(self):
        # Identity orientation and -10 rad/s pitch produce +80 Nm before
        # saturation. At t=.50/.55/.60 s, 60% assistance applies 0/6/12 Nm.
        active = np.array([[True], [True], [True], [False]])
        rotation = np.tile(np.eye(3), (4, 1, 1, 1))
        omega = np.zeros((4, 1, 3)); omega[..., 1] = -10.
        wrench = np.zeros((4, 1, 6)); wrench[:, 0, 4] = [0., 6., 12., 0.]
        return dict(active=active, ticks=np.array([[201], [221], [241], [241]]),
                    assist_strength=np.full((4, 1), .60),
                    assist_effective_strength=np.array([[0.], [.30], [.60], [.60]]),
                    base_rotation_pre=rotation, assist_world_omega_pre=omega, assist_wrench=wrench)

    def test_ramp_saturation_and_inactive_zero(self):
        self.assertEqual(assistance_arrays(self.trace(), .6)['peak_torque_nm'], 12.)

    def test_report_cannot_relabel_old_strength(self):
        z = self.trace(); z['assist_strength'][:] = .625
        with self.assertRaises(AssertionError): assistance_arrays(z, .6)

    def test_full_strength_hidden_behind_lower_label(self):
        z = self.trace(); z['assist_wrench'][2, 0, 4] = 12.5
        with self.assertRaises(AssertionError): assistance_arrays(z, .6)

    def test_linear_lift_is_rejected(self):
        z = self.trace(); z['assist_wrench'][2, 0, 2] = 1.
        with self.assertRaises(AssertionError): assistance_arrays(z, .6)


if __name__ == '__main__':
    unittest.main()

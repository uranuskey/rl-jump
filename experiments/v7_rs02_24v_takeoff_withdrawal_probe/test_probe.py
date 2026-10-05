"""CPU checks for actual takeoff withdrawal and strict stage advancement."""
import copy
import unittest
import numpy as np
from probe_audit import audit_arrays
from probe_contract import admission, qualified, deepest_qualified


class WithdrawalTests(unittest.TestCase):
    def trace(self, before=.6125, after=.60):
        levels = np.array([before, before, (before+after)/2, after])[:, None]
        omega = np.zeros((4, 1, 3)); omega[..., 1] = -10.
        wrench = np.zeros((4, 1, 6)); wrench[..., 4] = 20*levels
        return dict(active=np.ones((4, 1), dtype=bool), ticks=np.array([[500], [501], [521], [541]]),
            phase=np.full((4, 1), 2), sensor_wheel_force_n=np.zeros((4, 1, 2)),
            sensor_com_vz_mps=np.array([[0.], [-.02], [-.3], [-.7]]),
            sensor_assist_apex_tick=np.array([[-1], [500], [500], [500]]),
            assist_strength=levels.copy(), assist_effective_strength=levels.copy(),
            base_rotation_pre=np.tile(np.eye(3), (4, 1, 1, 1)),
            assist_world_omega_pre=omega, assist_wrench=wrench)

    def model(self):
        return dict(native=dict(metrics=dict(worlds=45), admission=dict(passed=True), audits=dict(status='PASS')),
            batch_replays=[dict(repeat=i, metrics=dict(worlds=512), admission=dict(passed=True)) for i in (1, 2, 3)])

    def test_takeoff_and_landing_torque_caps(self):
        a = audit_arrays(self.trace(), .6125, .60)
        self.assertAlmostEqual(a['before_apex']['peak_torque_nm'], 12.25)
        self.assertAlmostEqual(a['after_ramp']['peak_torque_nm'], 12.)

    def test_uniform_575_reduces_takeoff_too(self):
        a = audit_arrays(self.trace(.575, .575), .575, .575)
        self.assertAlmostEqual(a['before_apex']['peak_torque_nm'], 11.5)
        self.assertAlmostEqual(a['after_ramp']['peak_torque_nm'], 11.5)

    def test_label_only_takeoff_reduction_rejected(self):
        z = self.trace(); z['assist_wrench'][0, 0, 4] = 12.5
        with self.assertRaises(AssertionError): audit_arrays(z, .6125, .60)

    def test_old_takeoff_schedule_rejected(self):
        with self.assertRaises(AssertionError): audit_arrays(self.trace(.625, .60), .6125, .60)

    def test_noncausal_apex_rejected(self):
        z = self.trace(); z['sensor_assist_apex_tick'][0, 0] = 500
        with self.assertRaises(AssertionError): audit_arrays(z, .6125, .60)

    def test_abrupt_step_rejected(self):
        z = self.trace(); z['assist_strength'][1, 0] = .60
        with self.assertRaises(AssertionError): audit_arrays(z, .6125, .60)

    def test_hidden_lift_rejected(self):
        z = self.trace(); z['assist_wrench'][2, 0, 2] = .01
        with self.assertRaises(AssertionError): audit_arrays(z, .6125, .60)

    def test_all_three_replays_required(self):
        row = self.model()
        self.assertTrue(qualified(row))
        row['batch_replays'][1]['admission']['passed'] = False
        self.assertFalse(qualified(row))

    def test_duplicate_repeat_not_three_passes(self):
        row = self.model(); row['batch_replays'][2]['repeat'] = 2
        self.assertFalse(qualified(row))

    def test_native_failure_blocks_reduction(self):
        row = self.model(); row['native']['audits']['status'] = 'UNQUALIFIED'
        self.assertFalse(qualified(row))

    def test_failed_intermediate_level_cannot_be_skipped(self):
        models = {k:self.model() for k in ('current625_600', 'takeoff6125_600', 'uniform600')}
        models['takeoff6125_600']['batch_replays'][0]['admission']['passed'] = False
        self.assertEqual(deepest_qualified(models), 'current625_600')

    def test_force_improvement_is_not_required_but_bounds_remain(self):
        anchor = dict(worlds=512, passed=512, mean_force_n=306., max_force_n=333.,
            mean_com_stroke_m=.053, mean_wheel_cm=9.25, mean_com_cm=11.97,
            mean_recovery_s=1.64, min_stable_s=1., max_rebound_mps=0.)
        candidate = copy.deepcopy(anchor)
        candidate.update(mean_force_n=307., max_force_n=334.)
        self.assertTrue(admission(candidate, anchor)['passed'])
        candidate['mean_force_n'] = 320.
        self.assertIn('mean_impact', admission(candidate, anchor)['reasons'])

    def test_nonzero_rebound_still_blocks_promotion(self):
        candidate = dict(worlds=512, passed=512, mean_force_n=306., max_force_n=333.,
            mean_com_stroke_m=.053, mean_wheel_cm=9.25, mean_com_cm=11.97,
            mean_recovery_s=1.64, min_stable_s=1., max_rebound_mps=.07)
        self.assertIn('no_rebound', admission(candidate, candidate)['reasons'])


if __name__ == '__main__':
    unittest.main()

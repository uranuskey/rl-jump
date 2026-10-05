"""Admission and evidence tests for a bounded physics correction."""
import copy
import unittest
import numpy as np
from fix_contract import checked_variant, prefix_qualified, qualified, deepest_qualified, VARIANTS
from fix_audit import audit_arrays


class FixTests(unittest.TestCase):
    def rows(self):
        return [dict(repeat=i, worlds=512, failed=0, max_mimic_v_rad_s=.007,
                     max_mimic_q_rad=.0002, prefix_s=1.20) for i in (1, 2, 3)]

    def model(self):
        return dict(native=dict(metrics=dict(worlds=45), admission=dict(passed=True),
            old_reference_admission=dict(passed=True), old_physics_comparison=None, audits=dict(status='PASS')),
            batch_replays=[dict(repeat=i, metrics=dict(worlds=512), admission=dict(passed=True),
                                old_reference_admission=dict(passed=True)) for i in (1, 2, 3)])

    def test_time_constant_respects_euler_two_dt_bound(self):
        self.assertTrue(all(checked_variant(n)['rotor_timeconst'] >= .005 for n in VARIANTS))

    def test_precision_experiment_does_not_change_coupling(self):
        self.assertEqual(checked_variant('solve_strict')['rotor_timeconst'], .020)
        self.assertLess(checked_variant('solve_strict')['tolerance'], 1e-8)

    def test_coupling_experiment_does_not_change_solver(self):
        a, b = checked_variant('original'), checked_variant('rotor10ms')
        a.pop('rotor_timeconst'); b.pop('rotor_timeconst')
        self.assertEqual(a, b)

    def test_returned_variant_cannot_mutate_frozen_contract(self):
        a = checked_variant('original'); a['rotor_timeconst'] = .001
        self.assertEqual(checked_variant('original')['rotor_timeconst'], .020)

    def test_reproductions_need_margin_not_only_a_lucky_pass(self):
        rows = self.rows()
        self.assertTrue(prefix_qualified(rows))
        rows[1]['max_mimic_v_rad_s'] = .00999
        self.assertFalse(prefix_qualified(rows))

    def test_guard_failure_cannot_be_ignored(self):
        rows = self.rows(); rows[0]['failed'] = 1
        self.assertFalse(prefix_qualified(rows))

    def test_duplicates_cannot_be_counted_as_three_replays(self):
        rows = self.rows(); rows[2]['repeat'] = 2
        self.assertFalse(prefix_qualified(rows))

    def test_nonfinite_measurement_cannot_qualify(self):
        rows = self.rows(); rows[0]['max_mimic_v_rad_s'] = float('nan')
        self.assertFalse(prefix_qualified(rows))

    def test_short_or_small_prefix_cannot_qualify(self):
        rows = self.rows(); rows[1]['prefix_s'] = 1.0
        self.assertFalse(prefix_qualified(rows))
        rows = self.rows(); rows[1]['worlds'] = 45
        self.assertFalse(prefix_qualified(rows))

    def test_new_reference_cannot_weaken_old_qualification(self):
        row = self.model(); self.assertTrue(qualified(row))
        row['native']['old_reference_admission']['passed'] = False
        self.assertFalse(qualified(row))
        row = self.model(); row['batch_replays'][1]['old_reference_admission']['passed'] = False
        self.assertFalse(qualified(row))

    def test_same_level_physics_regression_blocks_withdrawal(self):
        row = self.model(); row['native']['old_physics_comparison'] = dict(passed=False)
        self.assertFalse(qualified(row))

    def test_failure_stops_deeper_levels(self):
        models = {n:self.model() for n in ('current625_600','takeoff6125_600','uniform600')}
        models['takeoff6125_600']['batch_replays'][0]['admission']['passed'] = False
        self.assertEqual(deepest_qualified(models), 'current625_600')

    def test_half_assistance_requires_actual_lower_torque(self):
        levels = np.full((4,1), .5)
        omega = np.zeros((4,1,3)); omega[...,1] = -10
        wrench = np.zeros((4,1,6)); wrench[...,4] = 10
        z = dict(active=np.ones((4,1), bool), ticks=np.array([[500],[501],[521],[541]]),
            phase=np.full((4,1),2), sensor_wheel_force_n=np.zeros((4,1,2)),
            sensor_com_vz_mps=np.array([[0.],[-.02],[-.3],[-.7]]),
            sensor_assist_apex_tick=np.array([[-1],[500],[500],[500]]),
            assist_strength=levels.copy(), assist_effective_strength=levels.copy(),
            base_rotation_pre=np.tile(np.eye(3),(4,1,1,1)), assist_world_omega_pre=omega,
            assist_wrench=wrench)
        self.assertEqual(audit_arrays(z,.5,.5)['before_apex']['peak_torque_nm'], 10)
        z['assist_wrench'][0,0,4] = 12.5
        with self.assertRaises(AssertionError): audit_arrays(z,.5,.5)


if __name__ == '__main__':
    unittest.main()

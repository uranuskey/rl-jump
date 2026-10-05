import copy
import unittest
from learning_contract import (learning_admission, qualification, selected_better,
                               diagnostic_better, evaluation_entries, replay_qualified)


def anchor():
    return dict(worlds=512, passed=512, mean_force_n=306., max_force_n=333.,
        mean_wheel_cm=9.25, mean_com_cm=11.97, mean_com_stroke_m=.0528,
        mean_recovery_s=1.64, min_stable_s=1., max_rebound_mps=0.)


class LearningGates(unittest.TestCase):
    def test_small_rebound_can_learn_but_cannot_qualify(self):
        candidate = anchor(); candidate['max_rebound_mps'] = .0782
        self.assertTrue(learning_admission(candidate, anchor())['passed'])
        self.assertFalse(qualification(candidate, anchor())['passed'])
        self.assertFalse(selected_better(candidate, None, anchor()))

    def test_entry_bound_is_explicit(self):
        candidate = anchor(); candidate['max_rebound_mps'] = .10
        self.assertTrue(learning_admission(candidate, anchor())['passed'])
        candidate['max_rebound_mps'] = .100001
        self.assertFalse(learning_admission(candidate, anchor())['passed'])

    def test_nan_cannot_bypass_gate(self):
        for key in ('max_rebound_mps', 'mean_force_n', 'mean_wheel_cm'):
            candidate = anchor(); candidate[key] = float('nan')
            self.assertFalse(learning_admission(candidate, anchor())['passed'])

    def test_failed_physical_case_stays_blocked(self):
        candidate = anchor(); candidate['passed'] = 511
        self.assertFalse(learning_admission(candidate, anchor())['passed'])

    def test_height_loss_is_not_traded_for_soft_landing(self):
        for key, value in [('mean_wheel_cm', 8.9), ('mean_com_cm', 11.5), ('mean_com_stroke_m', .049)]:
            candidate = anchor(); candidate.update(mean_force_n=250.)
            candidate[key] = value
            self.assertFalse(learning_admission(candidate, anchor())['passed'])

    def test_stability_recovery_and_impact_gates_remain(self):
        for key, value in [('min_stable_s', .99), ('mean_recovery_s', 1.75),
                           ('mean_force_n', 313.), ('max_force_n', 345.)]:
            candidate = anchor(); candidate[key] = value
            self.assertFalse(learning_admission(candidate, anchor())['passed'])

    def test_first_qualified_model_does_not_need_to_beat_unqualified_seed_force(self):
        seed = anchor(); seed.update(max_rebound_mps=.0782, mean_force_n=305.)
        candidate = anchor(); candidate['mean_force_n'] = 307.
        self.assertTrue(selected_better(candidate, seed, anchor()))
        self.assertTrue(selected_better(candidate, None, anchor()))

    def test_qualified_incumbent_keeps_original_improvement_margin(self):
        candidate = anchor(); candidate['mean_force_n'] = 305.9
        self.assertFalse(selected_better(candidate, anchor(), anchor()))
        candidate['mean_force_n'] = 305.7
        self.assertTrue(selected_better(candidate, anchor(), anchor()))

    def test_diagnostic_candidate_is_not_automatically_selected(self):
        seed = anchor(); seed['max_rebound_mps'] = .08
        candidate = anchor(); candidate['max_rebound_mps'] = .04
        self.assertTrue(diagnostic_better(candidate, seed, anchor()))
        self.assertFalse(selected_better(candidate, None, anchor()))

    def test_missing_selected_is_explicit(self):
        train = dict(contract=dict(strength=.6), source_checkpoint='source',
            initial_checkpoint='seed', final_checkpoint='latest', selected=None,
            diagnostic_candidate=dict(checkpoint='candidate'))
        self.assertEqual([x[0] for x in evaluation_entries(train)],
                         ['reference625', 'seed', 'candidate', 'latest'])
        train['selected'] = dict(checkpoint='qualified')
        self.assertEqual(evaluation_entries(train)[2], ('selected', 'qualified', .6))

    def test_final_replay_requires_all_three_zero_rebound_batches(self):
        rows = [dict(repeat=i, metrics=anchor(), admission=qualification(anchor(), anchor())) for i in (1, 2, 3)]
        self.assertTrue(replay_qualified(rows))
        self.assertFalse(replay_qualified(rows[:2]))
        rows[2]['metrics']['max_rebound_mps'] = .001
        rows[2]['admission'] = qualification(rows[2]['metrics'], anchor())
        self.assertFalse(replay_qualified(rows))

    def test_duplicate_replay_cannot_count_twice(self):
        row = dict(repeat=1, metrics=anchor(), admission=qualification(anchor(), anchor()))
        self.assertFalse(replay_qualified([copy.deepcopy(row) for _ in range(3)]))


if __name__ == '__main__':
    unittest.main()

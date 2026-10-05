import copy
import unittest
from curriculum_contract import BUDGET,LEVELS,chunk_size,decision,qualification,learning_entry


def rows():
    return dict(native=dict(metrics={'worlds':45},audits={'status':'PASS'},
        strict_admission={'passed':True},learning_admission={'passed':True}),
        batch=[dict(repeat=i,metrics={'worlds':512},strict_admission={'passed':True},
            learning_admission={'passed':True}) for i in (1,2,3)])


class Curriculum(unittest.TestCase):
    def test_seed_pass_skips_all_training(self): self.assertEqual(decision(rows(),0),'ADVANCE')
    def test_does_not_wait_for_128(self): self.assertEqual(decision(rows(),2),'ADVANCE')
    def test_pass_can_advance_at_budget(self): self.assertEqual(decision(rows(),128),'ADVANCE')
    def test_one_failed_replay_needs_learning(self):
        r=rows();r['batch'][1]['strict_admission']['passed']=False
        self.assertEqual(decision(r,2),'LEARN')
    def test_failure_outside_learning_entry_stops(self):
        r=rows();r['batch'][0]['strict_admission']['passed']=False;r['batch'][0]['learning_admission']['passed']=False
        self.assertEqual(decision(r,0),'ENTRY_FAILED')
    def test_budget_is_shared(self):
        r=rows();r['batch'][1]['strict_admission']['passed']=False
        self.assertEqual(decision(r,128),'BUDGET_EXHAUSTED')
    def test_initial_short_chunk(self):self.assertEqual(chunk_size(20,0),2)
    def test_subsequent_eight(self):self.assertEqual(chunk_size(22,2),8)
    def test_never_exceeds_budget(self):self.assertEqual(chunk_size(127,10),1)
    def test_missing_repeat_no_promotion(self):
        r=rows();r['batch'].pop();self.assertFalse(qualification(r));self.assertFalse(learning_entry(r))
    def test_failed_native_audit_no_promotion(self):
        r=rows();r['native']['audits']['status']='UNQUALIFIED';self.assertEqual(decision(r,0),'ENTRY_FAILED')
    def test_levels_are_monotone(self):
        previous=(.625,.60)
        for name,a,b in LEVELS:
            self.assertLessEqual(a,previous[0]);self.assertLessEqual(b,previous[1]);self.assertGreaterEqual(a,b);previous=(a,b)
        self.assertEqual(previous,(.50,.50))


if __name__=='__main__':unittest.main()

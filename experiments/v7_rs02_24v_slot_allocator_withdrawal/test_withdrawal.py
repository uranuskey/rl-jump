import copy,unittest
from allocation_contract import BUDGET,PRIOR_UPDATES,LEVELS,chunk_size,decision,qualification
def rows():
    return dict(native=dict(metrics={'worlds':45},audits={'status':'PASS'},
        strict_admission={'passed':True},learning_admission={'passed':True}),
        batch=[dict(repeat=i,metrics={'worlds':512},strict_admission={'passed':True},
                    learning_admission={'passed':True}) for i in (1,2,3)])
class Withdrawal(unittest.TestCase):
    def test_prior_updates_count_against_budget(self):self.assertEqual(PRIOR_UPDATES+BUDGET,128)
    def test_zero_training_promotion(self):self.assertEqual(decision(rows(),0),'ADVANCE')
    def test_budget_does_not_delay_promotion(self):self.assertEqual(decision(rows(),BUDGET),'ADVANCE')
    def test_failed_contact_stops(self):
        r=rows();r['batch'][1]['strict_admission']['passed']=False;r['batch'][1]['learning_admission']['passed']=False
        self.assertEqual(decision(r,0),'ENTRY_FAILED')
    def test_rebound_entry_can_learn(self):
        r=rows();r['batch'][0]['strict_admission']['passed']=False
        self.assertEqual(decision(r,0),'LEARN')
    def test_shared_budget_stops_learning(self):
        r=rows();r['batch'][0]['strict_admission']['passed']=False
        self.assertEqual(decision(r,BUDGET),'BUDGET_EXHAUSTED')
    def test_chunk_sizes_and_remaining(self):
        self.assertEqual(chunk_size(0,0),2);self.assertEqual(chunk_size(2,2),8)
        self.assertEqual(chunk_size(BUDGET-1,2),1)
    def test_no_missing_or_duplicate_replay(self):
        r=rows();r['batch'][2]['repeat']=2;self.assertFalse(qualification(r))
        r=rows();r['batch'].pop();self.assertFalse(qualification(r))
    def test_native_audit_required(self):
        r=rows();r['native']['audits']['status']='UNQUALIFIED';self.assertEqual(decision(r,0),'ENTRY_FAILED')
    def test_revalidate_changed_controller_and_stop_at50(self):
        self.assertEqual(LEVELS,[('uniform550',.55,.55),('uniform525',.525,.525),('uniform500',.5,.5)])
if __name__=='__main__':unittest.main()

import unittest
from transfer_contract import admission,promote,final_qualified


def metrics():
    return dict(worlds=512,passed=512,mean_wheel_cm=9.25,mean_com_cm=11.97,
        mean_force_n=306.,max_force_n=332.,mean_com_stroke_m=.053,
        mean_recovery_s=1.64,min_stable_s=1.,max_rebound_mps=0.)


class Gates(unittest.TestCase):
    def gate(self,changes=None,learning=False,pre=.006):
        a=metrics(); m={**a,**(changes or {})}
        return admission(m,{'new':a,'old':a},pre,learning)

    def test_original_strict_pass(self): self.assertTrue(self.gate()['passed'])
    def test_rebound_entry_not_promotion(self):
        self.assertTrue(self.gate({'max_rebound_mps':.076},True)['passed'])
        self.assertFalse(self.gate({'max_rebound_mps':.076})['passed'])
    def test_entry_rebound_limit(self): self.assertFalse(self.gate({'max_rebound_mps':.10001},True)['passed'])
    def test_physical_failure_not_admitted(self): self.assertFalse(self.gate({'passed':511},True)['passed'])
    def test_height_preserved(self): self.assertFalse(self.gate({'mean_wheel_cm':8.9},True)['passed'])
    def test_force_cap_preserved(self): self.assertFalse(self.gate({'max_force_n':350},True)['passed'])
    def test_margin_preserved(self): self.assertFalse(self.gate(pre=.0081,learning=True)['passed'])
    def test_both_references_required(self):
        a=metrics(); weaker={**a,'max_force_n':400.}
        self.assertFalse(admission({**a,'max_force_n':350.},{'new':weaker,'old':a},.006,True)['passed'])
    def test_seed_cannot_be_selected(self):
        self.assertFalse(promote({'update':0,'strict_admission':{'passed':True}}))
    def test_select_without_force_improvement(self):
        self.assertTrue(promote({'update':8,'strict_admission':{'passed':True},'metrics':{'mean_force_n':310}}))
    def test_all_three_replays_required(self):
        row=dict(native=dict(metrics={'worlds':45},audits={'status':'PASS'},strict_admission={'passed':True}),
            batch=[dict(repeat=i,metrics={'worlds':512},strict_admission={'passed':True}) for i in (1,2,3)])
        self.assertTrue(final_qualified(row))
        row['batch'][0]['strict_admission']['passed']=False
        self.assertFalse(final_qualified(row))
        row['batch']=row['batch'][1:]
        self.assertFalse(final_qualified(row))


if __name__=='__main__': unittest.main()

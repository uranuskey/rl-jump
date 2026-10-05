import copy,unittest
import numpy as np
from zero_contract import LEVELS,BUDGET,checked_levels,decision,qualification,zero_qualified,chunk_size
from zero_wrench_audit import external_arrays

def rows(strength=.5):
    def row(worlds):
        return dict(before=strength,after=strength,metrics={'worlds':worlds},
            audits={'status':'PASS'},strict_admission={'passed':True},learning_admission={'passed':True},
            external_wrench=dict(min_sampled_ticks=2000,max_abs_linear_force_n=0.,
                max_abs_body_torque_nm=20*strength,max_abs_root_generalized_force=0.))
    return dict(native=row(45),batch=[dict(repeat=i,**row(512)) for i in (1,2,3)])

def trace():
    return dict(sensor_external_body_wrench=np.zeros((4,2,3,6)),
        sensor_external_root_generalized_force=np.zeros((4,2,6)),
        sensor_external_base_body_id=np.ones((4,2),dtype=int),assist_wrench=np.zeros((4,2,6)))

class Contract(unittest.TestCase):
    def test_revalidate50_then_exact_zero(self):
        self.assertEqual(LEVELS[0],('uniform500',.5,.5))
        self.assertEqual(LEVELS[-1],('uniform000',0.,0.))
        self.assertEqual(len(LEVELS),21)
        self.assertEqual([int(round(a*1000)) for _,a,b in LEVELS],list(range(500,-1,-25)))
    def test_reject_undeclared_or_negative_levels(self):
        for x in ((-.025,-.025),(.525,.525),(.5,.475),(float('nan'),0)):
            with self.assertRaises(AssertionError):checked_levels(*x)
    def test_promotion_without_ppo(self):self.assertEqual(decision(rows(),0),'ADVANCE')
    def test_qualification_still_advances_at_budget(self):self.assertEqual(decision(rows(),BUDGET),'ADVANCE')
    def test_physical_failure_needs_repair(self):
        r=rows();r['batch'][1]['strict_admission']['passed']=False
        r['batch'][1]['learning_admission']['passed']=False
        self.assertEqual(decision(r,0),'ENTRY_FAILED')
    def test_rebound_learning_does_not_qualify(self):
        r=rows();r['batch'][0]['strict_admission']['passed']=False
        self.assertEqual(decision(r,0),'LEARN');self.assertFalse(qualification(r))
    def test_bounded_learning_chunk(self):
        self.assertEqual(chunk_size(0,0),2);self.assertEqual(chunk_size(2,2),8)
        self.assertEqual(chunk_size(BUDGET-1,20),1)
    def test_fixed_three_replays(self):
        r=rows();r['batch'][2]['repeat']=2;self.assertFalse(qualification(r))
    def test_missing_force_evidence_not_accepted(self):
        r=rows();r['batch'][0].pop('external_wrench');self.assertFalse(qualification(r))
    def test_hidden_root_force_rejected(self):
        r=rows();r['native']['external_wrench']['max_abs_root_generalized_force']=.1
        self.assertFalse(qualification(r))
    def test_zero_requires_exact_zero(self):
        r=rows(0);self.assertTrue(qualification(r))
        r['batch'][0]['external_wrench']['max_abs_body_torque_nm']=1e-12
        self.assertFalse(qualification(r))
    def test_no_zero_claim_at_nonzero_floor(self):
        self.assertFalse(zero_qualified([dict(before=.025,after=.025)]))
        self.assertTrue(zero_qualified([dict(before=0.,after=0.)]))
    def test_zero_trace(self):self.assertTrue(external_arrays(trace(),0,0)['zero_external_wrench'])
    def test_tampered_nonbase_torque(self):
        z=trace();z['sensor_external_body_wrench'][0,0,2,3]=1e-9
        with self.assertRaises(AssertionError):external_arrays(z,0,0)
    def test_tampered_root_force(self):
        z=trace();z['sensor_external_root_generalized_force'][2,1,0]=1e-9
        with self.assertRaises(AssertionError):external_arrays(z,0,0)
    def test_declared_zero_but_actual_base_torque(self):
        z=trace();z['sensor_external_body_wrench'][0,0,1,4]=1e-9;z['assist_wrench'][0,0,4]=1e-9
        with self.assertRaises(AssertionError):external_arrays(z,0,0)
if __name__=='__main__':unittest.main()


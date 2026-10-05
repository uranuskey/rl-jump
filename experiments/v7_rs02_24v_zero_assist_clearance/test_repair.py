import copy,unittest,tempfile
from pathlib import Path
import numpy as np
from repair_profiles import WINDOWS,candidates,choose,validate_profile
from repair_contract import LEVELS,decision,qualification
from pose_evidence import check_pose,digest as sha

BASE=dict(name='slot_allocation_125',slot_action_bound=1.25,slot_kx=3750,slot_dx=40,slot_force_cap_n=50)
def rows():
    return [dict(profile=p,max_slot_error_mm=12-i*.3,strict_admission={'passed':True},external_valid=True)
        for i,p in enumerate(candidates(BASE))]

class Repair(unittest.TestCase):
    def test_only_window_and_name_can_change(self):
        for p in candidates(BASE):validate_profile(BASE,p)
        p=candidates(BASE)[1];p['slot_force_cap_n']=51
        with self.assertRaises(AssertionError):validate_profile(BASE,p)
    def test_undeclared_window_rejected(self):
        p=candidates(BASE)[0];p['slot_action_bound']=2.5
        with self.assertRaises(AssertionError):validate_profile(BASE,p)
    def test_requires_half_mm_improvement(self):
        r=rows();self.assertEqual(choose(r),r[2])
    def test_skip_strict_failure(self):
        r=rows();r[2]['strict_admission']['passed']=False
        self.assertEqual(choose(r),r[3])
    def test_no_promotion_from_same_quality(self):
        r=rows()
        for x in r:x['max_slot_error_mm']=12
        self.assertIsNone(choose(r))
    def test_hidden_external_force_excludes_screen_candidate(self):
        r=rows()
        for x in r[2:]:x['external_valid']=False
        self.assertIsNone(choose(r))
    def test_candidate_order_is_fixed(self):
        r=rows();r[0],r[1]=r[1],r[0]
        with self.assertRaises(AssertionError):choose(r)
    def test_nonfinite_metric_cannot_win(self):
        r=rows();r[1]['max_slot_error_mm']=float('nan')
        with self.assertRaises(AssertionError):choose(r)
    def test_requalify45_then_descend_to_zero(self):
        self.assertEqual(LEVELS[0],('uniform450',.45,.45))
        self.assertEqual(LEVELS[-1],('uniform000',0.,0.))
        self.assertEqual(len(LEVELS),19)
    def test_pose_evidence_hash_and_world_count(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'pose.npz';np.savez(p,q=np.zeros((2,4)),v=np.zeros((2,3)),ticks=np.ones(2),minimum_q=np.zeros((2,4)))
            check_pose(p,sha(p),2)
            with self.assertRaises(AssertionError):check_pose(p,'bad',2)
            with self.assertRaises(AssertionError):check_pose(p,sha(p),3)
    def test_nonfinite_pose_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'pose.npz';np.savez(p,q=np.full((2,4),np.nan),v=np.zeros((2,3)),ticks=np.ones(2),minimum_q=np.zeros((2,4)))
            with self.assertRaises(AssertionError):check_pose(p,sha(p),2)

if __name__=='__main__':unittest.main()

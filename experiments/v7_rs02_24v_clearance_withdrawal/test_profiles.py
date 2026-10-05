import unittest
from profile_contract import candidates,validate_profile
class Profiles(unittest.TestCase):
    def setUp(self):
        self.base=dict(name='late3_slot_high',slot_kx=3000.,slot_dx=32.,slot_force_cap_n=40.,
            enabled=True,brake_start_m=.14,brake_full_m=.115,brake_kp_extra=.2,brake_kd_extra=.1,
            horizon_s=.03,extra_limit_m=.003)
    def test_source_unchanged(self):
        a=self.base.copy();profiles=candidates(self.base)
        self.assertEqual(a,self.base);self.assertEqual(profiles[0],a)
    def test_each_change_stays_within_scope(self):
        for p in candidates(self.base):self.assertTrue(validate_profile(self.base,p))
    def test_reject_shortened_stroke_protection(self):
        p=candidates(self.base)[1];p['brake_full_m']=.13
        with self.assertRaises(AssertionError):validate_profile(self.base,p)
    def test_reject_expanded_cap(self):
        p=candidates(self.base)[1];p['slot_force_cap_n']=81
        with self.assertRaises(AssertionError):validate_profile(self.base,p)
if __name__=='__main__':unittest.main()

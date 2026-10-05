import unittest
from allocation_profiles import BOUNDS,candidates,validate_profile,choose
class Profiles(unittest.TestCase):
    def test_only_window_and_name_change(self):
        b={'name':'old','slot_kx':3750.,'slot_dx':40.,'slot_force_cap_n':50.,'other':17.}
        for p in candidates(b):self.assertTrue(validate_profile(b,p))
    def test_reject_gain_change(self):
        b={'name':'old','slot_kx':3750.};p=candidates(b)[1];p['slot_kx']=9999.
        with self.assertRaises(AssertionError):validate_profile(b,p)
    def test_material_improvement_and_least_range(self):
        r=[{'strict_admission':{'passed':True},'max_slot_error_mm':x} for x in (12.,11.1,10.9,8.)]
        self.assertIs(choose(r),r[2])
        r[2]['strict_admission']['passed']=False
        self.assertIs(choose(r),r[3])
    def test_none_is_not_permission_to_retry(self):
        r=[{'strict_admission':{'passed':False},'max_slot_error_mm':x} for x in (12.,10.,9.,8.)]
        self.assertIsNone(choose(r))
    def test_bounded_search(self):self.assertEqual(BOUNDS,(1.,1.25,1.5,2.))
if __name__=='__main__':unittest.main()

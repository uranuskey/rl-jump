"""CPU checks for preserving the chosen policy and its feedback outputs."""
import unittest
import torch
from spread_probe import change_parameters, PROFILES, load_controller_audit, PARENT
from compliant_control import encode, decode


class ParameterContract(unittest.TestCase):
    def test_controller_audit_resolves_exact_v3_file(self):
        from pathlib import Path
        audit=load_controller_audit()
        self.assertEqual(Path(audit.__code__.co_filename).resolve(),(PARENT/'audit.py').resolve())

    def test_baseline_exact_and_only_requested_parameters_change(self):
        initial=torch.tensor([.16255,.19258,.04785,.12162,.60617,.47064,.41921,.73907])
        raw=torch.cat((encode(initial),torch.arange(8)/100)).repeat(405,1)
        changed=change_parameters(raw,PROFILES)
        self.assertTrue(torch.equal(changed[:45],raw[:45]))
        self.assertTrue(torch.equal(changed[:,8:],raw[:,8:]))
        values,_=decode(changed)
        for i,p in enumerate(PROFILES):
            expected=initial.clone()
            expected[0]+=p['air_delta']
            expected[1]+=p['time_delta']
            expected[6]*=p['kd_ratio']
            torch.testing.assert_close(values[i*45:(i+1)*45],expected.expand(45,-1))

    def test_out_of_bounds_is_rejected_not_silently_clipped(self):
        raw=torch.cat((encode([.146,.11,.05,.12,.6,.45,.4,.7]),torch.zeros(8))).repeat(45,1)
        with self.assertRaises(AssertionError):
            change_parameters(raw,[PROFILES[1]])


if __name__=='__main__':
    unittest.main()

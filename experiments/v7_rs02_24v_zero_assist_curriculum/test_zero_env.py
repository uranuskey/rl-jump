import unittest
from types import SimpleNamespace
from unittest.mock import patch
import zero_runtime
import torch
from allocation_env import AllocationEnv
from allocation_env_v2 import AllocationEnvV2
from zero_env import ZeroAssistEnv

class Environment(unittest.TestCase):
    def constructor(self,strength):
        def init(e,n,profiles,**kw):
            self.assertEqual((kw['before'],kw['after']),(.5,.5))
            e.n,e.device=n,'cpu';e.assist_strength=torch.full((n,),.5)
        with patch.object(AllocationEnvV2,'__init__',init):
            return ZeroAssistEnv(2,[],before=strength,after=strength)
    def test_exact_zero_after_construction(self):
        e=self.constructor(0.)
        self.assertEqual((e.before,e.after,e.assist_target),(0.,0.,0.))
        self.assertEqual(int(torch.count_nonzero(e.assist_strength)),0)
    def test_zero_event_tracking_not_skipped(self):
        e=self.constructor(0.);e.assist_apex_tick=torch.tensor([-1,-1])
        e.ticks=torch.tensor([490,491]);e.task=SimpleNamespace(height_score=SimpleNamespace(apex=torch.tensor([True,False])))
        e.terminal_mask=lambda:torch.tensor([False,False])
        with patch.object(AllocationEnv,'control_tick',lambda obj,x:x):
            e.control_tick(torch.zeros(2,15))
        self.assertEqual(e.assist_apex_tick.tolist(),[490,-1])
        self.assertEqual(int(torch.count_nonzero(e.assist_strength)),0)
    def test_whole_body_and_root_force_accumulation(self):
        e=self.constructor(.475);e.external=torch.zeros(2,3,6);e.force=torch.zeros(2,16)
        e.external[1,2,4]=.7;e.force[0,2]=.3
        e.paused=torch.zeros(2,dtype=torch.bool);e.record=True;e.ticks=torch.ones(2,dtype=torch.long);e.base=1
        with patch.object(AllocationEnvV2,'sensors',lambda obj,motor=None:{}):
            x=e.sensors({})
        self.assertAlmostEqual(e.external_peak[1,1].item(),.7,places=6)
        self.assertAlmostEqual(e.external_peak[0,2].item(),.3,places=6)
        self.assertEqual(e.external_sample_ticks.tolist(),[1,1])
        self.assertTrue(torch.equal(x['external_body_wrench'],e.external))
        self.assertNotEqual(x['external_body_wrench'].data_ptr(),e.external.data_ptr())
    def test_monitor_reset_does_not_leak_old_peak(self):
        e=self.constructor(.5);e.external_peak.fill_(2);e.external_sample_ticks.fill_(2000)
        with patch.object(AllocationEnvV2,'reset',lambda obj,mask,**kwargs:None):
            e.reset(torch.tensor([True,False]))
        self.assertEqual(e.external_peak[0].sum().item(),0)
        self.assertEqual(e.external_peak[1].sum().item(),6)
        self.assertEqual(e.external_sample_ticks.tolist(),[0,2000])
if __name__=='__main__':unittest.main()


"""Regression for overridden control functions accidentally skipping event updates."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import allocation_paths
import torch
from allocation_env import AllocationEnv
from allocation_env_v2 import AllocationEnvV2
class ApexIntegration(unittest.TestCase):
    def env(self,before=.55,after=.525):
        e=object.__new__(AllocationEnvV2)
        e.before,e.after=before,after
        e.assist_apex_tick=torch.tensor([-1,-1])
        e.assist_strength=torch.full((2,),before)
        e.ticks=torch.tensor([488,489])
        e.task=SimpleNamespace(height_score=SimpleNamespace(apex=torch.tensor([False,True])))
        e.terminal_mask=lambda:torch.tensor([False,False])
        return e
    def call(self,e):
        held=torch.randn(2,15)
        with patch.object(AllocationEnv,'control_tick',lambda obj,x:x):
            self.assertIs(e.control_tick(held),held)
    def test_pre_apex_and_first_event(self):
        e=self.env();self.call(e)
        self.assertEqual(e.assist_apex_tick.tolist(),[-1,489])
        self.assertTrue(torch.allclose(e.assist_strength,torch.tensor([.55,.55])))
    def test_apex_not_overwritten_and_exact_ramp(self):
        e=self.env();self.call(e);e.ticks+=20;self.call(e)
        self.assertEqual(e.assist_apex_tick.tolist(),[-1,489])
        self.assertTrue(torch.allclose(e.assist_strength,torch.tensor([.55,.5375])))
        e.ticks+=20;self.call(e)
        self.assertTrue(torch.allclose(e.assist_strength,torch.tensor([.55,.525])))
    def test_terminal_cannot_start_apex(self):
        e=self.env();e.terminal_mask=lambda:torch.tensor([False,True]);self.call(e)
        self.assertEqual(e.assist_apex_tick.tolist(),[-1,-1])
    def test_uniform_strength_unchanged_but_event_recorded(self):
        e=self.env(.55,.55);self.call(e);e.ticks+=40;self.call(e)
        self.assertEqual(e.assist_apex_tick.tolist(),[-1,489])
        self.assertTrue(torch.equal(e.assist_strength,torch.full((2,),.55)))
    def test_sensor_event_may_only_follow_observed_apex(self):
        e=self.env();e.task.height_score.apex.zero_();self.call(e)
        self.assertEqual(e.assist_apex_tick.tolist(),[-1,-1])
        e.ticks+=1;e.task.height_score.apex[0]=True;self.call(e)
        self.assertEqual(e.assist_apex_tick.tolist(),[489,-1])
if __name__=='__main__':unittest.main()

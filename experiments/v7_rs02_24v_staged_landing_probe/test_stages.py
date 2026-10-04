"""CPU contract checks for the new causal schedule and unchanged air tracking."""
import unittest
import staged_probe  # establish the exact frozen import chain
import torch
from staged_control import modulate,profile,table


class StagedContract(unittest.TestCase):
    def sample(self,*,enabled=True,seen=False,speed=-1.3,drop=.005,previous=0.,cfg=None,ticks=400,touch=.99):
        n=45
        one=torch.ones(n)
        p=torch.tensor([.1575,.1926,.04785,.1216,.606,.4706,.4192,.739]).expand(n,-1)
        c=dict(enabled=torch.full((n,),enabled),kp=.4706*one,kd=.4192*one,
               force=50*one if seen else 0*one,height=.18*one,velocity=-.7*one,
               state=dict(gate_tick=torch.full((n,),360)))
        result=modulate(c,p,table([cfg or profile('test',early_kp=.6,early_kd=.8)],'cpu'),
            previous*one,ticks=torch.full((n,),ticks),touchdown=touch*one if seen else 0*one,
            touch_com=.3*one,com_z=(.3-drop)*one,com_vz=speed*one,mass=7.228699)
        return c,result

    def test_disabled_and_pre_apex_paths_are_bitwise_unchanged(self):
        for kwargs in (dict(enabled=False),dict(cfg=profile('baseline',enabled=False),seen=True)):
            c,(out,_,_)=self.sample(**kwargs)
            for k in ('kp','kd','force','height','velocity'):
                self.assertTrue(torch.equal(c[k],out[k]))

    def test_air_velocity_tracking_and_reference_are_preserved(self):
        c,(out,_,progress)=self.sample()
        self.assertTrue(bool((out['kp']<c['kp']).all()))
        for k in ('kd','velocity','height','force'):
            self.assertTrue(torch.equal(out[k],c[k]))
        self.assertTrue(bool((progress==0).all()))

    def test_compression_and_speed_increase_braking(self):
        _,(_,_,low)=self.sample(seen=True,drop=.003)
        _,(_,_,deep)=self.sample(seen=True,drop=.013)
        _,(_,_,fast)=self.sample(seen=True,drop=.003,speed=-2.)
        self.assertTrue(bool((deep>low).all()))
        self.assertTrue(bool((fast>low).all()))

    def test_braking_stage_cannot_relax_after_progress(self):
        _,(_,_,progress)=self.sample(seen=True,drop=.001,speed=-.5,previous=.8)
        torch.testing.assert_close(progress,torch.full((45,),.8))

    def test_original_gains_restore_after_recovery(self):
        c,(out,_,_)=self.sample(seen=True,ticks=900,touch=.99)
        for k in ('kp','kd','force'):
            self.assertTrue(torch.equal(c[k],out[k]))

    def test_profiles_and_audit_resolve_without_name_collision(self):
        self.assertEqual(len(staged_probe.PROFILES)*45,495)
        self.assertEqual(staged_probe.load_controller_audit().__code__.co_filename,
                         str(staged_probe.PARENT/'audit.py'))


if __name__=='__main__':
    unittest.main()

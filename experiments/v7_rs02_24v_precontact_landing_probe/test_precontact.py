"""Meaningful CPU checks for causal feedback, travel bounds, and phase isolation."""
import unittest
import torch
from precontact_control import DT, initial, modulate, profile, table


class PrecontactContract(unittest.TestCase):
    def sample(self, *, cfg=None, gap=.02, closing=.95, com=-1.4, seen=False,
               enabled=True, height=.19, old_offset=0., old_speed=0., protect=False):
        n=45; one=torch.ones(n)
        c=dict(enabled=torch.full((n,),enabled),height=height*one,velocity=-.75*one,
               kp=.47*one,kd=.42*one,force=torch.zeros(n))
        s=initial(n,'cpu'); s['count'].fill_(2)
        clear=torch.full((n,2),gap)
        s['clearance2']=clear+closing*2*DT
        s['clearance1']=clear+closing*DT
        s['offset'].fill_(old_offset);s['speed'].fill_(old_speed)
        result=modulate(c,table([cfg or profile('test')],'cpu'),s,clearance=clear,
            surface_z=torch.zeros(n),com_vz=com*one,touchdown=one if seen else 0*one,
            leg_height=torch.full((n,2),height),protect_landing=protect)
        return c,result

    def test_baseline_preapex_ascent_and_postcontact_unchanged(self):
        for kw in (dict(cfg=profile('base',enabled=False)),dict(enabled=False),
                   dict(com=.2),dict(seen=True,old_offset=-.005,old_speed=.5)):
            c,(out,diag,_)=self.sample(**kw)
            for key in ('height','velocity','kp','kd','force'):
                self.assertTrue(torch.equal(c[key],out[key]))
            self.assertTrue(bool((diag['pre_offset_m']==0).all()))

    def test_closer_ground_and_faster_approach_increase_retraction(self):
        _,(_,far,_)=self.sample(gap=.08)
        _,(_,near,_)=self.sample(gap=.02)
        _,(_,slow,_)=self.sample(gap=.02,closing=.2)
        self.assertTrue(bool((near['pre_offset_m']<far['pre_offset_m']).all()))
        self.assertTrue(bool((near['pre_offset_m']<slow['pre_offset_m']).all()))

    def test_no_impedance_or_force_change(self):
        c,(out,_,_)=self.sample()
        for key in ('kp','kd','force'):
            self.assertTrue(torch.equal(c[key],out[key]))

    def test_displacement_velocity_consistency_and_bounds(self):
        old=-.004
        _,(_,diag,_)=self.sample(gap=.002,closing=2.,old_offset=old,old_speed=.6)
        torch.testing.assert_close(diag['pre_offset_velocity_mps'],(diag['pre_offset_m']-old)/DT)
        self.assertTrue(bool((diag['pre_offset_m']>=-.006).all()))
        self.assertTrue(bool((diag['pre_offset_velocity_mps'].abs()<=.65+1e-5).all()))

    def test_nominal_travel_reserve(self):
        _,(out,_,_)=self.sample(height=.163,gap=.001,closing=2.,old_offset=-.005,old_speed=.6)
        self.assertTrue(bool((out['height']>=.160-1e-7).all()))

    def test_ground_translation_does_not_invent_wheel_velocity(self):
        c,(_,diag,state)=self.sample()
        clear=state['clearance1'];previous=initial(45,'cpu');previous['count'].fill_(2)
        previous['clearance2']=clear+.95*2*DT
        a=modulate(c,table([profile('test')],'cpu'),previous,clearance=clear,
            surface_z=torch.zeros(45),com_vz=torch.full((45,),-1.4),touchdown=torch.zeros(45),
            leg_height=torch.full((45,2),.19))
        b=modulate(c,table([profile('test')],'cpu'),previous,clearance=clear,
            surface_z=torch.full((45,),.01),com_vz=torch.full((45,),-1.4),touchdown=torch.zeros(45),
            leg_height=torch.full((45,2),.19))
        self.assertTrue(torch.equal(a[1]['pre_wheel_vz_mps'],b[1]['pre_wheel_vz_mps']))
        self.assertTrue(bool((b[1]['pre_ttc_s']<a[1]['pre_ttc_s']).all()))

    def test_geometry_protection_is_postcontact_only_and_keeps_reference(self):
        c,(air,_,_)=self.sample(height=.115,protect=True)
        c,(landed,diag,_)=self.sample(height=.115,protect=True,seen=True)
        self.assertTrue(torch.equal(air['kp'],c['kp']))
        torch.testing.assert_close(landed['kp'],1.2*c['kp'])
        torch.testing.assert_close(landed['kd'],1.1*c['kd'])
        for key in ('height','velocity','force'):
            self.assertTrue(torch.equal(landed[key],c[key]))


if __name__=='__main__':
    unittest.main()

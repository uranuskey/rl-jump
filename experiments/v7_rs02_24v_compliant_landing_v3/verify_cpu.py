"""Behavioral tests for admission-critical control changes; CPU only."""
import unittest
import compliant_paths
import torch
from compliant_control import LOW,HIGH,INITIAL,decode,encode,initial_raw,tick,hermite_down,air_reference
from compliant_servo import payload,servo,legacy_servo
from rs02_actuator import PhysicalStepFIFO
from height_contract import reference_joints


class ControlTests(unittest.TestCase):
    def test_air_curve_late_retraction_and_exact_velocity(self):
        age=torch.linspace(0,.22,2201,dtype=torch.float64)
        h,v=air_reference(age*0+.224,age*0+.165,age*0+.22,age)
        numeric=(h[2:]-h[:-2])/.0002
        self.assertLess(float((numeric-v[1:-1]).abs().max()),1e-5)
        self.assertTrue(bool((h[1:]<=h[:-1]).all()))
        self.assertAlmostEqual(float(v[0]),0)
        self.assertAlmostEqual(float(v[-1]),0)
        self.assertGreater(float(h[1100]),(.224+.165)/2)
    def test_independent_distance(self):
        values,_=decode(torch.randn(500,16)*10)
        self.assertTrue(bool((values[:,2]>=.03).all() and (values[:,2]<=.07).all()))
        self.assertTrue(torch.allclose(decode(initial_raw()[None])[0][0],torch.tensor(INITIAL)))
    def test_monotone_velocity_matched_curve(self):
        age=torch.linspace(0,.20,201)
        h,v=hermite_down(torch.full_like(age,.20),torch.full_like(age,.145),torch.full_like(age,-1.4),torch.full_like(age,.12),age)
        self.assertLess(float(v[0]),-1)
        self.assertTrue(bool((h[1:]<=h[:-1]+1e-7).all()))
        self.assertGreaterEqual(float(h.min()),.145-1e-6)
        self.assertAlmostEqual(float(v[-1]),0)
    def test_no_crouch_overshoot_for_extremes(self):
        for depth in (.03,.07):
            for duration in (.07,.16):
                a=torch.linspace(0,duration,1001)
                h,v=hermite_down(a*0+.19,a*0+.19-depth,a*0-2,a*0+duration,a)
                self.assertTrue(bool((h>=.19-depth-1e-6).all() and (h<=.190001).all()))
                self.assertTrue(bool((v<=1e-5).all()))
    def args(self):
        z=torch.zeros(3)
        state=dict(gate_tick=torch.full((3,),-1,dtype=torch.long),air_start=z.clone(),touch_seen=z.bool(),touch_height=z.clone(),touch_velocity=z.clone())
        args=dict(ticks=torch.tensor([360,361,360]),apex=torch.tensor([False,True,False]),terminal=z.bool(),launch_height=z+.22,touchdown=z.clone(),leg_height=torch.full((3,2),.20),incident_vz=z-1.4,gravity=torch.zeros(3,3),gyro=torch.zeros(3,3),linear=torch.zeros(3,3),wheel_speed=torch.zeros(3,2),mass=7.228699)
        return state,args
    def test_per_world_gate_and_no_mutation(self):
        state,args=self.args()
        a=tick(initial_raw().expand(3,-1),state,**args)
        b=tick(initial_raw().expand(3,-1)+2,state,**args)
        self.assertEqual(a['enabled'].tolist(),[False,True,False])
        self.assertEqual(a['state']['gate_tick'].tolist(),[-1,361,-1])
        self.assertEqual(state['gate_tick'].tolist(),[-1,-1,-1])
        self.assertTrue(torch.equal(a['correction'][~args['apex']],b['correction'][~args['apex']]))
    def test_touch_reference_uses_measured_leg_and_incident_velocity(self):
        state,args=self.args()
        a=tick(initial_raw().expand(3,-1),state,**args)
        args['ticks']+=60
        args['touchdown'][1]=args['ticks'][1]*.0025
        b=tick(initial_raw().expand(3,-1),a['state'],**args)
        self.assertAlmostEqual(float(b['height'][1]),.20,places=6)
        self.assertLess(float(b['velocity'][1]),-1)
        self.assertAlmostEqual(float(b['bottom'][1]),.15,places=6)
    def test_soften_before_contact_restore_after_recovery(self):
        state,args=self.args()
        a=tick(initial_raw().expand(3,-1),state,**args)
        args['ticks']+=40
        b=tick(initial_raw().expand(3,-1),a['state'],**args)
        self.assertAlmostEqual(float(b['kd'][1]),INITIAL[6],places=6)
        self.assertEqual(float(b['force'][1]),0)
        args['touchdown'][1]=args['ticks'][1]*.0025
        c=tick(initial_raw().expand(3,-1),b['state'],**args)
        args['ticks']+=400
        d=tick(initial_raw().expand(3,-1),c['state'],**args)
        self.assertAlmostEqual(float(d['height'][1]),.18,places=6)
        self.assertEqual(float(d['kp'][1]),1)
        self.assertEqual(float(d['kd'][1]),1)
        self.assertEqual(float(d['force'][1]),0)
    def test_legacy_servo_is_bitwise_identical_before_gate(self):
        n=9
        h=torch.linspace(.12,.22,n)
        q=torch.cat((reference_joints(h),torch.zeros(n,2)),1)
        v=torch.randn(n,6)*.3
        r=torch.eye(3).expand(n,3,3)
        request=payload(torch.randn(n,6).tanh(),h,torch.randn(n,4),torch.zeros(n))
        original=legacy_servo(request[:,:12],q,v,r,torch.ones(n,2))
        new=servo(request,q,v,r,torch.ones(n,2))
        for name in original:
            self.assertTrue(torch.equal(original[name],new[name]),name)
    def test_gains_and_reference_keep_same_fifo_latency(self):
        queue=PhysicalStepFIFO(9,15,'cpu')
        queue.set_delays(torch.arange(9),torch.arange(9))
        requests=[]
        for step in range(20):
            request=torch.full((9,15),float(step+1))
            requests.append(request)
            arrived=queue.push(request)
            for row in range(9):
                expected=requests[step-row][row] if step>=row else torch.zeros(15)
                self.assertTrue(torch.equal(arrived[row],expected))
    def test_actual_ppo_update_uses_new_16d_contract(self):
        from compliant_learning import Policy,PPO
        p=Policy()
        o=torch.randn(64,17)
        with torch.no_grad():
            a=p.distribution(o).sample()
        before=p.actor[-1].weight.detach().clone()
        result=PPO(p).update(o,a,-100*a[:,2].square(),torch.ones(64,dtype=torch.bool))
        self.assertGreater(result['actor_steps'],0)
        self.assertFalse(torch.equal(before,p.actor[-1].weight))


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)

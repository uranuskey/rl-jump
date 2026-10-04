import unittest
import torch
from slot_control import feedback,kinematics,target,table,make,NOMINAL


def sample(n=45,enabled=True):
    q=torch.tensor([-1.8885932,2.3222656,-1.8899488,2.3212271]).expand(n,-1).clone()
    c=dict(enabled=torch.full((n,),enabled),correction=torch.zeros(n,6),kp=torch.full((n,),.47))
    return c,q,torch.zeros_like(q)


class SlotContract(unittest.TestCase):
    def test_disabled_and_preapex_preserve_actions(self):
        for cfg,en in [(make('off',enabled=False),True),(make('on',1500,16,25),False)]:
            c,q,v=sample(enabled=en);c['correction'].fill_(.2)
            out,d=feedback(c,table([cfg],45,'cpu'),q,v)
            self.assertTrue(torch.equal(out,torch.tanh(c['correction'])))
            self.assertTrue(bool((d['slot_motor_increment_nm']==0).all()))

    def test_kinematics_matches_documented_failed_pose(self):
        _,q,v=sample();s=kinematics(q,v)
        self.assertAlmostEqual(float(s['height'][0,0]),.098767337,places=6)
        self.assertAlmostEqual(float(s['x'][0,0]),-.038812361,places=6)

    def test_motor_jacobian_matches_finite_difference(self):
        _,q,v=sample(n=1);q=q.double();v=v.double();s=kinematics(q,v)
        for j in range(2):
            altered=q.clone();eps=1e-7
            if j==0:altered[0,0]+=eps;altered[0,1]-=eps
            else:altered[0,1]+=eps
            dx=(kinematics(altered,v)['x'][0,0]-s['x'][0,0])/eps
            self.assertAlmostEqual(float(dx),float(s['jx'][0,0,j]),places=6)

    def test_cad_path_is_not_a_height_only_assumption(self):
        h=torch.tensor([[.09,.18]],dtype=torch.float64);x,_=target(h)
        self.assertAlmostEqual(float(x[0,0]),-NOMINAL[0]['dx_mm']/1000,places=9)
        self.assertGreater(float(abs(x[0,0]-x[0,1])),.0002)

    def test_allocator_preserves_cartesian_direction_and_bounds(self):
        c,q,v=sample();c['correction'][:,0]=2
        out,d=feedback(c,table([make('strong',100000,100,200)],45,'cpu'),q,v)
        self.assertLessEqual(float(out.abs().max()),1.000001)
        expected=kinematics(q,v)['jx']*d['slot_allocated_force_n'][:,:,None]
        torch.testing.assert_close(expected,d['slot_motor_increment_nm'])
        self.assertTrue(torch.equal(out[:,4:],torch.tanh(c['correction'])[:,4:]))
        self.assertTrue(bool((d['slot_allocation']<1).any()))

    def test_force_restores_toward_slot_and_velocity_damps_error(self):
        c,q,v=sample();cfg=table([make('test',100,1,200)],45,'cpu')
        _,a=feedback(c,cfg,q,v)
        self.assertTrue(bool((a['slot_requested_force_n']<0).all()))
        # Exact virtual work: tau dot theta_dot equals Fx dot x_dot.
        v[:]=torch.tensor([.5,-.1,.3,.2]);s=kinematics(q,v)
        _,b=feedback(c,cfg,q,v)
        mv=v.reshape(-1,2,2).cumsum(-1)
        torch.testing.assert_close((b['slot_motor_increment_nm']*mv).sum(-1),b['slot_allocated_force_n']*s['vx'])

    def test_training_batch_shape(self):
        c,q,v=sample(n=512)
        out,_=feedback(c,table([make('test',1500,16,25)],512,'cpu'),q,v)
        self.assertEqual(out.shape,(512,6))

    def test_ppo_actor_and_distribution_use_identical_shifted_mean(self):
        from slot_learning import Policy
        p=Policy();obs=torch.randn(45,17)
        self.assertTrue(torch.equal(p.actor(obs),p.distribution(obs).mean))
        loss=p.distribution(obs).log_prob(p.actor(obs).detach()+.01).mean()
        loss.backward()
        self.assertTrue(all(x.grad is not None and bool(torch.isfinite(x.grad).all()) for x in p.actor.parameters()))

    def test_initial_air_transform_matches_diagnostic_parameter_transform(self):
        from slot_learning import AirShift
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'v7_rs02_24v_impact_spread_probe'))
        from spread_probe import change_parameters
        raw=torch.zeros(45,16)
        expect=change_parameters(raw,[dict(air_delta=-.005,time_delta=0.,kd_ratio=1.)])
        self.assertTrue(torch.equal(AirShift()(raw),expect))


if __name__=='__main__':unittest.main()

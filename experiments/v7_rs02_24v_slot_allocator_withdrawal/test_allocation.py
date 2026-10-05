import unittest
import allocation_paths
import torch
from slot_control import feedback as legacy,table as oldtable
from allocation_control import feedback,table
from allocation_profiles import candidates
class Allocation(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(92);self.n=80
        self.q=torch.tensor([-1.8,2.25,-1.8,2.25]).expand(self.n,-1).clone()+torch.randn(self.n,4)*.04
        self.v=torch.randn(self.n,4)*2
        self.c=dict(enabled=torch.ones(self.n,dtype=torch.bool),kp=torch.full((self.n,),.55),correction=torch.randn(self.n,6)*.12)
        self.p={'name':'base','slot_kx':3750.,'slot_dx':40.,'slot_force_cap_n':50.}
    def test_legacy_equivalence(self):
        p=candidates(self.p)[0];a,da=feedback(self.c,table([p],self.n,'cpu'),self.q,self.v)
        b,db=legacy(self.c,oldtable([self.p],self.n,'cpu'),self.q,self.v)
        self.assertTrue(torch.equal(a,b))
        for k in db:self.assertTrue(torch.equal(da[k],db[k]),k)
    def test_disabled_exact_original(self):
        self.c['enabled'].zero_()
        a,d=feedback(self.c,table([candidates(self.p)[-1]],self.n,'cpu'),self.q,self.v)
        self.assertTrue(torch.equal(a,torch.tanh(self.c['correction'])))
        self.assertEqual(int(torch.count_nonzero(d['slot_motor_increment_nm'])),0)
    def test_window_and_wheels(self):
        for p in candidates(self.p):
            a,d=feedback(self.c,table([p],self.n,'cpu'),self.q,self.v)
            self.assertLessEqual(float(a[:,:4].abs().max()),p['slot_action_bound']+1e-6)
            self.assertTrue(torch.equal(a[:,4:],torch.tanh(self.c['correction'])[:,4:]))
            self.assertTrue(bool(((d['slot_allocation']>=0)&(d['slot_allocation']<=1)).all()))
    def test_release_saturated_authority(self):
        rows=[feedback(self.c,table([p],self.n,'cpu'),self.q,self.v)[1] for p in candidates(self.p)]
        self.assertTrue(bool((rows[-1]['slot_allocated_force_n'].abs()>=rows[0]['slot_allocated_force_n'].abs()-1e-5).all()))
        self.assertGreater(float((rows[-1]['slot_allocated_force_n'].abs()-rows[0]['slot_allocated_force_n'].abs()).max()),5.)
    def test_common_scale_preserves_direction(self):
        _,d=feedback(self.c,table([candidates(self.p)[-1]],self.n,'cpu'),self.q,self.v)
        angles=self.q.reshape(-1,2,2).cumsum(-1);jx=angles.cos()*torch.tensor([.105,.145])
        expected=jx*d['slot_allocated_force_n'][:,:,None]
        self.assertTrue(torch.allclose(expected,d['slot_motor_increment_nm'],atol=1e-6))
    def test_no_extra_gpu_host_decisions(self):
        import inspect
        s=inspect.getsource(feedback)
        for word in ('.item(','.cpu(','.numpy(','.tolist('):self.assertNotIn(word,s)
if __name__=='__main__':unittest.main()

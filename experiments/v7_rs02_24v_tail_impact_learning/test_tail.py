import copy,math,unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import tail_contract as c
from test_rare import rows
class Contract(unittest.TestCase):
    def test_original_gates_identical_functions(self):
        import rare_contract as r
        for name in ('admission','qualification','learning_admission','learning_entry','metrics','wrench_valid','zero_qualified'):
            self.assertIs(getattr(c,name),getattr(r,name))
    def test_zero_below_soft_training_target(self):
        self.assertEqual([c.penalty(v) for v in (0,300,329,330)],[0]*4)
    def test_tail_slope_and_cap(self):
        self.assertEqual(c.penalty(340),-20);self.assertEqual(c.penalty(350),-80)
        self.assertEqual(c.penalty(1000),-100);self.assertLess(c.penalty(341),c.penalty(340))
    def test_nonfinite_and_negative_force_rejected(self):
        for v in (-1,float('nan'),float('inf')):
            with self.assertRaises(AssertionError):c.penalty(v)
    def test_reward_keeps_failures_and_original_terms(self):
        s={'cases':[dict(world=0,reward=-300,gate_tick=1,landing_metrics={'peak_force_n':340})]}
        old=copy.deepcopy(s);o=c.objective(s)
        self.assertEqual(o['rows'][0]['effective_reward'],-320);self.assertEqual(s,old)
    def test_no_plan_is_not_falsely_eligible(self):
        s={'cases':[dict(world=0,reward=-300,gate_tick=-1,landing_metrics={'peak_force_n':0})]}
        self.assertFalse(c.objective(s)['rows'][0]['eligible'])
    def test_world_identity_cannot_shift_penalty(self):
        with self.assertRaises(AssertionError):c.objective({'cases':[dict(world=1,reward=1,gate_tick=1,landing_metrics={'peak_force_n':330})]})
    def test_original_failure_stays_failed(self):
        r=rows(0);r['batch'][0]['strict_admission']['comparisons']['corrected']['checks']['worst_impact']=False
        r['batch'][0]['strict_admission']['comparisons']['corrected']['passed']=False
        r['batch'][0]['strict_admission']['passed']=False;r['batch'][0]['learning_admission']['passed']=False
        self.assertFalse(c.qualification(r));self.assertEqual(c.decision(r,2,0),'FRONTIER')
        self.assertEqual(c.decision(r,2,3),'ENTRY_FAILED')
    def test_budget_is_shared(self):
        self.assertEqual(c.BUDGET,124);self.assertEqual(c.amount(123,False),1)
        self.assertEqual(c.decision(rows(1),124,0),'BUDGET_EXHAUSTED')
    def test_all_pass_advances_at_budget_boundary(self):
        self.assertEqual(c.decision(rows(0),124,0),'ADVANCE')
    def test_zero_torque_still_required(self):
        r=rows(0)
        for row in [r['native']]+r['batch']:
            row['before']=row['after']=0.;row['external_wrench']['max_abs_body_torque_nm']=0.
        self.assertTrue(c.qualification(r))
        r['native']['external_wrench']['max_abs_body_torque_nm']=1e-8
        self.assertFalse(c.qualification(r))
    def test_training_requires_entry_proof(self):
        r=dict(mode='train',role='frontier_train',reuse_preflight=False,level_index=0,before=.35,after=.35,
               global_updates=0,updates=2,resume_optimizer=False,entry_qualification='',entry_result_sha256='sha')
        with self.assertRaises(AssertionError):c.validate_request(r)
    def test_no_training_over_budget(self):
        r=dict(mode='train',role='frontier_train',reuse_preflight=False,level_index=0,before=.35,after=.35,
               global_updates=123,updates=2,resume_optimizer=False,entry_qualification='proof',entry_result_sha256='sha')
        with self.assertRaises(AssertionError):c.validate_request(r)
    def test_audit_rejects_modified_penalty(self):
        import importlib.util
        fake=SimpleNamespace(read=None,sha=lambda p:'hash')
        spec=importlib.util.spec_from_file_location('tail_audit_cpu_test',Path(__file__).with_name('tail_audit.py'))
        module=importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules',{'tail_runtime':fake}):spec.loader.exec_module(module)
        check_training=module.check_training
        s={'cases':[dict(world=i,reward=100.,gate_tick=1,landing_metrics={'peak_force_n':p}) for i,p in enumerate((340.,350.))]}
        obj=c.objective(s)
        sample={'sample_kind':'stochastic_training','summary':'s'}
        req=dict(updates=1,global_updates=0,contract={})
        entry=dict(global_update=1,sample=sample,objective={'file':'o','sha256':'hash'},
                   stats=dict(actor_steps=8,eligible_samples=2,mean_return=.5,max_accepted_kl=.01))
        result=dict(completed_updates=1,actor_hash_before='a',actor_hash_after='b',actor_steps=8,updates=[entry],
                    checkpoint={'path':'ck','sha256':'hash'})
        def read(path):return s if Path(path).name=='s' else obj
        fake.read=read
        with patch.dict('sys.modules',{'rare_row_audit':SimpleNamespace(check_row=lambda *args:None)}):
            self.assertEqual(check_training(Path('.'),req,result,'a'),8)
            obj['rows'][0]['tail_penalty']=0
            with self.assertRaises(AssertionError):check_training(Path('.'),req,result,'a')
if __name__=='__main__':unittest.main()

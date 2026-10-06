import ast,copy,importlib.util,math,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import peak_contract as c
import target_contract as old
from test_target import ANCHOR,ANCHORS,rows as mean_rows,row
HERE=Path(__file__).resolve().parent
CTX={'anchors_native':ANCHORS,'anchors_batch':ANCHORS}
def peak_rows(strength=.25):
    q=mean_rows(False,strength);r=q['native']
    r['metrics']['max_force_n']=1.03*ANCHOR['max_force_n']*1.00025
    r['strict_admission']=c.admission(r['metrics'],ANCHORS,.006)
    r['learning_admission']={'passed':False}
    return q
class Entry(unittest.TestCase):
    def test_final_functions_identical(self):
        for name in ('qualification','admission','learning_admission','metrics','objective','wrench_valid','zero_qualified','near_mean_entry'):
            self.assertIs(getattr(c,name),getattr(old,name))
    def test_peak_route_is_learning_not_qualification(self):
        q=peak_rows();saved=copy.deepcopy(q)
        self.assertFalse(c.qualification(q));self.assertFalse(old.learning_entry(q,CTX))
        e=c.learning_evidence(q,CTX)
        self.assertTrue(e['near_native_peak_entry']);self.assertEqual(e['route'],'near_native_peak')
        self.assertEqual(c.decision(q,CTX,0,0),'LEARN');self.assertEqual(q,saved)
    def test_exact_bound_and_next_float(self):
        q=peak_rows();bound=1.03*ANCHOR['max_force_n']*1.0005
        q['native']['metrics']['max_force_n']=bound;self.assertTrue(c.near_native_peak_entry(q,ANCHORS))
        q['native']['metrics']['max_force_n']=math.nextafter(bound,math.inf);self.assertFalse(c.near_native_peak_entry(q,ANCHORS))
    def test_native_mean_and_all_other_checks_not_relaxed(self):
        for name in ANCHORS:
            for key in c.CHECKS-{'worst_impact'}:
                q=peak_rows();q['native']['strict_admission']['comparisons'][name]['checks'][key]=False
                self.assertFalse(c.near_native_peak_entry(q,ANCHORS),(name,key))
    def test_both_caps_finite_positive_required(self):
        for value in (math.nan,math.inf,-1.,0.):
            a=copy.deepcopy(ANCHORS);a['original_reference']['max_force_n']=value
            self.assertFalse(c.near_native_peak_entry(peak_rows(),a))
        a=copy.deepcopy(ANCHORS);a['original_reference']['max_force_n']*=.99
        self.assertFalse(c.near_native_peak_entry(peak_rows(),a))
    def test_nonfinite_or_negative_peak_rejected(self):
        for value in (math.nan,math.inf,-1.):
            q=peak_rows();q['native']['metrics']['max_force_n']=value
            self.assertFalse(c.near_native_peak_entry(q,ANCHORS))
    def test_all_batches_strict_no_combined_failure(self):
        for native in (False,True):
            q=peak_rows();r=q['native'] if native else q['batch'][0]
            r['metrics']['passed']-=1;self.assertFalse(c.near_native_peak_entry(q,ANCHORS))
        for name in ANCHORS:
            for key in c.CHECKS:
                q=peak_rows();q['batch'][0]['strict_admission']['comparisons'][name]['checks'][key]=False
                self.assertFalse(c.near_native_peak_entry(q,ANCHORS),(name,key))
        q=peak_rows();q['batch'][1]['strict_admission']['passed']=False
        self.assertFalse(c.near_native_peak_entry(q,ANCHORS))
    def test_complete_shapes_order_and_references(self):
        for change in ('remove_batch','repeat','size','native_size','audit','missing_reference','missing_check'):
            q=peak_rows()
            if change=='remove_batch':q['batch'].pop()
            if change=='repeat':q['batch'][1]['repeat']=1
            if change=='size':q['batch'][1]['metrics']['worlds']=511
            if change=='native_size':q['native']['metrics']['worlds']=44
            if change=='audit':q['native']['audits']['status']='FAIL'
            if change=='missing_reference':del q['native']['strict_admission']['comparisons']['original_reference']
            if change=='missing_check':del q['batch'][0]['strict_admission']['comparisons']['original_reference']['checks']['no_rebound']
            self.assertFalse(c.near_native_peak_entry(q,ANCHORS),change)
    def test_preapex_and_wrench_never_relaxed(self):
        for index in range(4):
            q=peak_rows();r=([q['native']]+q['batch'])[index]
            r['strict_admission']['pre_apex_constraint_margin']=False
            self.assertFalse(c.near_native_peak_entry(q,ANCHORS))
            for key in ('max_abs_linear_force_n','max_abs_root_generalized_force'):
                q=peak_rows();r=([q['native']]+q['batch'])[index];r['external_wrench'][key]=1e-9
                self.assertFalse(c.near_native_peak_entry(q,ANCHORS))
    def test_zero_requires_exact_actual_zero(self):
        q=peak_rows(0);self.assertTrue(c.near_native_peak_entry(q,ANCHORS))
        q['native']['external_wrench']['max_abs_body_torque_nm']=1e-9
        self.assertFalse(c.near_native_peak_entry(q,ANCHORS));self.assertFalse(c.qualification(q))
    def test_parent_mean_and_rare_routes_preserved(self):
        q=mean_rows(True,.25)
        self.assertTrue(old.learning_entry(q,CTX));self.assertTrue(c.learning_entry(q,CTX));self.assertFalse(c.near_native_peak_entry(q,ANCHORS))
        q=dict(native=row(45,strength=.25),batch=[row(512,i,bad=int(i==2),strength=.25) for i in (1,2,3)])
        self.assertTrue(old.learning_entry(q,CTX));self.assertTrue(c.learning_entry(q,CTX));self.assertFalse(c.near_native_peak_entry(q,ANCHORS))
    def test_budget_and_single_update_limit(self):
        self.assertEqual((c.BUDGET,c.PRIOR_SHARED_UPDATES,c.MAX_LEVEL_UPDATES,c.EXPLORATION_OFFSET),(92,36,8,22))
        self.assertEqual(c.amount(91),1);self.assertEqual(c.amount(0),1)
        self.assertEqual(c.decision(peak_rows(),CTX,92,0),'BUDGET_EXHAUSTED')
        self.assertEqual(c.decision(peak_rows(),CTX,8,8),'TARGET_REPAIR_LIMIT')
        self.assertEqual(c.decision(mean_rows(False,.25),CTX,92,8),'ADVANCE')
    def test_reuse_only_source_initial_level(self):
        ck={'path':'source','sha256':'s'};r=dict(mode='qualify',level_index=0,global_updates=0,checkpoint=ck,contract={'source_checkpoint':ck})
        self.assertEqual(c.reuse_key(r),'source_failed')
        for k,v in [('mode','train'),('level_index',1),('global_updates',1),('checkpoint',{'path':'new'})]:
            t=copy.deepcopy(r);t[k]=v;self.assertIsNone(c.reuse_key(t))
    def test_sample_and_build_physics_unchanged(self):
        def sample(p):return next(x for x in ast.parse(p.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='sample')
        self.assertEqual(ast.dump(sample(HERE/'peak_worker.py')),ast.dump(sample(HERE.parent/'v7_rs02_24v_target_mean_learning/target_worker.py')))
        text=(HERE/'peak_worker.py').read_text()
        self.assertIn('from balanced_env import BalancedAssistEnv',text);self.assertIn('from balanced_trace_audit import audit',text)
    def test_optimizer_only_declared_controls_changed(self):
        parent=(HERE.parent/'v7_rs02_24v_slot_landing_probe/slot_ppo.py').read_text()
        expected=parent.replace('import copy\n','import copy\nfrom peak_contract import ACTOR_LR,MAX_ACCEPTED_KL\n').replace('actor_lr=5e-5','actor_lr=ACTOR_LR').replace('kl<=.03','kl<=MAX_ACCEPTED_KL').replace('max_accepted_kl>.02','max_accepted_kl>MAX_ACCEPTED_KL*(2/3)').replace('max_accepted_kl<.005','max_accepted_kl<MAX_ACCEPTED_KL/6')
        self.assertEqual((HERE/'peak_ppo.py').read_text(),expected)
        self.assertEqual(c.ACTOR_LR,2.5e-6);self.assertEqual(c.MAX_ACCEPTED_KL,.005)
class Schedule(unittest.TestCase):
    def execute_fake(self,same=False):
        fake=SimpleNamespace(sha=lambda p:str(p),write=lambda *a:None)
        spec=importlib.util.spec_from_file_location('peak_course_test',HERE/'peak_course.py');mod=importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules',{'peak_runtime':fake,'peak_audit':SimpleNamespace(check_qualification=lambda *a:None)}):spec.loader.exec_module(mod)
        ck={'path':'a','sha256':'a','global_update':6};ctx=dict(CTX,source_checkpoint=ck,source_actor_hash='a')
        result=dict(contract=ctx,frozen_sha256='f');calls=[];actor=['a']
        def worker(run,req,result):
            c.validate_request(req);calls.append(copy.deepcopy(req));i=req['level_index'];u=req['global_updates']
            if req['mode']=='train':
                before=actor[0];actor[0]=before if same else 'actor'+str(u+1)
                return Path('t'+str(len(calls))),dict(completed_updates=1,actor_steps=8,actor_hash_before=before,actor_hash_after=actor[0],checkpoint=dict(path=actor[0],sha256=actor[0],global_update=u+1))
            q=peak_rows(req['before']) if (i==0 and u<2) or (i==1 and u<3) else mean_rows(False,req['before'])
            if i==2:
                for r in [q['native']]+q['batch']:
                    r['strict_admission']['passed']=False;r['learning_admission']['passed']=False
                    r['strict_admission']['comparisons']['original_reference']['checks']['all_cases']=False
            return Path('q'+str(len(calls))),dict(actor_hash=actor[0],request=req,qualification=q,qualified=c.qualification(q),learning_entry=c.learning_entry(q,ctx))
        mod.worker=worker;mod.execute(Path('.'),result);return result,calls
    def test_single_update_resume_fresh_advance(self):
        result,calls=self.execute_fake();self.assertTrue(calls[0]['reuse_source_evidence'])
        self.assertFalse(any(r['reuse_source_evidence'] for r in calls[1:]))
        training=[r for r in calls if r['mode']=='train']
        self.assertEqual([r['updates'] for r in training],[1,1,1])
        self.assertEqual([r['resume_optimizer'] for r in training],[False,True,False])
        self.assertEqual(result['completed_updates'],3);self.assertEqual(result['actor_steps'],24)
        self.assertEqual([r['name'] for r in result['qualified_levels']],['uniform250','uniform225'])
        self.assertEqual(result['stop_reason'],'ENTRY_FAILED')
    def test_same_actor_not_retested(self):
        with self.assertRaises(AssertionError):self.execute_fake(True)
if __name__=='__main__':unittest.main()

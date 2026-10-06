import ast,copy,importlib.util,math,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import target_contract as c
import balanced_contract as old
from test_rare import summary
HERE=Path(__file__).resolve().parent
ANCHOR=c.metrics(summary(512,0))
ANCHORS={k:copy.deepcopy(ANCHOR) for k in ('original_reference','corrected_reference')}

def row(n,repeat=None,mean=None,bad=0,strength=.2625):
    s=summary(n,bad)
    if mean is not None:
        for x in s['cases']:x['landing_metrics']['peak_force_n']=mean
    m=c.metrics(s)
    r=dict(metrics=m,strict_admission=c.admission(m,ANCHORS,.006),
        learning_admission=c.learning_admission(s,ANCHORS,.006),before=strength,after=strength,
        external_wrench=dict(min_sampled_ticks=100,max_abs_linear_force_n=0.,
            max_abs_root_generalized_force=0.,max_abs_body_torque_nm=20*strength),audits={'status':'PASS'})
    if repeat is not None:r['repeat']=repeat
    return r

def rows(near=True,strength=.2625):
    mean=1.02*ANCHOR['mean_force_n']*(1.001 if near else .999)
    return dict(native=row(45,strength=strength),batch=[row(512,i,mean=mean,strength=strength) for i in (1,2,3)])

class Entry(unittest.TestCase):
    def test_final_and_row_functions_are_identical(self):
        for key in ('qualification','admission','learning_admission','metrics','objective','wrench_valid','zero_qualified'):
            self.assertIs(getattr(c,key),getattr(old,key))
    def test_near_mean_learning_does_not_promote(self):
        q=rows();saved=copy.deepcopy(q)
        self.assertFalse(c.qualification(q));self.assertFalse(old.learning_entry(q))
        self.assertTrue(c.learning_entry(q,{'anchors_batch':ANCHORS}))
        self.assertEqual(c.decision(q,{'anchors_batch':ANCHORS},0,0),'LEARN')
        self.assertEqual(q,saved)
    def test_mean_bound_at_and_above_limit(self):
        q=rows();limit=1.02*ANCHOR['mean_force_n']*1.0025
        for r in q['batch']:r['metrics']['mean_force_n']=limit
        self.assertTrue(c.near_mean_entry(q,ANCHORS))
        q['batch'][2]['metrics']['mean_force_n']=math.nextafter(limit,math.inf)
        self.assertFalse(c.near_mean_entry(q,ANCHORS))
    def test_both_reference_caps_required(self):
        q=rows();anchors=copy.deepcopy(ANCHORS);anchors['original_reference']['mean_force_n']=300
        self.assertFalse(c.near_mean_entry(q,anchors))
    def test_all_other_checks_preserved_in_new_route(self):
        for name in ANCHORS:
            for key in c.CHECKS-{'mean_impact'}:
                q=rows();q['batch'][1]['strict_admission']['comparisons'][name]['checks'][key]=False
                self.assertFalse(c.near_mean_entry(q,ANCHORS),(name,key))
    def test_missing_check_or_reference_rejected(self):
        q=rows();del q['batch'][0]['strict_admission']['comparisons']['original_reference']
        self.assertFalse(c.near_mean_entry(q,ANCHORS))
        q=rows();del q['batch'][0]['strict_admission']['comparisons']['corrected_reference']['checks']['worst_impact']
        self.assertFalse(c.near_mean_entry(q,ANCHORS))
    def test_native_must_be_strictly_qualified(self):
        for key,value in [('strict_admission',{'passed':False}),('audits',{'status':'FAIL'}),('metrics',{'worlds':44})]:
            q=rows();q['native'][key]=value;self.assertFalse(c.near_mean_entry(q,ANCHORS))
    def test_all_three_ordered_512_required(self):
        q=rows();q['batch'].pop();self.assertFalse(c.near_mean_entry(q,ANCHORS))
        q=rows();q['batch'][1]['repeat']=1;self.assertFalse(c.near_mean_entry(q,ANCHORS))
        q=rows();q['batch'][1]['metrics']['worlds']=511;self.assertFalse(c.near_mean_entry(q,ANCHORS))
    def test_preapex_and_contact_count_rejected(self):
        q=rows();q['batch'][1]['strict_admission']['pre_apex_constraint_margin']=False
        self.assertFalse(c.near_mean_entry(q,ANCHORS))
        q=rows();q['batch'][1]['metrics']['passed']=511;self.assertFalse(c.near_mean_entry(q,ANCHORS))
    def test_nonfinite_negative_force_rejected(self):
        for value in [float('nan'),float('inf'),-1]:
            q=rows();q['batch'][0]['metrics']['mean_force_n']=value;self.assertFalse(c.near_mean_entry(q,ANCHORS))
    def test_new_entry_never_allows_unmeasured_force(self):
        for key in ('max_abs_linear_force_n','max_abs_root_generalized_force'):
            q=rows();q['batch'][0]['external_wrench'][key]=1e-9;self.assertFalse(c.near_mean_entry(q,ANCHORS))
        q=rows();q['batch'][0]['external_wrench']['min_sampled_ticks']=0;self.assertFalse(c.near_mean_entry(q,ANCHORS))
    def test_zero_requires_exact_actual_zero(self):
        q=rows(strength=0);self.assertTrue(c.near_mean_entry(q,ANCHORS))
        q['native']['external_wrench']['max_abs_body_torque_nm']=1e-9
        self.assertFalse(c.near_mean_entry(q,ANCHORS));self.assertFalse(c.qualification(q))
    def test_existing_rare_entry_is_preserved_without_promotion(self):
        q=dict(native=row(45),batch=[row(512,i,bad=1 if i==2 else 0) for i in (1,2,3)])
        self.assertTrue(old.learning_entry(q));self.assertTrue(c.learning_entry(q,{'anchors_batch':ANCHORS}))
        self.assertFalse(c.near_mean_entry(q,ANCHORS));self.assertFalse(c.qualification(q))
    def test_budget_and_local_repair_limit(self):
        q=rows();ctx={'anchors_batch':ANCHORS}
        self.assertEqual(c.BUDGET,98);self.assertEqual(c.BUDGET+c.PRIOR_SHARED_UPDATES,128)
        self.assertEqual(c.amount(97),1);self.assertEqual(c.amount(0),2)
        self.assertEqual(c.decision(q,ctx,98,0),'BUDGET_EXHAUSTED')
        self.assertEqual(c.decision(q,ctx,16,16),'TARGET_REPAIR_LIMIT')
        self.assertEqual(c.decision(rows(False),ctx,98,16),'ADVANCE')
    def test_source_reuse_requires_exact_actor_checkpoint_and_level(self):
        ck={'path':'source','sha256':'source'}
        r=dict(mode='qualify',level_index=0,global_updates=0,checkpoint=ck,contract={'source_checkpoint':ck})
        self.assertEqual(c.reuse_key(r),'source_failed')
        for key,value in [('mode','train'),('global_updates',2),('level_index',1),('checkpoint',{'sha256':'new'})]:
            x=copy.deepcopy(r);x[key]=value;self.assertIsNone(c.reuse_key(x))
    def test_full_rollout_sample_ast_unchanged(self):
        def sample(path):return next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='sample')
        self.assertEqual(ast.dump(sample(HERE/'target_worker.py')),ast.dump(sample(HERE.parent/'v7_rs02_24v_balanced_impact_learning/balanced_worker.py')))
    def test_same_frozen_environment_and_trace_audit_imports(self):
        text=(HERE/'target_worker.py').read_text()
        self.assertIn('from balanced_env import BalancedAssistEnv',text)
        self.assertIn('from balanced_trace_audit import audit',text)
        self.assertIn('EXPLORATION_OFFSET+global_update-1',text)
        self.assertEqual(c.EXPLORATION_OFFSET,16)

class Schedule(unittest.TestCase):
    def execute_fake(self,same_actor=False):
        fake=SimpleNamespace(sha=lambda p:str(p),write=lambda *args:None)
        spec=importlib.util.spec_from_file_location('target_course_test',HERE/'target_course.py');module=importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules',{'target_runtime':fake,'target_audit':SimpleNamespace(check_qualification=lambda *a:None)}):spec.loader.exec_module(module)
        ck={'path':'a','sha256':'a','global_update':16};ctx=dict(source_checkpoint=ck,source_actor_hash='a',anchors_batch=ANCHORS)
        result=dict(contract=ctx,frozen_sha256='f');calls=[];actor=['a']
        def worker(run,req,result):
            c.validate_request(req);calls.append(copy.deepcopy(req));i=req['level_index'];u=req['global_updates']
            if req['mode']=='train':
                before=actor[0];actor[0]=before if same_actor else 'actor'+str(u+2)
                return Path('t'+str(len(calls))),dict(completed_updates=2,actor_steps=16,actor_hash_before=before,
                    actor_hash_after=actor[0],checkpoint=dict(path=actor[0],sha256=actor[0],global_update=u+2))
            near=(i==0 and u<4) or (i==1 and u<6);q=rows(near,strength=req['before'])
            if i==2:
                for r in [q['native']]+q['batch']:
                    r['strict_admission']['passed']=False;r['learning_admission']['passed']=False
                    r['strict_admission']['comparisons']['original_reference']['checks']['all_cases']=False
            return Path('q'+str(len(calls))),dict(actor_hash=actor[0],request=req,qualification=q,
                qualified=c.qualification(q),learning_entry=c.learning_entry(q,ctx))
        module.worker=worker
        module.execute(Path('.'),result)
        return result,calls
    def test_reuse_failure_train_target_resume_then_fresh_at_new_level(self):
        result,calls=self.execute_fake()
        self.assertTrue(calls[0]['reuse_source_evidence']);self.assertFalse(any(x['reuse_source_evidence'] for x in calls[1:]))
        training=[x for x in calls if x['mode']=='train']
        self.assertEqual([x['level_index'] for x in training],[0,0,1])
        self.assertEqual([x['resume_optimizer'] for x in training],[False,True,False])
        self.assertEqual(result['completed_updates'],6);self.assertEqual(result['actor_steps'],48)
        self.assertEqual([x['name'] for x in result['qualified_levels']],['uniform2625','uniform250'])
        self.assertEqual(result['stop_reason'],'ENTRY_FAILED')
    def test_unchanged_actor_cannot_be_retried(self):
        with self.assertRaises(AssertionError):self.execute_fake(True)

if __name__=='__main__':unittest.main()

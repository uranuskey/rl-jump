import ast,copy,math,unittest,importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import balanced_contract as c
from test_rare import rows as parent_rows
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def rows(bad=0,strength=.275):
    r=parent_rows(bad)
    for x in [r['native']]+r['batch']:
        x.update(before=strength,after=strength)
        x['external_wrench']['max_abs_body_torque_nm']=20*strength
    return r
def impact_failure(field='worst_impact'):
    r=rows()
    a=r['batch'][0]['strict_admission'];a['passed']=False
    for v in a['comparisons'].values():v['passed']=False;v['checks'][field]=False
    r['batch'][0]['learning_admission']['passed']=False
    return r
def node(path,name):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    return next(x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name==name)
class Contract(unittest.TestCase):
    def test_final_metric_gates_are_same_functions(self):
        import rare_contract as old
        for k in ('metrics','admission','learning_admission','zero_qualified'):
            self.assertIs(getattr(c,k),getattr(old,k))
    def test_fine_level_is_explicit_only(self):
        self.assertEqual(c.LEVELS[1],('uniform2625',.2625,.2625))
        self.assertEqual(c.checked_levels(.2625,.2625),(.2625,.2625))
        for x in (.261,.276,-.01):
            with self.assertRaises(AssertionError):c.checked_levels(x,x)
    def test_finer_level_same_torque_envelope(self):
        r=rows(strength=.2625);self.assertTrue(c.qualification(r))
        r['native']['external_wrench']['max_abs_body_torque_nm']=5.25006
        self.assertFalse(c.qualification(r))
    def test_zero_needs_exact_actual_zero(self):
        r=rows(strength=0);self.assertTrue(c.qualification(r))
        r['native']['external_wrench']['max_abs_body_torque_nm']=1e-8
        self.assertFalse(c.qualification(r))
    def test_linear_and_root_force_never_allowed(self):
        for key in ('max_abs_linear_force_n','max_abs_root_generalized_force'):
            r=rows();r['native']['external_wrench'][key]=1e-8
            self.assertFalse(c.qualification(r));self.assertFalse(c.learning_entry(r))
    def test_all_three_batches_required(self):
        r=rows();r['batch'].pop()
        self.assertFalse(c.qualification(r));self.assertFalse(c.learning_entry(r))
    def test_small_contact_is_learning_only(self):
        self.assertTrue(c.learning_entry(rows(1)));self.assertFalse(c.qualification(rows(1)))
    def test_mean_and_worst_fail_only_return_to_qualified_frontier(self):
        for k in ('mean_impact','worst_impact'):
            r=impact_failure(k)
            self.assertFalse(c.learning_entry(r));self.assertFalse(c.qualification(r))
            self.assertEqual(c.decision(r,0,0),'FRONTIER')
            self.assertEqual(c.decision(r,0,8),'ENTRY_FAILED')
    def test_other_failures_never_get_impact_exception(self):
        for k in ('all_cases','wheel_height','actual_stroke','no_rebound','stable_interval'):
            self.assertEqual(c.decision(impact_failure(k),0,0),'ENTRY_FAILED')
    def test_common_range_cost_and_tail_preserved(self):
        values=[c.bulk_penalty(x)+c.tail_penalty(x) for x in (0,280,300,310,330,340,350)]
        self.assertEqual(values,[0,0,-40,-60,-100,-120,-180])
        self.assertTrue(all(a>=b for a,b in zip(values,values[1:])))
    def test_bad_force_rejected(self):
        for p in (-1,float('nan'),float('inf')):
            with self.assertRaises(AssertionError):c.bulk_penalty(p)
    def test_original_failed_reward_never_gets_bonus(self):
        s={'cases':[dict(world=0,reward=-300,gate_tick=1,landing_metrics={'peak_force_n':340})]}
        old=copy.deepcopy(s);o=c.objective(s)
        self.assertEqual(o['rows'][0]['effective_reward'],-420)
        self.assertEqual(s,old)
    def test_unexecuted_plan_not_eligible(self):
        s={'cases':[dict(world=0,reward=-300,gate_tick=-1,landing_metrics={'peak_force_n':0})]}
        self.assertFalse(c.objective(s)['rows'][0]['eligible'])
    def test_world_order_bound(self):
        with self.assertRaises(AssertionError):c.objective({'cases':[dict(world=2)]})
    def test_shared_budget_and_short_chunks(self):
        self.assertEqual(c.BUDGET,114);self.assertEqual(c.PRIOR_SHARED_UPDATES,14)
        self.assertEqual(c.amount(0,True),2);self.assertEqual(c.amount(2,False),2);self.assertEqual(c.amount(113,False),1)
        self.assertEqual(c.decision(rows(),114,0),'ADVANCE')
        self.assertEqual(c.decision(impact_failure(),114,0),'BUDGET_EXHAUSTED')
    def test_source_reuse_only_exact_initial_and_failed_source(self):
        ck={'sha256':'old'};r=dict(contract={'source_checkpoint':ck},checkpoint=ck,global_updates=0,role='initial_frontier',level_index=0)
        self.assertEqual(c.reuse_key(r),'source_qualified')
        r.update(role='target',level_index=2);self.assertEqual(c.reuse_key(r),'source_failed')
        r['global_updates']=2;self.assertIsNone(c.reuse_key(r))
        r.update(global_updates=0,checkpoint={'sha256':'new'});self.assertIsNone(c.reuse_key(r))
    def test_training_cannot_claim_evidence_reuse(self):
        r=dict(mode='train',role='frontier_train',reuse_preflight=False,level_index=0,before=.275,after=.275,
               global_updates=0,updates=2,resume_optimizer=False,entry_qualification='proof',entry_result_sha256='sha',
               reuse_source_evidence=True)
        with self.assertRaises(AssertionError):c.validate_request(r)
        r['reuse_source_evidence']=False;self.assertTrue(c.validate_request(r))
        r['updates']=8
        with self.assertRaises(AssertionError):c.validate_request(r)
    def test_whitelist_changes_do_not_change_physics(self):
        old=ROOT/'v7_rs02_24v_zero_assist_curriculum'
        for prior,new in [('zero_env.py','balanced_env.py'),('zero_assist_audit.py','balanced_assist_audit.py'),
                          ('zero_wrench_audit.py','balanced_wrench_audit.py'),('zero_trace_audit.py','balanced_trace_audit.py')]:
            text=(HERE/new).read_text()
            text=text.replace('balanced_contract','zero_contract').replace('BalancedAssistEnv','ZeroAssistEnv')
            text=text.replace('balanced_assist_audit','zero_assist_audit').replace('balanced_wrench_audit','zero_wrench_audit')
            self.assertEqual(ast.dump(ast.parse(text)),ast.dump(ast.parse((old/prior).read_text())))
    def test_trial_and_sample_semantics_unchanged(self):
        old=node(ROOT/'v7_rs02_24v_tail_impact_learning/tail_worker.py','sample')
        text=(HERE/'balanced_worker.py').read_text().replace('balanced_trace_audit','zero_trace_audit')
        new=next(x for x in ast.parse(text).body if isinstance(x,ast.FunctionDef) and x.name=='sample')
        self.assertEqual(ast.dump(old),ast.dump(new))
    def test_actual_wrench_audit_at_new_level(self):
        from balanced_wrench_audit import external_arrays
        z={'sensor_external_body_wrench':np.zeros((2,1,3,6)), 'sensor_external_root_generalized_force':np.zeros((2,1,6)),
           'sensor_external_base_body_id':np.ones((2,1),dtype=int),'assist_wrench':np.zeros((2,1,6))}
        z['sensor_external_body_wrench'][...,1,3]=5.;z['assist_wrench'][...,3]=5.
        self.assertFalse(external_arrays(z,.2625,.2625)['zero_external_wrench'])
        z['sensor_external_body_wrench'][...,2,3]=.001
        with self.assertRaises(AssertionError):external_arrays(z,.2625,.2625)
    def test_zero_wrench_audit_rejects_tiny_body_force(self):
        from balanced_wrench_audit import external_arrays
        z={'sensor_external_body_wrench':np.zeros((2,1,1,6)), 'sensor_external_root_generalized_force':np.zeros((2,1,6)),
           'sensor_external_base_body_id':np.zeros((2,1),dtype=int),'assist_wrench':np.zeros((2,1,6))}
        self.assertTrue(external_arrays(z,0,0)['zero_external_wrench'])
        z['sensor_external_body_wrench'][...,0,3]=1e-9;z['assist_wrench'][...,3]=1e-9
        with self.assertRaises(AssertionError):external_arrays(z,0,0)
    def test_schedule_reuses_old_failure_without_replay_then_learns_at_finer_level(self):
        fake=SimpleNamespace(sha=lambda p:str(p),write=lambda *a:None)
        spec=importlib.util.spec_from_file_location('balanced_course_test',HERE/'balanced_course.py')
        module=importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules',{'balanced_runtime':fake,'balanced_audit':SimpleNamespace(check_qualification=lambda *a:None)}):
            spec.loader.exec_module(module)
        ck={'path':'a','sha256':'a','global_update':0}
        result={'contract':{'source_checkpoint':ck,'source_actor_hash':'a'},'frozen_sha256':'f'}
        calls=[];current=['a']
        def worker(run,request,result):
            calls.append(copy.deepcopy(request));i=request['level_index']
            if request['mode']=='train':
                self.assertEqual(i,1);self.assertEqual(request['updates'],2);self.assertFalse(request['resume_optimizer'])
                current[0]='b'
                return Path('train'),dict(completed_updates=2,actor_steps=16,actor_hash_before='a',actor_hash_after='b',
                    checkpoint={'path':'b','sha256':'b','global_update':2})
            qrows=rows(strength=request['before'])
            if i==2 and current[0]=='a':qrows=impact_failure()
            if i==3:qrows=impact_failure('all_cases')
            return Path('q'+str(len(calls))),dict(actor_hash=current[0],request=request,
                qualified=c.qualification(qrows),learning_entry=c.learning_entry(qrows),qualification=qrows)
        module.worker=worker;module.execute(Path('.'),result)
        self.assertEqual(result['stop_reason'],'ENTRY_FAILED');self.assertEqual(result['completed_updates'],2)
        self.assertEqual(result['deepest_qualified'],'uniform250')
        self.assertEqual([r['reuse_source_evidence'] for r in calls],[True,False,True,False,False,False])
        self.assertEqual(len([r for r in calls if r['mode']=='train']),1)
if __name__=='__main__':unittest.main()

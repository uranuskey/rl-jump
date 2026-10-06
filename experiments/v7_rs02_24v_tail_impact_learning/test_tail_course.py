"""No simulator: prove fallback cannot waive entry or re-test the same failed actor."""
import unittest,tempfile,importlib.util,contextlib,io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tail_contract as c
from test_rare import rows

def make_rows(strength,kind='pass'):
    r=rows(1 if kind=='learn' else 0)
    for row in [r['native']]+r['batch']:
        row['before']=row['after']=strength;row['external_wrench']['max_abs_body_torque_nm']=20*strength
    if kind=='impact':
        a=r['batch'][0]['strict_admission'];a['passed']=False
        k=next(iter(a['comparisons']));a['comparisons'][k]['passed']=False;a['comparisons'][k]['checks']['worst_impact']=False
        r['batch'][0]['learning_admission']['passed']=False
    return r

class Course(unittest.TestCase):
    def exercise(self,fail_target=1,fail_always=False,frontier_fails=False,unchanged=False,learn=False,initial_fails=False):
        with tempfile.TemporaryDirectory() as temp:
            runtime=SimpleNamespace(HERE=Path(temp),write=lambda *args:None,sha=lambda p:str(p))
            audit=SimpleNamespace(check_qualification=lambda *args:None)
            spec=importlib.util.spec_from_file_location('tail_course_fake',Path(__file__).with_name('tail_course.py'))
            mod=importlib.util.module_from_spec(spec)
            with patch.dict('sys.modules',{'tail_runtime':runtime,'tail_audit':audit}):spec.loader.exec_module(mod)
            calls=[];state={'actor':'a','failures':0}
            def worker(run,req,result):
                c.validate_request(req);calls.append(req.copy());folder=Path(temp)/str(len(calls))
                if req['mode']=='train':
                    old=state['actor'];state['actor']=old if unchanged else 'a'+str(len(calls))
                    return folder,dict(request=req,actor_hash_before=old,actor_hash_after=state['actor'],
                        completed_updates=req['updates'],actor_steps=8*req['updates'],
                        checkpoint={'path':'ck'+str(len(calls)),'sha256':'sha','global_update':req['global_updates']+req['updates']})
                kind='pass'
                if req['role']=='initial_frontier' and initial_fails:kind='impact'
                if req['role']=='target' and req['level_index']==fail_target and (fail_always or state['failures']==0):
                    state['failures']+=1;kind='learn' if learn else 'impact'
                if req['role']=='frontier_check' and frontier_fails:kind='impact'
                r=make_rows(req['before'],kind)
                return folder,dict(request=req,actor_hash=state['actor'],qualification=r,qualified=c.qualification(r),learning_entry=c.learning_entry(r))
            mod.worker=worker
            r=dict(contract={'source_checkpoint':{'path':'seed','sha256':'seed'},'source_actor_hash':'a'},frozen_sha256='f')
            with contextlib.redirect_stdout(io.StringIO()):mod.execute(Path(temp),r)
            return calls,r
    def test_new_source_must_requalify_before_learning(self):
        calls,r=self.exercise(initial_fails=True)
        self.assertEqual(len(calls),1);self.assertEqual(r['stop_reason'],'INITIAL_FRONTIER_FAILED')
    def test_first_failed_lower_level_returns_to_qualified_frontier(self):
        calls,r=self.exercise()
        self.assertEqual([x['role'] for x in calls[:6]],['initial_frontier','frontier_train','target','frontier_check','frontier_train','target'])
        self.assertEqual(calls[4]['updates'],8);self.assertTrue(calls[4]['resume_optimizer'])
        self.assertEqual(r['completed_updates'],10);self.assertEqual(r['actor_steps'],80)
        self.assertTrue(r['zero_assistance_qualified'])
    def test_failed_frontier_never_trains(self):
        calls,r=self.exercise(frontier_fails=True)
        self.assertEqual(len(calls),4);self.assertEqual(r['completed_updates'],2)
        self.assertEqual(r['stop_reason'],'FRONTIER_RECHECK_FAILED')
    def test_current_actor_strict_proof_reused_without_physics_retry(self):
        calls,r=self.exercise(fail_target=2)
        checks=[x for x in calls if x['role']=='frontier_check']
        self.assertFalse(checks)
        train=[x for x in calls if x['role']=='frontier_train'][1]
        self.assertEqual(train['level_index'],1);self.assertFalse(train['resume_optimizer'])
    def test_identical_actor_never_retries(self):
        with self.assertRaises(AssertionError):self.exercise(unchanged=True)
    def test_bounded_three_returns_preserve_failure(self):
        calls,r=self.exercise(fail_always=True)
        self.assertEqual(r['stop_reason'],'ENTRY_FAILED');self.assertEqual(r['completed_updates'],26)
        self.assertEqual(len([x for x in calls if x['role']=='frontier_check']),3)
    def test_original_learning_route_stays_available(self):
        calls,r=self.exercise(learn=True)
        self.assertEqual(calls[3]['role'],'target_train');self.assertEqual(calls[3]['updates'],2)
        self.assertFalse(calls[3]['resume_optimizer']);self.assertEqual(r['completed_updates'],4)
if __name__=='__main__':unittest.main()

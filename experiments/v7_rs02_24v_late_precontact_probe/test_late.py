import ast,copy,json,sys,unittest
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import late_contract as c
import horizon_contract as parent

def archived():
    base=HERE.parent/'v7_rs02_24v_frontier_admission_recovery/runs'
    for p in (base/'course_01/level_01_u0003_target/result.json',base/'parent_review_01/closure_20261007_bundle/level_01_u0003_target/result.json'):
        if p.exists():return json.loads(p.read_text('utf-8-sig'))
    raise AssertionError('Completed failed source JSON required')

def request(index=0,level=0):
    old=archived()['request']['contract'];source=old['profile']
    contract=dict(source_profile=source,profile=c.profile(source,index),source_checkpoint=dict(path='source',sha256='source'),source_actor_hash='actor')
    return dict(mode='qualify',role='candidate' if level==0 else 'advance',global_updates=0,reuse_source_evidence=False,reuse_preflight=False,
        candidate_index=index,level_index=level,before=c.LEVELS[level][1],after=c.LEVELS[level][2],checkpoint=contract['source_checkpoint'],
        contract=contract,controller_id=c.controller_id('actor',contract['profile']))

class Gates(unittest.TestCase):
    def test_all_final_functions_are_identical(self):
        for name in ('qualification','admission','learning_admission','metrics','wrench_valid','zero_qualified'):
            self.assertIs(getattr(c,name),getattr(parent,name))
    def test_only_timing_and_name_change(self):
        source=archived()['request']['contract']['profile'];original=copy.deepcopy(source)
        for i,h in enumerate((.0275,.025,.020)):
            p=c.profile(source,i);self.assertEqual(p['horizon_s'],h)
            self.assertEqual({k for k in p if p[k]!=source[k]},{'horizon_s','name'})
            self.assertEqual(p['extra_limit_m'],.003);self.assertEqual(p['slot_action_bound'],1.5)
        self.assertEqual(source,original)
    def test_new_controller_ids_and_no_inherited_qualification(self):
        old=archived();source=old['request']['contract']['profile'];actor=old['actor_hash']
        ids={c.controller_id(actor,c.profile(source,i)) for i in range(3)}
        self.assertEqual(len(ids),3);self.assertNotIn(c.controller_id(actor,source),ids)
        self.assertFalse(c.qualification(old['qualification']))
        self.assertFalse(set(c.HORIZONS).intersection({.030,.035,.040,.045}))
    def test_sampling_function_exactly_inherited(self):
        tree=ast.parse((HERE/'late_worker.py').read_text('utf-8'))
        imports=[n for n in tree.body if isinstance(n,ast.ImportFrom) and n.module=='recovery_worker']
        self.assertTrue(any(any(x.name=='sample' and x.asname is None for x in n.names) for n in imports))
        self.assertFalse(any(isinstance(n,ast.FunctionDef) and n.name=='sample' for n in tree.body))
    def test_physics_environment_constructor_unchanged(self):
        def call(path):
            tree=ast.parse(path.read_text('utf-8'))
            return next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='BalancedAssistEnv')
        self.assertEqual(ast.dump(call(HERE/'late_worker.py')),ast.dump(call(HERE.parent/'v7_rs02_24v_frontier_admission_recovery/recovery_worker.py')))
    def test_reject_undeclared_profile(self):
        r=request();r['contract']['profile']['horizon_s']=.05
        with self.assertRaises(AssertionError):c.validate_request(r)
    def test_reject_inherited_profile_identity(self):
        r=request();r['controller_id']=c.controller_id('actor',r['contract']['source_profile'])
        with self.assertRaises(AssertionError):c.validate_request(r)
    def test_reject_ppo_or_reuse(self):
        for key,value in [('mode','train'),('global_updates',1),('reuse_source_evidence',True)]:
            r=request();r[key]=value
            with self.assertRaises(AssertionError):c.validate_request(r)
    def test_levels_and_budget(self):
        self.assertEqual(c.LEVELS,parent.LEVELS);self.assertEqual(c.LEVELS[0][1],.2375)
        self.assertEqual((c.PRIOR_SHARED_UPDATES,c.REMAINING_SHARED_UPDATES),(44,84))
    def test_failed_descent_stops_selected_profile(self):
        self.assertEqual(c.transition(0,1,False),('TARGET_FAILED',0,1))
        self.assertEqual(c.transition(2,0,False),('NO_PROFILE_QUALIFIED',2,0))

class Ledger(unittest.TestCase):
    def fixture(self,answers):
        files={};index=level=0;events=[];promotions=[];selected=None
        for passed in answers:
            req=request(index,level);folder=Path(f'mock/p{index}_l{level}')
            q=dict(request=req,qualified=passed,actor_hash='actor',controller_id=req['controller_id'])
            files[str(folder)]=q
            event=dict(mode='qualify',folder=str(folder),result_sha256=str(folder/'result.json'))
            if passed:
                selected=index;promotions.append(dict(name=c.LEVELS[level][0],before=req['before'],after=req['after'],profile=req['contract']['profile'],
                    checkpoint=req['checkpoint'],actor_hash='actor',controller_id=req['controller_id'],at_global_update=0,
                    qualification_folder=str(folder),qualification_result_sha256=event['result_sha256']))
            action,index,level=c.transition(index,level,passed);event['decision']=action;events.append(event)
        d=dict(contract={},events=events,qualified_levels=promotions,selected_candidate_index=selected,stop_reason=action,
            completed_updates=0,actor_steps=0,new_physical_trials=1581*len(events),zero_assistance_qualified=c.zero_qualified(promotions),
            deepest_qualified=promotions[-1]['name'] if promotions else None)
        files['mock']=d
        rt=SimpleNamespace(checked=lambda p:files[str(p)],contract=lambda:{},sha=str,verify=lambda:'f',now=lambda:'now',write=lambda *args:None)
        tree=ast.parse((HERE/'late_audit.py').read_text('utf-8'));func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='audit')
        ns=dict(rt=rt,Path=Path,LEVELS=c.LEVELS,transition=c.transition,zero_qualified=c.zero_qualified,check_qualification=lambda *args:None)
        exec(compile(ast.Module(body=[func],type_ignores=[]),'audit','exec'),ns)
        return ns,files,d
    def test_full_replay_and_zero(self):
        ns,_,d=self.fixture([False]+[True]*len(c.LEVELS));out=ns['audit'](Path('mock'))
        self.assertTrue(out['zero_assistance_qualified']);self.assertEqual(out['completed_updates'],0)
    def test_all_candidates_fail(self):
        ns,_,_=self.fixture([False,False,False]);self.assertEqual(ns['audit'](Path('mock'))['stop_reason'],'NO_PROFILE_QUALIFIED')
    def test_post_selection_failure_is_final(self):
        ns,_,_=self.fixture([True,False]);self.assertEqual(ns['audit'](Path('mock'))['stop_reason'],'TARGET_FAILED')
    def test_reject_duplicate_or_wrong_order(self):
        ns,_,d=self.fixture([False,False,False]);d['events'][1]=copy.deepcopy(d['events'][0])
        with self.assertRaises(AssertionError):ns['audit'](Path('mock'))
    def test_reject_false_promotion(self):
        ns,_,d=self.fixture([True,False]);d['events'][-1]['decision']='ADVANCE'
        with self.assertRaises(AssertionError):ns['audit'](Path('mock'))
    def test_reject_false_ppo_claim(self):
        ns,_,d=self.fixture([False,False,False]);d['completed_updates']=1
        with self.assertRaises(AssertionError):ns['audit'](Path('mock'))

if __name__=='__main__':unittest.main()

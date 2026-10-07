"""Gate preservation, refusal tests, and an independent mocked ledger replay."""
import ast,copy,json,sys,unittest
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import midpoint_contract as c
import boundary_contract as parent

def node(path,name):
    return next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name==name)

def archived():
    base=HERE.parent/'v7_rs02_24v_slot_boundary_recovery/runs'
    for path in [base/'course_01/level_00_u0001_frontier_check/result.json',
                 base/'parent_review_01/frontier25_u1_bundle/level_00_u0001_frontier_check/result.json']:
        if path.exists():return json.loads(path.read_text(encoding='utf-8-sig'))
    raise AssertionError('Preserved source qualification evidence is required')

class Gates(unittest.TestCase):
    def test_final_and_learning_functions_identical(self):
        for name in ('qualification','admission','metrics','learning_admission','objective','zero_qualified',
                     'wrench_valid','checked_levels','original_learning_entry','near_mean_entry','near_native_peak_entry',
                     'learning_entry','learning_evidence','impact_only_failure'):
            self.assertEqual(ast.dump(node(HERE/'midpoint_gates.py',name)),ast.dump(node(HERE.parent/'v7_rs02_24v_slot_boundary_recovery/boundary_contract.py',name))) if hasattr(getattr(c,name),'__code__') and getattr(c,name).__module__=='midpoint_gates' else self.assertIs(getattr(c,name),getattr(parent,name))
        self.assertEqual(c.LEVELS[0],('uniform23125',0.23125,0.23125))
        self.assertEqual(c.LEVELS[1],('uniform225',0.225,0.225))
        for level in c.LEVELS:self.assertTrue(c.checked_levels(level[1],level[2]))
    def test_source_failed_remains_failed_but_has_existing_entry(self):
        r=archived();rows=r['qualification'];contract=r['request']['contract']
        self.assertFalse(c.qualification(rows));self.assertTrue(c.learning_entry(rows,contract))
        self.assertEqual(c.learning_evidence(rows,contract),r['learning_evidence'])
        self.assertEqual(c.frontier_action(rows,contract,0,0,False),'TRAIN_BOUNDED_RECOVERY')
    def test_native_peak_allowance_not_expanded(self):
        r=archived();rows=copy.deepcopy(r['qualification']);contract=r['request']['contract']
        cap=1.03*contract['anchors_native']['corrected_reference']['max_force_n']*(1+c.NATIVE_PEAK_ENTRY_RELATIVE_MARGIN)
        rows['native']['metrics']['max_force_n']=cap+1e-6
        self.assertFalse(c.learning_entry(rows,contract))
        self.assertEqual(c.frontier_action(rows,contract,0,0,False),'RECOVERY_ENTRY_FAILED')
    def test_no_combination_of_native_and_batch_misses(self):
        r=archived();rows=copy.deepcopy(r['qualification']);contract=r['request']['contract']
        rows['batch'][0]['strict_admission']['passed']=False
        rows['batch'][0]['strict_admission']['comparisons']['corrected_reference']['checks']['mean_impact']=False
        self.assertFalse(c.learning_entry(rows,contract))
    def test_budget_bounds(self):
        self.assertEqual((c.PRIOR_SHARED_UPDATES,c.BUDGET,c.EXPLORATION_OFFSET),(53,75,39))
        r=archived();rows=r['qualification'];contract=r['request']['contract']
        self.assertEqual(c.frontier_action(rows,contract,75,0,False),'BUDGET_EXHAUSTED')
        self.assertEqual(c.frontier_action(rows,contract,0,8,False),'FRONTIER_RECOVERY_LIMIT')
        self.assertEqual((c.ACTOR_LR,c.MAX_ACCEPTED_KL),(2.5e-6,.005))
    def test_sampling_and_physics_build_unchanged(self):
        old=HERE.parent/'v7_rs02_24v_frontier_admission_recovery/recovery_worker.py'
        for name in ('build','sample'):
            self.assertEqual(ast.dump(node(old,name)).replace('recovery_frozen_sha256','midpoint_frozen_sha256').replace('boundary_env','midpoint_env').replace('boundary_trace_audit','midpoint_trace_audit'),
                             ast.dump(node(HERE/'midpoint_worker.py',name)))
    def test_failed_source_reuse_only_initial(self):
        ck=dict(path='s',sha256='s');r=dict(mode='qualify',role='initial_recovery',global_updates=0,level_index=0,
            checkpoint=ck,contract=dict(source_checkpoint=ck))
        self.assertIsNone(c.reuse_key(r))
        r.update(role='target',level_index=1);self.assertEqual(c.reuse_key(r),'source_failed')
        r['level_index']=0;self.assertIsNone(c.reuse_key(r))
        r['role']='frontier_recovery_check';self.assertIsNone(c.reuse_key(r))
        r['role']='initial_recovery';r['global_updates']=1;self.assertIsNone(c.reuse_key(r))

class Ledger(unittest.TestCase):
    def simulate(self,scenario,budget=75):
        files={};requests=[];run=Path('mock_run');actor='source';train_count=0
        def write(path,obj):files[str(path)]=copy.deepcopy(obj)
        def read(path):return copy.deepcopy(files[str(path)])
        def digest(path):return 'sha:'+str(path)
        contract=dict(source_checkpoint=dict(path='source',sha256='source'),source_actor_hash='source')
        rt=SimpleNamespace(write=write,read=read,sha=digest,checked=lambda path:read(path/'result.json'),
                           contract=lambda:contract,verify=lambda:'f',now=lambda:'mock')
        def worker(run,req,result):
            nonlocal actor,train_count
            c.validate_request(req);requests.append(copy.deepcopy(req))
            folder=run/f"level_{req['level_index']:02d}_u{req['global_updates']:04d}_{req['role']}"
            self.assertNotIn(str(folder/'result.json'),files)
            if req['mode']=='qualify':
                qualified,learnable=scenario(req,train_count)
                rows=dict(qualified=qualified,learnable=learnable)
                r=dict(request=req,actor_hash=actor,qualified=qualified,learning_entry=learnable,qualification=rows)
            else:
                train_count+=1;new='actor'+str(train_count)
                r=dict(request=req,actor_hash_before=actor,actor_hash_after=new,actor_steps=8,completed_updates=1,
                    checkpoint=dict(path=new,sha256=new));actor=new
            write(folder/'request.json',req);write(folder/'result.json',r);return folder,r
        def choice(rows,contract,total,returns,local=0):
            if rows['qualified']:return 'ADVANCE'
            if total>=budget:return 'BUDGET_EXHAUSTED'
            if rows['learnable']:return 'LEARN' if local<8 else 'TARGET_REPAIR_LIMIT'
            return 'FRONTIER' if returns<8 else 'ENTRY_FAILED'
        def route(rows,contract,total,used,must):
            if rows['qualified'] and not must:return 'FRONTIER_READY'
            if total>=budget:return 'BUDGET_EXHAUSTED'
            if used>=8:return 'FRONTIER_RECOVERY_LIMIT'
            if rows['qualified']:return 'TRAIN_QUALIFIED_FRONTIER'
            return 'TRAIN_BOUNDED_RECOVERY' if rows['learnable'] else 'RECOVERY_ENTRY_FAILED'
        # Full level list is used: the fixture has strict passes after its targeted failures.
        ns=dict(rt=rt,worker=worker,print=lambda *a,**k:None,check_qualification=lambda *a:None,check_training=lambda *a:8,
            LEVELS=c.LEVELS,BUDGET=budget,PRIOR_SHARED_UPDATES=53,OBJECTIVE=c.OBJECTIVE,Path=Path,
            MAX_FRONTIER_UPDATES=8,validate_request=c.validate_request,decision=choice,frontier_action=route,
            amount=c.amount,reuse_key=c.reuse_key,zero_qualified=c.zero_qualified)
        for filename,function in [('midpoint_course.py','execute'),('midpoint_audit.py','audit')]:
            exec(compile(ast.Module(body=[node(HERE/filename,function)],type_ignores=[]),filename,'exec'),ns)
        result=dict(contract=contract,frozen_sha256='f',final_frozen_sha256='f',status='COMPLETED',zero_assistance_qualified=False)
        ns['execute'](run,result);write(run/'result.json',result)
        write(run/'exit_receipt.json',dict(process_exited=True,exit_code=0))
        return ns,run,files,requests,result
    @staticmethod
    def recovering(req,count):
        if req['role']=='initial_recovery':return False,True
        if req['level_index']==0 and count==2:return False,True
        if req['role']=='target' and req['level_index']==1 and count==1:return False,False
        return True,True
    def test_strict_recovery_precedes_each_descent_and_audit_accepts(self):
        ns,run,files,requests,result=self.simulate(self.recovering)
        out=ns['audit'](run);self.assertEqual(out['completed_updates'],3)
        roles=[r['role'] for r in requests]
        self.assertEqual(roles[:8],['initial_recovery','frontier_recovery_train','frontier_recovery_check','target',
            'frontier_train','frontier_recovery_check','frontier_recovery_train','frontier_recovery_check'])
        self.assertFalse(requests[1]['resume_optimizer'])
        self.assertEqual(result['qualified_levels'][0]['at_global_update'],1)
    def test_audit_rejects_missing_strict_recovery_check(self):
        ns,run,files,requests,result=self.simulate(self.recovering)
        result['events']=[e for j,e in enumerate(result['events']) if j!=3]
        files[str(run/'result.json')]=result
        with self.assertRaises(AssertionError):ns['audit'](run)
    def test_audit_rejects_false_ready_route(self):
        ns,run,files,requests,result=self.simulate(self.recovering)
        result['events'][1]['decision']='FRONTIER_READY';files[str(run/'result.json')]=result
        with self.assertRaises(AssertionError):ns['audit'](run)
    def test_audit_rejects_optimizer_reset(self):
        ns,run,files,requests,result=self.simulate(self.recovering)
        event=next(e for e in result['events'] if e['mode']=='train');folder=Path(event['folder'])
        files[str(folder/'request.json')]['resume_optimizer']=True
        files[str(folder/'result.json')]['request']['resume_optimizer']=True
        with self.assertRaises(AssertionError):ns['audit'](run)
    def test_recovery_limit_is_enforced_without_descent(self):
        ns,run,files,requests,result=self.simulate(lambda *a:(False,True))
        self.assertEqual(result['completed_updates'],8);self.assertEqual(result['stop_reason'],'FRONTIER_RECOVERY_LIMIT')
        self.assertTrue(all(r['level_index']==0 for r in requests));ns['audit'](run)
    def test_rejected_recovery_entry_has_no_training(self):
        ns,run,files,requests,result=self.simulate(lambda *a:(False,False))
        self.assertEqual(result['stop_reason'],'RECOVERY_ENTRY_FAILED');self.assertEqual(len(requests),1);ns['audit'](run)
    def test_shared_budget_stops_recovery(self):
        ns,run,files,requests,result=self.simulate(lambda *a:(False,True),budget=2)
        self.assertEqual(result['completed_updates'],2);self.assertEqual(result['stop_reason'],'BUDGET_EXHAUSTED');ns['audit'](run)


    def test_selected_profile_source_gates_reused_then_fresh_training(self):
        def scenario(req,count):
            if req['level_index']==1 and count==0:return False,False
            return True,True
        ns,run,files,requests,result=self.simulate(scenario)
        out=ns['audit'](run);self.assertEqual(out['completed_updates'],1)
        self.assertEqual([r['role'] for r in requests[:5]],['initial_recovery','target','frontier_train','frontier_recovery_check','target'])
        self.assertFalse(requests[0]['reuse_source_evidence']);self.assertTrue(requests[1]['reuse_source_evidence'])
        self.assertFalse(requests[2]['resume_optimizer'])
        self.assertTrue(all(not r['reuse_source_evidence'] for r in requests[2:]))
        self.assertEqual(requests[2]['before'],0.23125);self.assertEqual(requests[4]['before'],0.225)

class MidpointPreservation(unittest.TestCase):
    def test_environment_and_force_audits_only_import_the_new_declaration(self):
        old=HERE.parent/'v7_rs02_24v_slot_boundary_recovery'
        for suffix in ('env','assist_audit','wrench_audit','trace_audit','row_audit'):
            s=(old/('boundary_'+suffix+'.py')).read_text('utf-8')
            for a,b in [('boundary_contract','midpoint_gates'),('boundary_trace_audit','midpoint_trace_audit'),('boundary_assist_audit','midpoint_assist_audit'),('boundary_wrench_audit','midpoint_wrench_audit')]:s=s.replace(a,b)
            self.assertEqual(ast.dump(ast.parse(s)),ast.dump(ast.parse((HERE/('midpoint_'+suffix+'.py')).read_text('utf-8'))))
    def test_one_midpoint_and_no_exhausted_frontier_training(self):
        self.assertEqual(c.LEVELS[0][1],(.2375+.225)/2)
        self.assertTrue(all(before<.2375 and before==after for _,before,after in c.LEVELS))
        self.assertEqual(c.checked_levels(.23125,.23125),(.23125,.23125))
        with self.assertRaises(AssertionError):c.checked_levels(.232,.232)
    def test_source_failure_uses_cpu_only_without_qualification_transfer(self):
        ck=dict(path='p',sha256='h');r=dict(mode='qualify',role='target',global_updates=0,level_index=1,checkpoint=ck,contract=dict(source_checkpoint=ck))
        self.assertEqual(c.reuse_key(r),'source_failed')
        r.update(role='initial_recovery',level_index=0);self.assertIsNone(c.reuse_key(r))
        r.update(role='target',level_index=1,global_updates=1);self.assertIsNone(c.reuse_key(r))

if __name__=='__main__':unittest.main()

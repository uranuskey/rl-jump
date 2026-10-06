"""Contract and execution-order tests; no GPU physics."""
import ast,copy,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import boundary_contract as c
import peak_contract as p

def node(path,name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)

class BoundaryTests(unittest.TestCase):
    def test_original_gate_functions_identical(self):
        for n in ('admission','metrics','learning_admission','objective','zero_qualified'):self.assertIs(getattr(c,n),getattr(p,n))
    def test_declared_levels(self):
        self.assertEqual(c.LEVELS[:3],[('uniform250',.25,.25),('uniform2375',.2375,.2375),('uniform225',.225,.225)])
        self.assertEqual(c.BUDGET,89);self.assertEqual(c.PRIOR_SHARED_UPDATES,39);self.assertEqual(c.EXPLORATION_OFFSET,25)
        with self.assertRaises(AssertionError):c.checked_levels(.239,.239)
    def test_wrench_bound_not_relaxed(self):
        row=dict(before=.2375,after=.2375,external_wrench=dict(min_sampled_ticks=2000,max_abs_linear_force_n=0.,max_abs_root_generalized_force=0.,max_abs_body_torque_nm=4.75004))
        self.assertTrue(c.wrench_valid(row));row['external_wrench']['max_abs_body_torque_nm']=4.75006;self.assertFalse(c.wrench_valid(row))
        row.update(before=0.,after=0.);row['external_wrench']['max_abs_body_torque_nm']=1e-12;self.assertFalse(c.wrench_valid(row))
    def test_scalar_boundary(self):
        import numpy as np
        h=np.float32(.1420000046491623)
        self.assertEqual(float((h-np.float32(.09))/np.float32(.002)),26.)
        self.assertEqual(float((h-np.float32(.09))*(np.float32(1)/np.float32(.002))),25.999998092651367)
    def test_trace_assertions_unchanged(self):
        old=HERE.parent/'v7_rs02_24v_balanced_impact_learning/balanced_trace_audit.py'
        oldnode=node(old,'slot_audit');newnode=node(HERE/'boundary_trace_audit.py','slot_audit')
        oldasserts=[ast.dump(n) for n in ast.walk(oldnode) if isinstance(n,ast.Assert)]
        newasserts=[ast.dump(n) for n in ast.walk(newnode) if isinstance(n,ast.Assert)]
        self.assertEqual(oldasserts,newasserts)
        # Exactly the index expression is replaced; all other old operations remain.
        replacement=ast.parse('index=np.clip(np.floor((hc-np.float32(.09))*(np.float32(1.)/np.float32(.002))).astype(int),0,len(rows)-2)').body[0].value
        class Fix(ast.NodeTransformer):
            def visit_Assign(self,n):
                if any(isinstance(t,ast.Name) and t.id=='index' for t in n.targets):n.value=replacement
                return self.generic_visit(n)
        self.assertEqual(ast.dump(Fix().visit(oldnode)),ast.dump(newnode))
    def test_environment_and_wrench_function_bodies_identical(self):
        base=HERE.parent/'v7_rs02_24v_balanced_impact_learning'
        for old,new,name in [('balanced_env.py','boundary_env.py','BalancedAssistEnv'),('balanced_assist_audit.py','boundary_assist_audit.py','audit_arrays'),('balanced_wrench_audit.py','boundary_wrench_audit.py','external_arrays')]:
            a=ast.dump(node(base/old,name)).replace('balanced_contract','boundary_contract')
            self.assertEqual(a,ast.dump(node(HERE/new,name)))
    def test_sampling_and_build_same_physics(self):
        base=HERE.parent/'v7_rs02_24v_target_peak_learning/peak_worker.py'
        for name in ('sample','build'):
            a=ast.dump(node(base,name)).replace('balanced_trace_audit','boundary_trace_audit').replace('balanced_env','boundary_env').replace('peak_frozen_sha256','boundary_frozen_sha256')
            self.assertEqual(a,ast.dump(node(HERE/'boundary_worker.py',name)))
    def test_ppo_limits_preserved(self):
        self.assertEqual(c.ACT0R_LR if hasattr(c,'ACT0R_LR') else c.ACTOR_LR,p.ACTOR_LR)
        self.assertEqual(c.MAX_ACCEPTED_KL,.005)
        for n in (0,88):self.assertEqual(c.amount(n),1)
        with self.assertRaises(AssertionError):c.amount(89)
    def test_native_routes_body_preserved(self):
        for name,old in [('near_mean_entry','v7_rs02_24v_target_mean_learning/target_contract.py'),('near_native_peak_entry','v7_rs02_24v_target_peak_learning/peak_contract.py')]:
            self.assertEqual(ast.dump(node(HERE.parent/old,name)),ast.dump(node(HERE/'boundary_contract.py',name)))
    def test_source_reuse_only_initial_qualified(self):
        source=dict(path='source',sha256='sha');r=dict(mode='qualify',role='initial_frontier',level_index=0,global_updates=0,checkpoint=source,contract=dict(source_checkpoint=source))
        self.assertEqual(c.reuse_key(r),'source_qualified')
        for level in (1,2):
            r['level_index']=level;r['role']='target';self.assertIsNone(c.reuse_key(r))
    def test_course_prime_and_frontier_order(self):
        from types import SimpleNamespace
        rt=SimpleNamespace(write=lambda *a:None,sha=lambda p:str(p))
        log=[];actor='source';traincount=0
        def worker(run,req,result):
            nonlocal actor,traincount
            log.append(copy.deepcopy(req));folder=run/(str(len(log))+'_'+req['role'])
            if req['mode']=='qualify':
                q=req['role']!='target' or traincount>1
                return folder,dict(request=req,actor_hash=actor,qualified=q,learning_entry=q,qualification=dict(qualified=q))
            traincount+=1;new='actor'+str(traincount)
            out=dict(actor_hash_before=actor,actor_hash_after=new,actor_steps=8,completed_updates=1,checkpoint=dict(path=new,sha256=new))
            actor=new;return folder,out
        ns=dict(rt=rt,worker=worker,check_qualification=lambda *a:None,
            decision=lambda rows,*a:'ADVANCE' if rows['qualified'] else 'FRONTIER',
            LEVELS=c.LEVELS[:3],amount=c.amount,reuse_key=c.reuse_key,zero_qualified=c.zero_qualified)
        tree=ast.Module(body=[node(HERE/'boundary_course.py','execute')],type_ignores=[])
        exec(compile(tree,'boundary_course.execute','exec'),ns)
        r=dict(contract=dict(source_checkpoint=dict(path='source',sha256='sha'),source_actor_hash='source'),frozen_sha256='f')
        ns['execute'](Path('mock'),r)
        self.assertEqual(log[0]['role'],'initial_frontier');self.assertEqual(log[1]['role'],'frontier_train')
        self.assertTrue(log[1]['resume_optimizer'])
        self.assertEqual(log[2]['before'],.2375);self.assertNotEqual(log[2]['checkpoint']['path'],'source')
        self.assertEqual(log[3]['role'],'frontier_check');self.assertEqual(log[4]['role'],'frontier_train')
        self.assertEqual(log[-1]['before'],.225);self.assertNotEqual(log[-1]['checkpoint']['path'],'source')
        self.assertEqual(r['completed_updates'],2);self.assertEqual(r['actor_steps'],16)

if __name__=='__main__':unittest.main()

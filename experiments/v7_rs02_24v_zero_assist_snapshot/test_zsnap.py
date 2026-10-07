"""No-physics checks for a direct zero diagnostic and unchanged strict gates."""
import ast,copy,json,unittest
from pathlib import Path
import zsnap_contract as c
import zsnap_gates as gates
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'v7_rs02_24v_midpoint_curriculum'
def node(path,name):
 return next(n for n in ast.parse(path.read_text('utf-8')).body if isinstance(n,ast.FunctionDef) and n.name==name)
def source():
 for p in [OLD/'runs/course_01/level_00_u0001_frontier_recovery_check/result.json',OLD/'runs/parent_review_01/u1_1416_bundle/level_00_u0001_frontier_recovery_check/result.json']:
  if p.exists():return json.loads(p.read_text('utf-8-sig'))
 raise AssertionError('Preserved failed evidence required')
def request():
 ck=dict(path='source',sha256='s')
 return dict(mode='qualify',role='zero_probe',level_index=0,before=0.0,after=0.0,global_updates=0,
  reuse_preflight=False,reuse_source_evidence=False,checkpoint=ck,contract=dict(source_checkpoint=ck,no_training=True))
class Snapshot(unittest.TestCase):
 def test_exact_zero_request_and_no_training(self):
  self.assertTrue(c.validate_request(request()));self.assertEqual(c.BUDGET,0)
  for field,value in [('mode','train'),('role','target'),('before',.23125),('after',.225),('global_updates',1),('reuse_source_evidence',True),('reuse_preflight',True)]:
   r=request();r[field]=value
   with self.assertRaises(AssertionError):c.validate_request(r)
 def test_no_qualification_reuse(self):
  self.assertIsNone(c.reuse_key(request()))
  r=request();r['checkpoint']=dict(path='other',sha256='s')
  with self.assertRaises(AssertionError):c.validate_request(r)
 def test_entire_gate_module_unchanged(self):
  self.assertEqual(ast.dump(ast.parse((OLD/'midpoint_gates.py').read_text('utf-8'))),ast.dump(ast.parse((HERE/'zsnap_gates.py').read_text('utf-8'))))
 def test_source_failure_not_promoted(self):
  r=source();self.assertFalse(c.qualification(r['qualification']));self.assertFalse(c.learning_entry(r['qualification'],r['request']['contract']))
 def test_zero_requires_actual_zero_wrench(self):
  row=copy.deepcopy(source()['qualification']['native']);row.update(before=0.0,after=0.0)
  row['external_wrench'].update(max_abs_linear_force_n=0,max_abs_body_torque_nm=0,max_abs_root_generalized_force=0)
  self.assertTrue(gates.wrench_valid(row))
  for key in ['max_abs_linear_force_n','max_abs_body_torque_nm','max_abs_root_generalized_force']:
   bad=copy.deepcopy(row);bad['external_wrench'][key]=1e-12
   self.assertFalse(gates.wrench_valid(bad))
  row['external_wrench']['min_sampled_ticks']=0;self.assertFalse(gates.wrench_valid(row))
 def test_build_sampling_and_complete_evaluation_identical(self):
  for name in ('build','sample','qualify'):
   expected=ast.dump(node(OLD/'midpoint_worker.py',name)).replace('midpoint_','zsnap_')
   self.assertEqual(expected,ast.dump(node(HERE/'zsnap_worker.py',name)))
 def test_unchanged_environment_and_400hz_audits(self):
  for suffix in ('env','assist_audit','wrench_audit','trace_audit','row_audit'):
   expected=(OLD/('midpoint_'+suffix+'.py')).read_text('utf-8').replace('midpoint_','zsnap_')
   self.assertEqual(ast.dump(ast.parse(expected)),ast.dump(ast.parse((HERE/('zsnap_'+suffix+'.py')).read_text('utf-8'))))
 def test_qualification_audit_identical(self):
  expected=ast.dump(node(OLD/'midpoint_audit.py','check_qualification')).replace('midpoint_','zsnap_')
  self.assertEqual(expected,ast.dump(node(HERE/'zsnap_audit.py','check_qualification')))
 def test_no_train_function_or_ppo_dependency(self):
  tree=ast.parse((HERE/'zsnap_worker.py').read_text('utf-8'))
  self.assertFalse(any(isinstance(n,ast.FunctionDef) and n.name=='train' for n in tree.body))
  self.assertNotIn('peak_ppo',(HERE/'zsnap_worker.py').read_text('utf-8'))
 def test_audit_rejects_nonzero_actual_force_or_extra_stage(self):
  from types import SimpleNamespace
  run=Path('mock_run');folder=run/'probe';req=request();req['contract']['source_actor_hash']='a'
  row=dict(external_wrench=dict(min_sampled_ticks=2000,max_abs_linear_force_n=0,max_abs_body_torque_nm=0,max_abs_root_generalized_force=0))
  q=dict(request=req,actor_hash='a',qualified=False,new_physical_trials=1581,qualification=dict(native=row,batch=[copy.deepcopy(row) for _ in range(3)]))
  d=dict(contract=req['contract'],events=[dict(folder=str(folder),result_sha256='s')],completed_updates=0,actor_steps=0,zero_assistance_qualified=False,stop_reason='ZERO_PROBE_FAILED')
  rt=SimpleNamespace(checked=lambda p:d if p==run else q,contract=lambda:req['contract'],sha=lambda p:'s',read=lambda p:req,write=lambda *a:None,now=lambda:'t',verify=lambda:'f')
  ns=dict(rt=rt,Path=Path,validate_request=c.validate_request,check_qualification=lambda *a:None)
  exec(compile(ast.Module(body=[node(HERE/'zsnap_audit.py','audit')],type_ignores=[]),'audit','exec'),ns)
  self.assertFalse(ns['audit'](run)['zero_assistance_qualified'])
  q['qualification']['batch'][1]['external_wrench']['max_abs_body_torque_nm']=1e-12
  with self.assertRaises(AssertionError):ns['audit'](run)
  q['qualification']['batch'][1]['external_wrench']['max_abs_body_torque_nm']=0
  d['events'].append(d['events'][0])
  with self.assertRaises(AssertionError):ns['audit'](run)
if __name__=='__main__':unittest.main()

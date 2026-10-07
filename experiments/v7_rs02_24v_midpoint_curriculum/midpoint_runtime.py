"""Frozen single midpoint between the exhausted frontier and failed target."""
from pathlib import Path
from copy import deepcopy
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PARENT=HERE.parent/'v7_rs02_24v_timing_policy_learning'
sys.path.insert(0,str(PARENT))
import timing_runtime as parent
from timing_runtime import read,write,sha,now,exclusive,metrics,exploration
from midpoint_contract import (LEVELS,BUDGET,PRIOR_SHARED_UPDATES,OBJECTIVE,EXPLORATION_OFFSET,MAX_FRONTIER_RETURNS,MAX_LEVEL_UPDATES,MAX_FRONTIER_UPDATES,ACTOR_LR,MAX_ACCEPTED_KL)
PARENT_SHA='f705dab221a76a589b6e34fd6a6a0e512b8ca7f2e12d296fae0449b08221bebc'
COURSE_SHA='d311e005620213ea50a0d3bf4af8b840b612f0ab9b2bec1b50ceb2d29a32647a'
QUALIFIED_SHA='76934640bc8589a54c26c4eec2a90b1423a0226710ea92a2a22c20490b7bf4be'
FAILED_SHA='7b0dfb7ce0408c4a3d97961d89d4260ec62fe345f073209b67c805b1b8c9870f'
DIAG_SHA='3fd176e3d2626fe52c3adc9947c8ccff0f43e8792af47f9c603f698d2d570a1f'
SOURCE_SHA='4d4198713a2b79554f8c6cd30213e58ba078dfa0e7d31d9c5dea836e8f0713ff'
SOURCE_ACTOR='2a7849bd4a94b9097c22139cf889ccfc1c725ceb64eddb01e13c874e0242ca8b'
SOURCE_PARENT_SHA=PARENT_SHA

def verify():
 assert parent.verify()==PARENT_SHA
 f=read(HERE/'FROZEN.json');assert f['parent_frozen_sha256']==PARENT_SHA
 for name,digest in f['sha256'].items():assert sha(ROOT/name)==digest,name
 return sha(HERE/'FROZEN.json')

def source_evidence(c,key='source_failed'):
 assert key in ('source_qualified','source_failed')
 from timing_audit import check_qualification
 item=c[key];folder=Path(item['folder']);r=parent.checked(folder)
 assert sha(folder/'result.json')==item['result_sha256']==(QUALIFIED_SHA if key=='source_qualified' else FAILED_SHA)
 assert r['actor_hash']==item['actor_hash']==SOURCE_ACTOR
 assert r['qualified']==item['qualified']==(key=='source_qualified')
 assert r['request']['checkpoint']==c['source_checkpoint'] and r['request']['contract']['profile']==c['profile']
 assert r['request']['before']==r['request']['after']==(.2375 if key=='source_qualified' else .225)
 check_qualification(folder,r['request'],r)
 return r

def contract():
 run=PARENT/'runs/course_01';d=parent.checked(run)
 assert sha(run/'result.json')==COURSE_SHA and d['stop_reason']=='ENTRY_FAILED'
 assert d['completed_updates']==9 and d['actor_steps']==72 and d['deepest_qualified']=='uniform2375'
 for p in [run/'audit_exit_receipt.json',PARENT/'runs/course_01_launcher/wrapper_exit_receipt.json']:
  e=read(p);assert e['process_exited'] and e['exit_code']==0
 a=read(run/'audit_result.json');assert a['status']=='AUDITED' and a['result_sha256']==COURSE_SHA and a['updates_in_128_budget']==53
 repair=read(run/'parent_repair_request.json');assert repair['cause']=='ENTRY_FAILED' and not repair['goal_complete']
 diag=PARENT/'runs/parent_review_01/closure_tensor_native_1324.json';assert sha(diag)==DIAG_SHA
 proof=read(diag);assert proof['status']=='PARENT_CLOSURE_TENSOR_NATIVE_VERIFIED' and proof['new_physical_trials_in_audit']==0
 assert len(proof['checkpoints'])==9 and proof['completed_updates']==9 and proof['actor_steps']==72 and proof['shared_remaining']==75
 e=read(diag.parent/'parent_closure_native_1324.exit.json');assert e['process_exited'] and e['exit_code']==0
 c=deepcopy(d['contract']);source=d['latest_checkpoint'];assert sha(source['path'])==source['sha256']==SOURCE_SHA
 qfolder=run/'level_00_u0009_frontier_recovery_check';q=parent.checked(qfolder)
 assert sha(qfolder/'result.json')==QUALIFIED_SHA and q['request']['checkpoint']==source and q['actor_hash']==SOURCE_ACTOR
 assert q['qualified'] and q['request']['before']==q['request']['after']==.2375
 profile=deepcopy(q['request']['contract']['profile']);assert profile['horizon_s']==.0275
 fallback=dict(name='uniform2375',before=.2375,after=.2375,checkpoint=source,actor_hash=SOURCE_ACTOR,profile=profile,qualification_folder=str(qfolder),qualification_result_sha256=QUALIFIED_SHA)
 c.update(source_checkpoint=source,source_profile=profile,profile=profile,source_actor_hash=SOURCE_ACTOR,
  source_parent_frozen_sha256=PARENT_SHA,source_frozen_field='timing_frozen_sha256',
  source_qualified=dict(folder=str(qfolder),result_sha256=QUALIFIED_SHA,actor_hash=SOURCE_ACTOR,qualified=True,learning_entry=True),
  source_failed=dict(folder=str(run/'level_01_u0009_target'),result_sha256=FAILED_SHA,actor_hash=SOURCE_ACTOR,qualified=False,learning_entry=False),
  completed_parent=dict(course=str(run),result_sha256=COURSE_SHA,stop_reason=d['stop_reason'],actual_exit=0,audited=True),
  diagnostic_evidence=dict(path=str(diag),sha256=DIAG_SHA,physical_trials=0),
  preserved_qualified_fallbacks=c['preserved_qualified_fallbacks']+d['qualified_levels']+[fallback],
  total_update_budget=BUDGET,prior_shared_updates=PRIOR_SHARED_UPDATES,remaining_shared_updates=BUDGET,
  levels=[list(x) for x in LEVELS],max_frontier_returns=MAX_FRONTIER_RETURNS,max_level_updates=MAX_LEVEL_UPDATES,max_frontier_updates_per_target=MAX_FRONTIER_UPDATES,
  exploration_update_offset=EXPLORATION_OFFSET,training_objective=OBJECTIVE,actor_initial_and_max_lr=ACTOR_LR,max_accepted_kl=MAX_ACCEPTED_KL,
  updates_per_chunk=1,first_update_resume_source_optimizer_rng=False,frontier_must_requalify_strict_before_descent=True,
  revision='One fixed 23.125 percent midpoint; no further training at exhausted 23.75 percent frontier',
  controller_and_physics_unchanged=True,physical_bounds_and_final_thresholds_unchanged=True,final_qualification_unchanged=True,learning_entry_thresholds_unchanged=True,
  midpoint_source_interval=[.2375,.225],new_midpoint=(.2375+.225)/2,old_frontier_updates_exhausted=8,old_target_updates=1)
 c.pop('source_controller_id',None)
 assert c['new_midpoint']==LEVELS[0][1]==.23125 and all(x[1]<.2375 for x in LEVELS)
 source_evidence(c,'source_qualified');source_evidence(c,'source_failed')
 return c

def checked(folder):
 folder=Path(folder);r=read(folder/'result.json');e=read(folder/'exit_receipt.json')
 assert e['process_exited'] and e['exit_code']==0 and r['status']=='COMPLETED'
 assert r['frozen_sha256']==r['final_frozen_sha256']==verify()
 return r

def limit_for(started,seconds=43200):
 inherited=parent.limit_for(started,seconds)
 def check():
  assert not (HERE/'STOP').exists(),'Midpoint STOP requested'
  inherited()
 return check

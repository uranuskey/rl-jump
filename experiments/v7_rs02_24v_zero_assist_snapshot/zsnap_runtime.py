"""Frozen direct zero-assistance diagnostic after the midpoint course exited."""
from pathlib import Path
from copy import deepcopy
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PARENT=HERE.parent/'v7_rs02_24v_midpoint_curriculum'
sys.path.insert(0,str(PARENT))
import midpoint_runtime as parent
from midpoint_runtime import read,write,sha,now,exclusive,metrics,exploration
PARENT_SHA='f4287cba393804479f59efd31abdf8d4f55c88be10b568cccc6d84ddfd05fd22'
COURSE_SHA='a1211806275f265796bdc59dca67dd743af2ba607e519810a922afc678d610ce'
SOURCE_SHA='456fee200d889bf626e44b183b85a62658b8e6678979c78b9b20d2ee377b20cd'
SOURCE_ACTOR='cedf66c677a88cb4707360a6ac830363679f0f55c53639fc08747643546cc88b'
DIAG_SHA='b85513e696405ba72f79434b2827d0f7fb02998ce576456c1d2b34bc423a2054'
FAILED_SHA='d4e2d46a101ee4c5b4a6f5649463b4c4ccc34f4abe7ea647ef26d67175741edc'
def verify():
 assert parent.verify()==PARENT_SHA
 f=read(HERE/'FROZEN.json');assert f['parent_frozen_sha256']==PARENT_SHA
 for name,digest in f['sha256'].items():assert sha(ROOT/name)==digest,name
 return sha(HERE/'FROZEN.json')
def contract():
 run=PARENT/'runs/course_01';d=parent.checked(run)
 assert sha(run/'result.json')==COURSE_SHA and d['stop_reason']=='RECOVERY_ENTRY_FAILED'
 assert d['completed_updates']==1 and d['actor_steps']==8
 for p in [run/'audit_exit_receipt.json',PARENT/'runs/course_01_launcher/wrapper_exit_receipt.json']:
  e=read(p);assert e['process_exited'] and e['exit_code']==0
 a=read(run/'audit_result.json');assert a['status']=='AUDITED' and a['result_sha256']==COURSE_SHA and a['updates_in_128_budget']==54
 diag=PARENT/'runs/parent_review_01/closure_tensor_native_1416.json';assert sha(diag)==DIAG_SHA
 proof=read(diag);assert proof['completed_updates']==1 and proof['actor_steps']==8 and proof['shared_remaining']==74 and proof['new_physical_trials_in_audit']==0
 e=read(diag.parent/'parent_closure_native_1416.exit.json');assert e['process_exited'] and e['exit_code']==0
 source=d['latest_checkpoint'];assert sha(source['path'])==source['sha256']==SOURCE_SHA
 failed=run/'level_00_u0001_frontier_recovery_check';q=parent.checked(failed)
 assert sha(failed/'result.json')==FAILED_SHA and q['actor_hash']==SOURCE_ACTOR and q['request']['checkpoint']==source
 assert not q['qualified'] and q['request']['before']==q['request']['after']==.23125
 c=deepcopy(d['contract']);profile=deepcopy(c['profile']);assert profile['horizon_s']==.0275
 c.update(source_checkpoint=source,source_profile=profile,profile=profile,source_actor_hash=SOURCE_ACTOR,
  source_parent_frozen_sha256=PARENT_SHA,source_frozen_field='midpoint_frozen_sha256',
  completed_parent=dict(course=str(run),result_sha256=COURSE_SHA,stop_reason=d['stop_reason'],actual_exit=0,audited=True),
  diagnostic_evidence=dict(path=str(diag),sha256=DIAG_SHA,physical_trials=0),
  source_failed=dict(folder=str(failed),result_sha256=FAILED_SHA,actor_hash=SOURCE_ACTOR,qualified=False,learning_entry=False),
  preserved_qualified_fallbacks=c['preserved_qualified_fallbacks']+d['qualified_levels'],
  levels=[['uniform000',0.0,0.0]],prior_shared_updates=54,total_update_budget=0,remaining_shared_updates=74,
  no_training=True,direct_zero_probe_authorized=True,repeat_failed_actor_forbidden=True,
  controller_and_physics_unchanged=True,physical_bounds_and_final_thresholds_unchanged=True,final_qualification_unchanged=True,
  learning_entry_thresholds_unchanged=True,first_update_resume_source_optimizer_rng=False,
  revision='User-requested one full native45 plus exactly three512 zero-assistance diagnostic; no PPO')
 return c
def checked(folder):
 folder=Path(folder);r=read(folder/'result.json');e=read(folder/'exit_receipt.json')
 assert e['process_exited'] and e['exit_code']==0 and r['status']=='COMPLETED'
 assert r['frozen_sha256']==r['final_frozen_sha256']==verify()
 return r
def limit_for(started,seconds=43200):
 inherited=parent.limit_for(started,seconds)
 def check():
  assert not (HERE/'STOP').exists(),'Zero snapshot STOP requested'
  inherited()
 return check

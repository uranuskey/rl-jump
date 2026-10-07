"""Bind the selected timing controller and CPU-reuse both completed parent gates."""
from pathlib import Path
from copy import deepcopy
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PARENT=HERE.parent/'v7_rs02_24v_late_precontact_probe'
sys.path.insert(0,str(PARENT))
import late_runtime as parent
from late_runtime import read,write,sha,now,exclusive,metrics
sys.path.insert(0,str(HERE.parent/'v7_rs02_24v_frontier_admission_recovery'))
from recovery_runtime import exploration
from timing_contract import (LEVELS,BUDGET,PRIOR_SHARED_UPDATES,OBJECTIVE,EXPLORATION_OFFSET,
    MAX_FRONTIER_RETURNS,MAX_LEVEL_UPDATES,MAX_FRONTIER_UPDATES,ACTOR_LR,MAX_ACCEPTED_KL,learning_entry)
PARENT_SHA='91e4a496fa6c8a5ef219b713fe2ce9e0272594656714832f3f36d1cb167a2e65'
COURSE_SHA='747cd1c9624d8ef930f6d510965d255e66e82b2b1e167ac7420c51782f8749ff'
QUALIFIED_SHA='a14a235b626166c834ce0073e790a6e6f3fea88f7d959d7f7ebe78daf847dad4'
FAILED_SHA='c8b3414c70692ec870fa6b4db8793c3d54ebea9e941c7c84cb61269cfa33e1bc'
DIAG_SHA='96ef3f4bd1c6bafe9634209cfb9345c082c66baf4c98c11cefd04849f58066c1'
SOURCE_SHA=parent.SOURCE_SHA
SOURCE_ACTOR=parent.SOURCE_ACTOR
SOURCE_PARENT_SHA=parent.SOURCE_PARENT_SHA

def verify():
    assert parent.verify()==PARENT_SHA
    f=read(HERE/'FROZEN.json');assert f['parent_frozen_sha256']==PARENT_SHA
    for name,digest in f['sha256'].items():assert sha(ROOT/name)==digest,name
    return sha(HERE/'FROZEN.json')

def source_evidence(c,key='source_failed'):
    assert key in ('source_qualified','source_failed')
    from late_audit import check_qualification
    from late_contract import controller_id
    p=c[key];folder=Path(p['folder']);q=parent.checked(folder)
    expected=QUALIFIED_SHA if key=='source_qualified' else FAILED_SHA
    assert sha(folder/'result.json')==p['result_sha256']==expected
    assert q['actor_hash']==p['actor_hash']==SOURCE_ACTOR
    assert q['qualified']==p['qualified']==(key=='source_qualified')
    assert q['request']['checkpoint']==c['source_checkpoint'] and q['request']['contract']['profile']==c['profile']
    assert q['controller_id']==c['source_controller_id']==controller_id(SOURCE_ACTOR,c['profile'])
    assert not q['learning_entry'] and q['completed_updates']==q['actor_steps']==0
    check_qualification(folder,q['request'],q)
    return q

def contract():
    run=PARENT/'runs/course_01';d=parent.checked(run)
    assert sha(run/'result.json')==COURSE_SHA and d['stop_reason']=='TARGET_FAILED'
    assert d['completed_updates']==d['actor_steps']==0 and d['new_physical_trials']==3162
    assert d['deepest_qualified']=='uniform2375' and d['selected_candidate_index']==0
    assert len(d['qualified_levels'])==1 and len(d['events'])==2
    for path in [run/'audit_exit_receipt.json',PARENT/'runs/course_01_launcher/wrapper_exit_receipt.json']:
        e=read(path);assert e['process_exited'] and e['exit_code']==0
    a=read(run/'audit_result.json');assert a['status']=='AUDITED' and a['result_sha256']==COURSE_SHA
    assert a['updates_in_128_budget']==PRIOR_SHARED_UPDATES and a['shared_remaining']==BUDGET
    repair=read(run/'parent_repair_request.json');assert repair['cause']=='TARGET_FAILED' and not repair['goal_complete']
    diag=PARENT/'runs/parent_review_01/closure_tensor_native_1008.json';assert sha(diag)==DIAG_SHA
    proof=read(diag);assert proof['status']=='PARENT_CLOSURE_TENSOR_NATIVE_VERIFIED' and proof['new_physical_trials_in_audit']==0
    e=read(diag.parent/'parent_closure_native_1008.exit.json');assert e['process_exited'] and e['exit_code']==0
    c=deepcopy(d['contract']);source=c['source_checkpoint'];qualified=d['qualified_levels'][0]
    assert sha(source['path'])==source['sha256']==SOURCE_SHA
    assert qualified['checkpoint']==source and qualified['actor_hash']==SOURCE_ACTOR
    source_profile=deepcopy(c['source_profile']);profile=deepcopy(qualified['profile'])
    assert source_profile['horizon_s']==0.030 and profile['horizon_s']==0.0275
    assert qualified['qualification_result_sha256']==QUALIFIED_SHA
    fallbacks=deepcopy(c['preserved_qualified_fallbacks'])+deepcopy(d['qualified_levels'])
    c.update(source_checkpoint=source,source_profile=source_profile,profile=profile,source_actor_hash=SOURCE_ACTOR,
        source_parent_frozen_sha256=SOURCE_PARENT_SHA,source_frozen_field='recovery_frozen_sha256',
        source_controller_id=qualified['controller_id'],
        source_qualified=dict(folder=str(run/'profile_00_level_00_candidate'),result_sha256=QUALIFIED_SHA,actor_hash=SOURCE_ACTOR,qualified=True,learning_entry=False),
        source_failed=dict(folder=str(run/'profile_00_level_01_advance'),result_sha256=FAILED_SHA,actor_hash=SOURCE_ACTOR,qualified=False,learning_entry=False),
        completed_parent=dict(course=str(run),result_sha256=COURSE_SHA,stop_reason=d['stop_reason'],actual_exit=0,audited=True),
        diagnostic_evidence=dict(path=str(diag),sha256=DIAG_SHA,physical_trials=0),
        preserved_qualified_fallbacks=fallbacks,total_update_budget=BUDGET,prior_shared_updates=PRIOR_SHARED_UPDATES,
        remaining_shared_updates=BUDGET,levels=[list(x) for x in LEVELS],max_frontier_returns=MAX_FRONTIER_RETURNS,
        max_level_updates=MAX_LEVEL_UPDATES,max_frontier_updates_per_target=MAX_FRONTIER_UPDATES,
        exploration_update_offset=EXPLORATION_OFFSET,training_objective=OBJECTIVE,
        actor_initial_and_max_lr=ACTOR_LR,max_accepted_kl=MAX_ACCEPTED_KL,updates_per_chunk=1,
        first_update_resume_source_optimizer_rng=False,frontier_must_requalify_strict_before_descent=True,
        revision='Learn under the fixed selected 27.5 ms timing profile at a strictly qualified 23.75 percent frontier',
        controller_and_physics_unchanged=True,physical_bounds_and_final_thresholds_unchanged=True,
        final_qualification_unchanged=True,learning_entry_thresholds_unchanged=True)
    for key in ('horizons_s','selection_rule','new_ppo_updates','new_actor_steps','mandatory_qualified_source_update_before_target'):c.pop(key,None)
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
        assert not (HERE/'STOP').exists(),'Timing policy STOP requested'
        inherited()
    return check

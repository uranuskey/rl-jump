"""Bind the failed parent, qualified fallback and unchanged physical contract."""
from pathlib import Path
from copy import deepcopy
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PARENT=HERE.parent/'v7_rs02_24v_target_peak_learning'
sys.path.insert(0,str(PARENT))
import peak_runtime as parent
from peak_runtime import read,write,sha,now,exclusive,metrics,exploration
from boundary_contract import LEVELS,BUDGET,PRIOR_SHARED_UPDATES,OBJECTIVE,EXPLORATION_OFFSET,MAX_FRONTIER_RETURNS,MAX_LEVEL_UPDATES,ACTOR_LR,MAX_ACCEPTED_KL
PARENT_SHA='86354cb2ec5ed3e6aba3c98cf6080cca91fb79cf4d1fba8199b60b12349453c6'
COURSE_SHA='1b8deda8f5c368ae5fac89c8872cc327ebe06642bc5948d522203663f7e34c7d'
QUALIFIED_SHA='8853d3f3ba46f3aa3ce037d6eef31b52c6c6a4df2f64d91a8fc05bcf701f0e53'
SOURCE_SHA='33a12a11ba21b6fc20bc7c751ba379f1d5acfeef8ca01ba324d3c1191b690f2a'
SOURCE_ACTOR='eeefd89ce0fd7ffc87f6c2bbf76b589ce8b5fef653c565becab0d32f6ae1d663'
FAILED_TRACE_SHA='c97f5ade8ad28f24854170eba5eca2e5ff581d9d9e11815b8a2f074d808c965e'
FAILED_SUMMARY_SHA='70aa286e8df2926953ac1ca2146cea1f7cc7b36d1e0e40f275dfc79c624688b4'

def verify():
    assert parent.verify()==PARENT_SHA
    f=read(HERE/'FROZEN.json');assert f['parent_frozen_sha256']==PARENT_SHA
    for name,digest in f['sha256'].items():assert sha(ROOT/name)==digest,name
    return sha(HERE/'FROZEN.json')

def source_evidence(c,key='source_qualified'):
    assert key=='source_qualified'
    from peak_audit import check_qualification
    p=c[key];folder=Path(p['folder']);q=parent.checked(folder)
    assert sha(folder/'result.json')==p['result_sha256']==QUALIFIED_SHA
    assert q['actor_hash']==p['actor_hash']==SOURCE_ACTOR and q['qualified']
    assert q['request']['checkpoint']==c['source_checkpoint']
    check_qualification(folder,q['request'],q)
    return q

def contract():
    run=PARENT/'runs/course_01';d=read(run/'result.json')
    assert sha(run/'result.json')==COURSE_SHA and d['status']=='ERROR_STOPPED'
    assert d['completed_updates']==3 and d['actor_steps']==24 and d['deepest_qualified']=='uniform250'
    for path in [run/'exit_receipt.json',PARENT/'runs/course_01_launcher/wrapper_exit_receipt.json']:
        e=read(path);assert e['process_exited'] and e['exit_code']==1
    assert not (run/'audit_exit_receipt.json').exists()
    r=read(run/'parent_repair_request.json');assert not r['goal_complete']
    c=deepcopy(d['contract'])
    source=d['latest_checkpoint'];assert sha(source['path'])==source['sha256']==SOURCE_SHA
    folder=run/'level_00_u0003_target';failed=run/'level_01_u0003_target'
    assert sha(failed/'native.json')==FAILED_SUMMARY_SHA and sha(failed/'native_traces.npz')==FAILED_TRACE_SHA
    e=read(failed/'exit_receipt.json');assert e['process_exited'] and e['exit_code']==1
    native=read(failed/'native.json');request=read(failed/'request.json')
    assert request['before']==request['after']==0.225 and request['checkpoint']==source
    from boundary_contract import admission
    strict=admission(metrics(native),c['anchors_native'],native['max_pre_apex_mimic_v_rad_s'])
    assert not strict['passed'] and native['passed']==45
    assert all(not v['checks']['mean_impact'] and not v['checks']['worst_impact'] for v in strict['comparisons'].values())
    fallbacks=deepcopy(c['preserved_qualified_fallbacks'])+deepcopy(d['qualified_levels'])
    c.update(source_checkpoint=source,source_profile=deepcopy(c['profile']),source_actor_hash=SOURCE_ACTOR,
        source_parent_frozen_sha256=PARENT_SHA,source_frozen_field='peak_frozen_sha256',
        source_qualified=dict(folder=str(folder),result_sha256=QUALIFIED_SHA,actor_hash=SOURCE_ACTOR,qualified=True),
        parent_runtime_failure=dict(course=str(run),result_sha256=COURSE_SHA,course_exit_code=1,final_audit_performed=False),
        preserved_partial_failure=dict(folder=str(failed),native_summary_sha256=FAILED_SUMMARY_SHA,native_trace_sha256=FAILED_TRACE_SHA,
            strength=0.225,actor_hash=SOURCE_ACTOR,completed_native_worlds=45,completed_batch_worlds=0,qualified=False,native_strict=strict),
        preserved_qualified_fallbacks=fallbacks,total_update_budget=BUDGET,prior_shared_updates=PRIOR_SHARED_UPDATES,
        remaining_shared_updates=BUDGET,levels=[list(x) for x in LEVELS],max_frontier_returns=MAX_FRONTIER_RETURNS,max_level_updates=MAX_LEVEL_UPDATES,
        exploration_update_offset=EXPLORATION_OFFSET,training_objective=OBJECTIVE,
        actor_initial_and_max_lr=ACTOR_LR,max_accepted_kl=MAX_ACCEPTED_KL,updates_per_chunk=1,
        first_update_resume_source_optimizer_rng=True,mandatory_qualified_source_update_before_target=True,
        revision='CUDA scalar reciprocal audit repair and finer withdrawal with qualified-frontier learning',
        controller_and_physics_unchanged=True,physical_bounds_and_final_thresholds_unchanged=True,
        final_qualification_unchanged=True,old_partial_failed_native_not_retested=True)
    c.pop('source_failed',None)
    source_evidence(c)
    return c

def checked(folder):
    folder=Path(folder);r=read(folder/'result.json');e=read(folder/'exit_receipt.json')
    assert e['process_exited'] and e['exit_code']==0 and r['status']=='COMPLETED'
    assert r['frozen_sha256']==r['final_frozen_sha256']==verify()
    return r

def limit_for(started,seconds=43200):
    inherited=parent.limit_for(started,seconds)
    def check():
        assert not (HERE/'STOP').exists(),'Boundary STOP requested'
        inherited()
    return check


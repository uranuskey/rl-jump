"""Bind completed parent, failed source, qualified fallbacks and unchanged gates."""
from pathlib import Path
from copy import deepcopy
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PARENT=HERE.parent/'v7_rs02_24v_slot_boundary_recovery'
sys.path.insert(0,str(PARENT))
import boundary_runtime as parent
from boundary_runtime import read,write,sha,now,exclusive,metrics,exploration
from recovery_contract import (LEVELS,BUDGET,PRIOR_SHARED_UPDATES,OBJECTIVE,EXPLORATION_OFFSET,
    MAX_FRONTIER_RETURNS,MAX_LEVEL_UPDATES,MAX_FRONTIER_UPDATES,ACTOR_LR,MAX_ACCEPTED_KL,learning_entry)
PARENT_SHA='c5cebb9b77c70db2ebb52ecc220a2ba39927e1afcd704f775c297cae1acdda3c'
COURSE_SHA='00676b60261f4cf26817898c32758c15b6e7edffcff0b8e742c4983e66d286e0'
FAILED_SHA='6b34a834b8ddc363e19b7a0455d438d274b22e06b292f5e9b44c59697dc558b7'
SOURCE_SHA='c858ba31a1a434555bae5a13ea5186ddb181118b5a9884879d46c72538c6ea35'
SOURCE_ACTOR='13b5c133d2f87af669245fa74c96f7cea18084e944e0ed2bb2530b58f1ee4969'

def verify():
    assert parent.verify()==PARENT_SHA
    f=read(HERE/'FROZEN.json');assert f['parent_frozen_sha256']==PARENT_SHA
    for name,digest in f['sha256'].items():assert sha(ROOT/name)==digest,name
    return sha(HERE/'FROZEN.json')

def source_evidence(c,key='source_failed'):
    assert key=='source_failed'
    from boundary_audit import check_qualification
    p=c[key];folder=Path(p['folder']);q=parent.checked(folder)
    assert sha(folder/'result.json')==p['result_sha256']==FAILED_SHA
    assert q['actor_hash']==p['actor_hash']==SOURCE_ACTOR and not q['qualified']
    assert q['request']['checkpoint']==c['source_checkpoint']
    assert q['learning_entry'] and q['learning_evidence']['near_native_peak_entry']
    assert learning_entry(q['qualification'],c)
    check_qualification(folder,q['request'],q)
    return q

def contract():
    run=PARENT/'runs/course_01';d=parent.checked(run)
    assert sha(run/'result.json')==COURSE_SHA and d['stop_reason']=='FRONTIER_RECHECK_FAILED'
    assert d['completed_updates']==1 and d['actor_steps']==8 and d['deepest_qualified']=='uniform250'
    for path in [run/'audit_exit_receipt.json',PARENT/'runs/course_01_launcher/wrapper_exit_receipt.json']:
        e=read(path);assert e['process_exited'] and e['exit_code']==0
    a=read(run/'audit_result.json');assert a['status']=='AUDITED' and a['result_sha256']==COURSE_SHA
    assert a['updates_in_128_budget']==PRIOR_SHARED_UPDATES
    repair=read(run/'parent_repair_request.json')
    assert repair['cause']=='FRONTIER_RECHECK_FAILED' and not repair['goal_complete']
    c=deepcopy(d['contract']);source=d['latest_checkpoint']
    assert sha(source['path'])==source['sha256']==SOURCE_SHA
    folder=run/'level_00_u0001_frontier_check'
    fallbacks=deepcopy(c['preserved_qualified_fallbacks'])+deepcopy(d['qualified_levels'])
    c.update(source_checkpoint=source,source_profile=deepcopy(c['profile']),source_actor_hash=SOURCE_ACTOR,
        source_parent_frozen_sha256=PARENT_SHA,source_frozen_field='boundary_frozen_sha256',
        source_failed=dict(folder=str(folder),result_sha256=FAILED_SHA,actor_hash=SOURCE_ACTOR,qualified=False,learning_entry=True),
        completed_parent=dict(course=str(run),result_sha256=COURSE_SHA,stop_reason=d['stop_reason'],actual_exit=0,audited=True),
        preserved_qualified_fallbacks=fallbacks,total_update_budget=BUDGET,prior_shared_updates=PRIOR_SHARED_UPDATES,
        remaining_shared_updates=BUDGET,levels=[list(x) for x in LEVELS],max_frontier_returns=MAX_FRONTIER_RETURNS,
        max_level_updates=MAX_LEVEL_UPDATES,max_frontier_updates_per_target=MAX_FRONTIER_UPDATES,
        exploration_update_offset=EXPLORATION_OFFSET,training_objective=OBJECTIVE,
        actor_initial_and_max_lr=ACTOR_LR,max_accepted_kl=MAX_ACCEPTED_KL,updates_per_chunk=1,
        first_update_resume_source_optimizer_rng=True,frontier_must_requalify_strict_before_descent=True,
        revision='Explicit bounded frontier recovery using unchanged existing learning routes',
        controller_and_physics_unchanged=True,physical_bounds_and_final_thresholds_unchanged=True,
        final_qualification_unchanged=True,learning_entry_thresholds_unchanged=True)
    c.pop('source_qualified',None);c.pop('mandatory_qualified_source_update_before_target',None)
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
        assert not (HERE/'STOP').exists(),'Recovery STOP requested'
        inherited()
    return check

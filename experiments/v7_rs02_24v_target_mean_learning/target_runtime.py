"""Bind the completed parent's failure, source checkpoint and qualified fallbacks."""
from pathlib import Path
from copy import deepcopy
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PARENT=HERE.parent/'v7_rs02_24v_balanced_impact_learning'
sys.path.insert(0,str(PARENT))
import balanced_runtime as parent
from balanced_runtime import read,write,sha,now,exclusive,metrics,exploration
from target_contract import LEVELS,BUDGET,PRIOR_SHARED_UPDATES,OBJECTIVE,MAX_LEVEL_UPDATES,MEAN_ENTRY_RELATIVE_MARGIN,EXPLORATION_OFFSET,near_mean_entry
PARENT_SHA='e50068477de73fb28357b009f3cfef141d6ac57a80f1b5d78defbe0ce67fcea3'
COURSE_SHA='86e00afc74eb48eafc1c0cbf8f47e2d6751bf18d811d79479b6ae7ae400c1336'
FAILED_SHA='1cbf4c7dd787c5f9dd51d1b1beeeb8ac017b5003fad2271db1a61ed619b8e2b2'
SOURCE_SHA='3ab3769a24bd5cac03ab2d93600ec4d0ca46cfe63751e0aac850f00e5cb0abe3'
SOURCE_ACTOR='a68255163f08098acfd0bb8f673fc7f54acffdd54edd0eeb2a28ae54d3212727'

def verify():
    assert parent.verify()==PARENT_SHA
    f=read(HERE/'FROZEN.json');assert f['parent_frozen_sha256']==PARENT_SHA
    for name,digest in f['sha256'].items():assert sha(ROOT/name)==digest,name
    return sha(HERE/'FROZEN.json')

def source_evidence(c,key='source_failed'):
    assert key=='source_failed'
    from balanced_audit import check_qualification
    proof=c[key];folder=Path(proof['folder']);q=parent.checked(folder)
    assert sha(folder/'result.json')==proof['result_sha256']==FAILED_SHA
    assert q['actor_hash']==proof['actor_hash']==SOURCE_ACTOR
    assert q['request']['checkpoint']==c['source_checkpoint'] and not q['qualified'] and not q['learning_entry']
    check_qualification(folder,q['request'],q)
    return q

def contract():
    c=deepcopy(parent.contract());run=PARENT/'runs/course_01'
    d=parent.checked(run);a=read(run/'audit_result.json')
    assert sha(run/'result.json')==a['result_sha256']==COURSE_SHA and a['status']=='AUDITED'
    for path in [run/'audit_exit_receipt.json',PARENT/'runs/course_01_launcher/wrapper_exit_receipt.json']:
        e=read(path);assert e['process_exited'] and e['exit_code']==0
    assert d['stop_reason']=='ENTRY_FAILED' and d['completed_updates']==16 and d['actor_steps']==128
    assert d['deepest_qualified']=='uniform275' and a['updates_in_128_budget']==PRIOR_SHARED_UPDATES
    repair=read(run/'parent_repair_request.json');assert repair['cause']=='ENTRY_FAILED' and not repair['goal_complete']
    source=d['latest_checkpoint'];assert sha(source['path'])==source['sha256']==SOURCE_SHA
    folder=run/'level_01_u0016_target'
    c.update(source_checkpoint=source,source_profile=deepcopy(c['profile']),source_actor_hash=SOURCE_ACTOR,
        source_parent_frozen_sha256=PARENT_SHA,source_frozen_field='balanced_frozen_sha256',
        source_failed=dict(folder=str(folder),result_sha256=FAILED_SHA,actor_hash=SOURCE_ACTOR,qualified=False,old_learning_entry=False),
        parent_balanced_course=str(run),parent_balanced_course_sha256=COURSE_SHA,
        preserved_qualified_fallbacks=d['qualified_levels'],total_update_budget=BUDGET,
        prior_shared_updates=PRIOR_SHARED_UPDATES,remaining_shared_updates=BUDGET,levels=[list(x) for x in LEVELS],
        max_level_updates=MAX_LEVEL_UPDATES,mean_learning_relative_margin=MEAN_ENTRY_RELATIVE_MARGIN,
        exploration_update_offset=EXPLORATION_OFFSET,training_objective=OBJECTIVE,
        revision='direct target learning with bounded mean-only entry; unchanged final qualification',
        learning_entry_changed=True,physical_bounds_and_final_thresholds_unchanged=True,
        controller_and_physics_unchanged=True,original_13_reward_terms_unchanged=True,
        source_failed_evidence_reused_without_new_trials=True)
    for key in ('source_qualified','max_frontier_returns','physical_entry_and_final_gates_unchanged',
                'source_qualification_reused_without_new_trials','source35_requalification_required'):
        c.pop(key,None)
    q=source_evidence(c)
    assert near_mean_entry(q['qualification'],c['anchors_batch'])
    return c

def checked(folder):
    folder=Path(folder);r=read(folder/'result.json');e=read(folder/'exit_receipt.json')
    assert e['process_exited'] and e['exit_code']==0 and r['status']=='COMPLETED'
    assert r['frozen_sha256']==r['final_frozen_sha256']==verify()
    return r

def limit_for(started,seconds=43200):
    inherited=parent.limit_for(started,seconds)
    def check():
        assert not (HERE/'STOP').exists(),'Target-mean STOP requested'
        inherited()
    return check

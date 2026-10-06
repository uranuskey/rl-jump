"""Bind completed target-mean failure and all preserved qualified fallbacks."""
from pathlib import Path
from copy import deepcopy
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PARENT=HERE.parent/'v7_rs02_24v_target_mean_learning'
sys.path.insert(0,str(PARENT))
import target_runtime as parent
from target_runtime import read,write,sha,now,exclusive,metrics,exploration
from peak_contract import LEVELS,BUDGET,PRIOR_SHARED_UPDATES,OBJECTIVE,MAX_LEVEL_UPDATES,EXPLORATION_OFFSET,NATIVE_PEAK_ENTRY_RELATIVE_MARGIN,ACTOR_LR,MAX_ACCEPTED_KL,near_native_peak_entry
PARENT_SHA='fd18374197646f7478bafd5d028ffdb74cdb66df184d46dccf06c5a891389818'
COURSE_SHA='9f8a7096bed12591a1f0fcf1ab317dc93f8e0113a5f6f327fdd5e790aa3422f2'
FAILED_SHA='7a0f0e7846aa689ff078da4228fb841cb57d9f18fa8724ad345220b79a09b331'
SOURCE_SHA='7f11f238241f2bb4c88bfcb20e944dd9f606ebf97c058c3ca640bd535dfcbf6d'
SOURCE_ACTOR='d77b3eff1ccca2b6abba0c87de4ed76e9634c799661e985c6900e976be7db16f'
def verify():
    assert parent.verify()==PARENT_SHA
    f=read(HERE/'FROZEN.json');assert f['parent_frozen_sha256']==PARENT_SHA
    for name,digest in f['sha256'].items():assert sha(ROOT/name)==digest,name
    return sha(HERE/'FROZEN.json')
def source_evidence(c,key='source_failed'):
    assert key=='source_failed'
    from target_audit import check_qualification
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
    assert d['stop_reason']=='ENTRY_FAILED' and d['completed_updates']==6 and d['actor_steps']==48
    assert d['deepest_qualified']=='uniform2625' and a['updates_in_128_budget']==PRIOR_SHARED_UPDATES
    repair=read(run/'parent_repair_request.json');assert repair['cause']=='ENTRY_FAILED' and not repair['goal_complete']
    source=d['latest_checkpoint'];assert sha(source['path'])==source['sha256']==SOURCE_SHA
    folder=run/'level_01_u0006_target'
    fallbacks=deepcopy(c['preserved_qualified_fallbacks'])+deepcopy(d['qualified_levels'])
    c.update(source_checkpoint=source,source_profile=deepcopy(c['profile']),source_actor_hash=SOURCE_ACTOR,
        source_parent_frozen_sha256=PARENT_SHA,source_frozen_field='target_frozen_sha256',
        source_failed=dict(folder=str(folder),result_sha256=FAILED_SHA,actor_hash=SOURCE_ACTOR,qualified=False,old_learning_entry=False),
        parent_target_course=str(run),parent_target_course_sha256=COURSE_SHA,
        preserved_qualified_fallbacks=fallbacks,total_update_budget=BUDGET,prior_shared_updates=PRIOR_SHARED_UPDATES,
        remaining_shared_updates=BUDGET,levels=[list(x) for x in LEVELS],max_level_updates=MAX_LEVEL_UPDATES,
        native_peak_learning_relative_margin=NATIVE_PEAK_ENTRY_RELATIVE_MARGIN,exploration_update_offset=EXPLORATION_OFFSET,
        training_objective=OBJECTIVE,actor_initial_and_max_lr=ACTOR_LR,max_accepted_kl=MAX_ACCEPTED_KL,updates_per_chunk=1,
        revision='bounded native-peak learning with smaller single PPO updates; all final gates unchanged',
        learning_entry_changed=True,physical_bounds_and_final_thresholds_unchanged=True,
        controller_and_physics_unchanged=True,original_13_reward_terms_unchanged=True,
        source_failed_evidence_reused_without_new_trials=True)
    q=source_evidence(c);assert near_native_peak_entry(q['qualification'],c['anchors_native'])
    return c
def checked(folder):
    folder=Path(folder);r=read(folder/'result.json');e=read(folder/'exit_receipt.json')
    assert e['process_exited'] and e['exit_code']==0 and r['status']=='COMPLETED'
    assert r['frozen_sha256']==r['final_frozen_sha256']==verify()
    return r
def limit_for(started,seconds=43200):
    inherited=parent.limit_for(started,seconds)
    def check():
        assert not (HERE/'STOP').exists(),'Target-peak STOP requested'
        inherited()
    return check

"""Bind the completed recovery course and its strict source plus failed target."""
from pathlib import Path
from copy import deepcopy
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PARENT=HERE.parent/'v7_rs02_24v_frontier_admission_recovery'
sys.path.insert(0,str(PARENT))
import recovery_runtime as parent
from recovery_runtime import read,write,sha,now,exclusive,metrics
from horizon_contract import LEVELS,HORIZONS,profile,controller_id,PRIOR_SHARED_UPDATES,REMAINING_SHARED_UPDATES
PARENT_SHA='42496bdefb050f6b7cef01b13554e9fc57a24eb4782dc4f12a86d247af4737c8'
COURSE_SHA='4a2cafd55efe2dcd29adb2d97a0caa93c2b6bf6e277191bbb14df565b7f2b0b0'
SOURCE_SHA='1881c05ae75d599e1c5e79bb5a78e3cd131d13a322a29a7848d58b28aa7b34a4'
SOURCE_ACTOR='d18846b30298105ac69e9aa2a4224941915ae07d9a0fb5f1c39d613a1c48718c'
QUALIFIED_SHA='f6f31ca57be5ee45f40d0df8daab2b4d7a35c8741da597a69d32a8fdaf96250b'
FAILED_SHA='f9c1d82110cd91e446b97482148a2f5ce778c753a1b824657a0b4c2ba554e1da'

def verify():
    assert parent.verify()==PARENT_SHA
    f=read(HERE/'FROZEN.json');assert f['parent_frozen_sha256']==PARENT_SHA
    for name,digest in f['sha256'].items():assert sha(ROOT/name)==digest,name
    return sha(HERE/'FROZEN.json')

def contract():
    run=PARENT/'runs/course_01';d=parent.checked(run)
    assert sha(run/'result.json')==COURSE_SHA and d['stop_reason']=='RECOVERY_ENTRY_FAILED'
    assert d['completed_updates']==4 and d['actor_steps']==32
    for path in [run/'audit_exit_receipt.json',PARENT/'runs/course_01_launcher/wrapper_exit_receipt.json']:
        e=read(path);assert e['process_exited'] and e['exit_code']==0
    a=read(run/'audit_result.json');assert a['status']=='AUDITED' and a['result_sha256']==COURSE_SHA and a['updates_in_128_budget']==44
    repair=read(run/'parent_repair_request.json');assert repair['cause']=='RECOVERY_ENTRY_FAILED' and not repair['goal_complete']
    source=dict(path=str(run/'level_00_u0002_frontier_recovery_train/update_0003.pt'),sha256=SOURCE_SHA,global_update=3)
    assert sha(source['path'])==SOURCE_SHA
    # Carry the exact physical references; predecessor-only training metadata stays
    # in the immutable parent result rather than masquerading as current settings.
    c={key:deepcopy(d['contract'][key]) for key in ('profile','anchors_native','anchors_batch')}
    c.update(source_checkpoint=source,source_profile=deepcopy(c['profile']),source_actor_hash=SOURCE_ACTOR,
        source_parent_frozen_sha256=PARENT_SHA,source_frozen_field='recovery_frozen_sha256',
        completed_parent=dict(course=str(run),result_sha256=COURSE_SHA,stop_reason=d['stop_reason'],actual_exit=0,audited=True),
        source_qualified=dict(folder=str(run/'level_00_u0003_frontier_recovery_check'),result_sha256=QUALIFIED_SHA,actor_hash=SOURCE_ACTOR,qualified=True,strength=0.25),
        source_failed=dict(folder=str(run/'level_01_u0003_target'),result_sha256=FAILED_SHA,actor_hash=SOURCE_ACTOR,qualified=False,strength=0.2375),
        preserved_qualified_fallbacks=deepcopy(d['contract']['preserved_qualified_fallbacks'])+deepcopy(d['qualified_levels']),
        levels=[list(x) for x in LEVELS],horizons_s=list(HORIZONS),prior_shared_updates=PRIOR_SHARED_UPDATES,
        remaining_shared_updates=REMAINING_SHARED_UPDATES,total_update_budget=0,new_ppo_updates=0,new_actor_steps=0,
        selection_rule='first strict candidate in frozen horizon order; then strict descent; never retest a controller-level pair',
        controller_change='horizon_s only (plus descriptive profile name); source actor fixed',
        inherited_source_qualification_for_candidates=False,controller_unchanged=False,
        physical_environment_unchanged=True,physical_bounds_and_final_thresholds_unchanged=True,
        training_disabled=True,no_new_ppo=True,autonomous_repair_authorized=True,
        revision='Anticipatory precontact timing with unchanged motor FIFO and 3 mm command budget')
    return c

def case_contract(index):
    c=contract();c['profile']=profile(c['source_profile'],index);return c

def source_evidence(c):
    from recovery_audit import check_qualification
    evidence=[]
    for key in ('source_qualified','source_failed'):
        ref=c[key];folder=Path(ref['folder']);q=parent.checked(folder)
        assert sha(folder/'result.json')==ref['result_sha256'] and q['actor_hash']==SOURCE_ACTOR
        assert q['qualified']==ref['qualified'] and q['request']['before']==q['request']['after']==ref['strength']
        assert q['request']['checkpoint']==c['source_checkpoint']
        assert q['request']['contract']['profile']==c['source_profile']
        check_qualification(folder,q['request'],q);evidence.append(q)
    return evidence

def checked(folder):
    folder=Path(folder);r=read(folder/'result.json');e=read(folder/'exit_receipt.json')
    assert e['process_exited'] and e['exit_code']==0 and r['status']=='COMPLETED'
    assert r['frozen_sha256']==r['final_frozen_sha256']==verify()
    return r

def limit_for(started,seconds=43200):
    inherited=parent.limit_for(started,seconds)
    def check():
        assert not (HERE/'STOP').exists(),'Horizon probe STOP requested'
        inherited()
    return check

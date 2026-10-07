"""Bind the fully closed long-horizon failure and the unchanged recovery actor."""
from pathlib import Path
from copy import deepcopy
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PARENT=HERE.parent/'v7_rs02_24v_precontact_horizon_probe'
sys.path.insert(0,str(PARENT))
import horizon_runtime as parent
from horizon_runtime import read,write,sha,now,exclusive,metrics
from late_contract import LEVELS,HORIZONS,profile,controller_id,PRIOR_SHARED_UPDATES,REMAINING_SHARED_UPDATES
PARENT_SHA='49f4d02fdb5080a4f0a2e28cce6b2744820a56c395f5b9e63dc523016c5897db'
SOURCE_PARENT_SHA=parent.PARENT_SHA
SOURCE_SHA=parent.SOURCE_SHA
SOURCE_ACTOR=parent.SOURCE_ACTOR
COURSE_SHA='1268a047c0859935c78ad3e5de64a01171ceebb768850a4e0020a1a7f80b9b03'
DIAG_SHA='9b82df9c3f844f79ea857ac8adb75782c8a71dc6c8486b77a8135c9a02d6ee71'
FAILED_SHAS=['5161c9b53bd8c0929565202a3b63596e08e088806a86adb2bca3cda90a739247',
 '214f21ed86999448d8763caaa6c4ed31721575161fc59c191cd0b8bf38b7f5ca',
 'cdfac737b84b29a0b3ce0e59d6ea978eb8ff4dace2e3a45c77e8ef2122bd9605']

def verify():
    assert parent.verify()==PARENT_SHA
    f=read(HERE/'FROZEN.json');assert f['parent_frozen_sha256']==PARENT_SHA
    for name,digest in f['sha256'].items():assert sha(ROOT/name)==digest,name
    return sha(HERE/'FROZEN.json')

def contract():
    run=PARENT/'runs/course_01';d=parent.checked(run)
    assert sha(run/'result.json')==COURSE_SHA and d['stop_reason']=='NO_PROFILE_QUALIFIED'
    assert d['completed_updates']==d['actor_steps']==0 and d['new_physical_trials']==4743
    assert d['selected_candidate_index'] is None and not d['qualified_levels'] and not d['zero_assistance_qualified']
    for path in [run/'audit_exit_receipt.json',PARENT/'runs/course_01_launcher/wrapper_exit_receipt.json']:
        e=read(path);assert e['process_exited'] and e['exit_code']==0
    a=read(run/'audit_result.json');assert a['status']=='AUDITED' and a['result_sha256']==COURSE_SHA
    assert a['updates_in_128_budget']==44 and a['shared_remaining']==84
    repair=read(run/'parent_repair_request.json');assert repair['cause']=='NO_PROFILE_QUALIFIED' and not repair['goal_complete']
    diag=PARENT/'runs/parent_review_01/closure_tensor_native_0924.json'
    assert sha(diag)==DIAG_SHA
    proof=read(diag);assert proof['status']=='PARENT_CLOSURE_TENSOR_NATIVE_VERIFIED' and proof['new_physical_trials_in_audit']==0
    e=read(diag.parent/'parent_closure_native_0924.exit.json');assert e['process_exited'] and e['exit_code']==0
    assert len(d['events'])==3
    failed=[]
    for i,event in enumerate(d['events']):
        folder=Path(event['folder']);q=parent.checked(folder)
        assert sha(folder/'result.json')==event['result_sha256']==FAILED_SHAS[i]
        assert q['actor_hash']==SOURCE_ACTOR and not q['qualified']
        assert q['request']['candidate_index']==i and q['request']['before']==q['request']['after']==0.2375
        failed.append(dict(folder=str(folder),result_sha256=FAILED_SHAS[i],profile=q['request']['contract']['profile'],
            controller_id=q['controller_id'],qualified=False,strength=0.2375))
    c=deepcopy(d['contract'])
    c['completed_source_recovery']=deepcopy(c['completed_parent'])
    c.update(completed_parent=dict(course=str(run),result_sha256=COURSE_SHA,stop_reason=d['stop_reason'],actual_exit=0,audited=True),
        failed_previous_profiles=failed,diagnostic_evidence=dict(path=str(diag),sha256=DIAG_SHA,physical_trials=0),
        levels=[list(x) for x in LEVELS],horizons_s=list(HORIZONS),prior_shared_updates=PRIOR_SHARED_UPDATES,
        remaining_shared_updates=REMAINING_SHARED_UPDATES,total_update_budget=0,new_ppo_updates=0,new_actor_steps=0,
        selection_rule='first strict candidate in frozen 27.5/25/20 ms order; then strict descent; no repeated controller-level pair',
        revision='Later causal precontact timing after the fully failed 35/40/45 ms experiment; unchanged 3 mm command budget')
    assert c['source_checkpoint']['sha256']==SOURCE_SHA and c['source_actor_hash']==SOURCE_ACTOR
    assert c['profile']==c['source_profile'] and c['source_profile']['horizon_s']==0.030
    assert not set(HORIZONS).intersection([0.030]+[p['profile']['horizon_s'] for p in failed])
    return c

def case_contract(index):
    c=contract();c['profile']=profile(c['source_profile'],index);return c

def source_evidence(c):
    evidence=parent.source_evidence(c)
    from horizon_audit import check_qualification
    for ref in c['failed_previous_profiles']:
        folder=Path(ref['folder']);q=parent.checked(folder)
        assert sha(folder/'result.json')==ref['result_sha256'] and not q['qualified']
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
        assert not (HERE/'STOP').exists(),'Late precontact probe STOP requested'
        inherited()
    return check

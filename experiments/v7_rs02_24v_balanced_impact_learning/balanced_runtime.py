"""Freeze the failed finite search and reuse the last qualified source, never failures as passes."""
from pathlib import Path
from copy import deepcopy
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PARENT=HERE.parent/'v7_rs02_24v_policy_step_filter'
sys.path.insert(0,str(PARENT))
import filter_runtime as parent
from filter_runtime import read,write,sha,now,exclusive,metrics
from tail_runtime import exploration
from balanced_contract import LEVELS,BUDGET,PRIOR_SHARED_UPDATES,OBJECTIVE,MAX_FRONTIER_RETURNS
PARENT_SHA='40bbe441f46ecb22c66cdbaa7d8fa653e988d28a5d89dbd1b23a8a0045463878'
COURSE_SHA='14a7a90b16a653f3da2bd63fd84158310382b64ba4cfb95d74937d03e009e73f'

def verify():
    assert parent.verify()==PARENT_SHA
    f=read(HERE/'FROZEN.json');assert f['parent_frozen_sha256']==PARENT_SHA
    for name,digest in f['sha256'].items():assert sha(ROOT/name)==digest,name
    return sha(HERE/'FROZEN.json')

def contract():
    c=deepcopy(parent.contract());run=PARENT/'runs/course_01'
    d=parent.checked(run);a=read(run/'audit_result.json')
    assert sha(run/'result.json')==a['result_sha256']==COURSE_SHA and a['status']=='AUDITED'
    for path in [run/'audit_exit_receipt.json',PARENT/'runs/course_01_launcher/wrapper_exit_receipt.json']:
        e=read(path);assert e['process_exited'] and e['exit_code']==0
    assert d['stop_reason']=='NO_FEASIBLE_STEP' and d['selected_candidate'] is None
    assert d['completed_updates']==d['actor_steps']==0 and not d['qualified_levels']
    from filter_audit import check_qualification
    from filter_contract import replay
    assert replay(d['events'])['stop_reason']=='NO_FEASIBLE_STEP'
    for candidate,event in zip(d['candidates'],d['events']):
        folder=Path(event['folder']);q=parent.checked(folder)
        assert sha(folder/'result.json')==event['result_sha256']
        assert q['request']['candidate']==candidate and not q['qualified']
        check_qualification(folder,q['request'],q)
    qualified=c['source_evidence']['level_03_u0002_target']
    failed=c['source_evidence']['level_04_u0002_target']
    c.update(source_checkpoint=c['source_a'],source_profile=deepcopy(c['profile']),source_actor_hash=c['source_actor_a'],
        source_parent_frozen_sha256=parent.PARENT_SHA,source_frozen_field='tail_frozen_sha256',
        source_qualified=qualified,source_failed=failed,parent_filter_course=str(run),parent_filter_course_sha256=COURSE_SHA,
        total_update_budget=BUDGET,prior_shared_updates=PRIOR_SHARED_UPDATES,remaining_shared_updates=BUDGET,
        levels=[list(x) for x in LEVELS],training_objective=OBJECTIVE,max_frontier_returns=MAX_FRONTIER_RETURNS,
        revision='fine assistance level plus bulk/tail objective and two-update learning checkpoints',
        no_new_ppo=False,controller_and_physics_unchanged=True,original_13_reward_terms_unchanged=True,
        physical_bounds_and_final_thresholds_unchanged=True,declared_new_assistance_level=.2625,
        source_qualification_reused_without_new_trials=True)
    c.pop('fractions',None)
    return c

def checked(folder):
    folder=Path(folder);r=read(folder/'result.json');e=read(folder/'exit_receipt.json')
    assert e['process_exited'] and e['exit_code']==0 and r['status']=='COMPLETED'
    assert r['frozen_sha256']==r['final_frozen_sha256']==verify()
    return r

def source_evidence(c,key):
    from tail_runtime import checked as parent_checked
    from tail_audit import check_qualification
    proof=c[key];folder=Path(proof['folder']);r=parent_checked(folder)
    assert sha(folder/'result.json')==proof['result_sha256']
    assert r['actor_hash']==proof['actor_hash']==c['source_actor_hash']
    assert r['request']['checkpoint']==c['source_checkpoint'] and r['qualified']==proof['qualified']
    check_qualification(folder,r['request'],r)
    return r

def limit_for(started,seconds=43200):
    inherited=parent.limit_for(started,seconds)
    def check():
        assert not (HERE/'STOP').exists(),'Balanced-impact STOP requested'
        inherited()
    return check

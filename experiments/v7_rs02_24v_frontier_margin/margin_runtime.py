"""Preserve all gates and parent evidence; add a qualified-frontier PPO warmup."""
from pathlib import Path
from copy import deepcopy
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PARENT=HERE.parent/'v7_rs02_24v_rare_contact_learning'
sys.path.insert(0,str(PARENT))
import rare_runtime as parent
from rare_runtime import read,write,sha,now,exclusive,exploration,metrics
from margin_contract import LEVELS,BUDGET,PRIOR_UPDATES,WARMUP_STRENGTH,WARMUP_UPDATES,impact_only_failure
PARENT_SHA='05917a540a6784ae4dd213f1d88e67d8f8a6464d00ab30974ed34323d75641b6'
PARENT_RESULT='eed56081f93547a6a912a892fe95e511115352239124a81a1c28ea1ef17cbe77'
QUALIFIED_SHA='25a7ba3fa61c2b5c61e704dced4526e663ed83a9fbd295b5d38fb46d46bd331b'
FAILED_SHA='c101c04abfd300c46b75d75130f63ad4acb5235688f880332eba4d2e2219a193'
SOURCE_SHA='23ab7c0ab4fe648bb41904ee146355a2ccc4bf828266d1a435ac2b42a47b272d'
SOURCE_ACTOR='0f1136bc03dc773fb74e29a4f3baf0324a2edf1a74a2b226d0240ad226bc0b0f'

def verify():
    assert parent.verify()==PARENT_SHA
    f=read(HERE/'FROZEN.json');assert f['parent_frozen_sha256']==PARENT_SHA
    for name,digest in f['sha256'].items():assert sha(ROOT/name)==digest,name
    return sha(HERE/'FROZEN.json')

def contract():
    c=deepcopy(parent.contract());folder=PARENT/'runs/course_01'
    d=parent.checked(folder);a=read(folder/'audit_result.json')
    assert a['status']=='AUDITED' and a['result_sha256']==sha(folder/'result.json')==PARENT_RESULT
    receipt=read(folder/'audit_exit_receipt.json');assert receipt['process_exited'] and receipt['exit_code']==0
    receipt=read(PARENT/'runs/course_01_launcher/wrapper_exit_receipt.json')
    assert receipt['process_exited'] and receipt['exit_code']==0
    assert d['deepest_qualified']=='uniform350' and d['stop_reason']=='ENTRY_FAILED'
    assert d['completed_updates']==2 and d['actor_steps']==16
    last=d['qualified_levels'][-1];qfolder=Path(last['qualification_folder']);q=parent.checked(qfolder)
    from rare_audit import check_qualification
    check_qualification(qfolder,read(qfolder/'request.json'),q)
    assert q['qualified'] and q['learning_entry'] and q['actor_hash']==SOURCE_ACTOR
    assert sha(qfolder/'result.json')==last['qualification_result_sha256']==QUALIFIED_SHA
    assert last['before']==last['after']==WARMUP_STRENGTH
    source=last['checkpoint'];assert sha(source['path'])==source['sha256']==SOURCE_SHA
    assert source==d['latest_checkpoint']
    failed=Path(d['events'][-1]['folder']);r=parent.checked(failed)
    check_qualification(failed,read(failed/'request.json'),r)
    assert sha(failed/'result.json')==FAILED_SHA and not r['qualified'] and not r['learning_entry']
    assert r['actor_hash']==SOURCE_ACTOR and r['request']['checkpoint']==source
    assert r['request']['before']==r['request']['after']==LEVELS[0][1]
    assert impact_only_failure(r['qualification'])
    c.pop('reused_initial_qualification',None)
    c.update(source_checkpoint=source,source_profile=deepcopy(c['profile']),source_actor_hash=SOURCE_ACTOR,
        source_parent_frozen_sha256=PARENT_SHA,source_frozen_field='rare_frozen_sha256',
        retained_previous_qualification=str(qfolder),retained_qualification_sha256=QUALIFIED_SHA,
        failed_lower_level=dict(folder=str(failed),result_sha256=FAILED_SHA,qualified=False,learning_entry=False),
        total_update_budget=BUDGET,prior_updates=PRIOR_UPDATES,previous_course_updates=2,
        warmup_strength=WARMUP_STRENGTH,warmup_updates=WARMUP_UPDATES,levels=[list(x) for x in LEVELS],
        revision='two PPO updates at qualified35 percent before fresh32.5 percent qualification',
        no_admission_qualification_controller_reward_physics_changes=True,old_failure_preserved=True)
    return c

def limit_for(started,seconds=43200):
    inherited=parent.limit_for(started,seconds)
    def check():
        assert not (HERE/'STOP').exists(),'Frontier-margin STOP requested'
        inherited()
    return check

def checked(folder):
    folder=Path(folder);d=read(folder/'result.json');r=read(folder/'exit_receipt.json')
    assert r['process_exited'] and r['exit_code']==0
    assert d['status']=='COMPLETED' and d['frozen_sha256']==d['final_frozen_sha256']==verify()
    return d

"""Bind two checkpoints from the same observed PPO update lineage."""
from pathlib import Path
from copy import deepcopy
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PARENT=HERE.parent/'v7_rs02_24v_tail_impact_learning'
sys.path.insert(0,str(PARENT))
import tail_runtime as parent
from tail_runtime import read,write,sha,now,exclusive,metrics
from filter_contract import LEVELS,FRACTIONS,PRIOR_SHARED_UPDATES
PARENT_SHA='9a6876de709f0406a485ceead4011dbe4b03c38bd21101b1ea98341e11ee841f'
COURSE_SHA='ef22d1a741aee91c4af069bf720ff037d967b2756f57199920657e86c576dd19'
SOURCE_A_SHA='cea33aef31b86f450615e5fd91b771c0a31ec838c783e9fd3fbb41e391836969'
SOURCE_B_SHA='bf7c49a46ea8f51847938b8f407c9eaceabe306573286632f323d1eb62530e4f'
ACTOR_A='aac42f334cd45f65ef85a2017b9851fc5e8e34c133018fa9f9708ea94d2ab605'
ACTOR_B='b81034c6d32a741a6b232c2d24e6ccaf9a8c38444cc3840b209ffd3c34414dfd'

def verify():
    assert parent.verify()==PARENT_SHA
    f=read(HERE/'FROZEN.json');assert f['parent_frozen_sha256']==PARENT_SHA
    for name,digest in f['sha256'].items():assert sha(ROOT/name)==digest,name
    return sha(HERE/'FROZEN.json')

def contract():
    c=deepcopy(parent.contract());folder=PARENT/'runs/course_01'
    result=parent.checked(folder);audit=read(folder/'audit_result.json')
    assert sha(folder/'result.json')==audit['result_sha256']==COURSE_SHA and audit['status']=='AUDITED'
    for path in [folder/'audit_exit_receipt.json',PARENT/'runs/course_01_launcher/wrapper_exit_receipt.json']:
        receipt=read(path);assert receipt['process_exited'] and receipt['exit_code']==0
    assert result['stop_reason']=='ENTRY_FAILED' and result['deepest_qualified']=='uniform275'
    assert result['completed_updates']==10 and result['actor_steps']==80 and audit['updates_in_128_budget']==14
    a=parent.checked(folder/'level_00_u0000_frontier_train')
    b=parent.checked(folder/'level_03_u0002_frontier_train')
    assert a['actor_hash_after']==b['actor_hash_before']==ACTOR_A and b['actor_hash_after']==ACTOR_B
    assert a['checkpoint']['sha256']==sha(a['checkpoint']['path'])==SOURCE_A_SHA
    assert b['checkpoint']['sha256']==sha(b['checkpoint']['path'])==SOURCE_B_SHA
    assert b['request']['checkpoint']==a['checkpoint'] and b['completed_updates']==8 and b['actor_steps']==64
    evidence={}
    from tail_audit import check_qualification
    for name,expected,actor,qualified in [
        ('level_03_u0002_target','323171275e8999e1cb4211c6ad500e523a41728a8af806513390aa97f8020cf9',ACTOR_A,True),
        ('level_04_u0002_target','357e92d9eea4615d8ce4f43fd6559848efcea8f70fdba1ea97655e0a3e597931',ACTOR_A,False),
        ('level_04_u0010_target','029aeee020917c986e44f8e71d6486178b7a2ae02b3384538be8ed2006558423',ACTOR_B,False)]:
        qfolder=folder/name;q=parent.checked(qfolder)
        assert sha(qfolder/'result.json')==expected and q['actor_hash']==actor and q['qualified']==qualified
        check_qualification(qfolder,q['request'],q)
        evidence[name]=dict(folder=str(qfolder),result_sha256=expected,qualified=qualified,actor_hash=actor)
    c.update(source_a=a['checkpoint'],source_b=b['checkpoint'],source_actor_a=ACTOR_A,source_actor_b=ACTOR_B,
             parent_course=str(folder),parent_course_sha256=COURSE_SHA,parent_frozen_sha256=PARENT_SHA,
             source_evidence=evidence,prior_shared_updates=PRIOR_SHARED_UPDATES,remaining_shared_updates=114,
             total_update_budget=0,levels=[list(x) for x in LEVELS],fractions=list(FRACTIONS),
             revision='finite actor-parameter backtracking along a single PPO lineage',
             no_new_ppo=True,physical_entry_and_final_gates_unchanged=True,
             controller_and_physics_unchanged=True,original_13_reward_terms_unchanged=True)
    c['source35_requalification_required']=False
    return c

def checked(folder):
    folder=Path(folder);r=read(folder/'result.json');e=read(folder/'exit_receipt.json')
    assert e['process_exited'] and e['exit_code']==0 and r['status']=='COMPLETED'
    assert r['frozen_sha256']==r['final_frozen_sha256']==verify()
    return r

def limit_for(started,seconds=43200):
    check_parent=parent.limit_for(started,seconds)
    def check():
        assert not (HERE/'STOP').exists(),'Policy-step filter STOP requested'
        check_parent()
    return check

"""Bind the failed frontier course and preserve its immutable checkpoints."""
from pathlib import Path
from copy import deepcopy
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PARENT=HERE.parent/'v7_rs02_24v_frontier_margin'
sys.path.insert(0,str(PARENT))
import margin_runtime as parent
from margin_runtime import read,write,sha,now,exclusive,exploration,metrics
from tail_contract import LEVELS,BUDGET,PRIOR_SHARED_UPDATES,OBJECTIVE,MAX_FRONTIER_RETURNS,impact_only_failure
PARENT_SHA='40d39ffa2ab763870b91a765057fe868b2db997461d6e374a04ab3c1956fead8'
FAILED_SHA='1ec8561f4b18c7aca2dc7f4329b82bd22c9f869a5254a8596f224a5ed1c19903'
TRAIN_SHA='e221949cad9689ee9ef159830e88940cd869a4b8e7c179172ab4c87baee54f5a'
SOURCE_SHA='4bdf01aa79dfa9adb2c99b2ca45a789f66e66a6b45b48e131c75d539f0c393d0'
SOURCE_ACTOR='dc348ad16d7f1b1bda62534a55077b5a5ca2563344cc0a8dbf61aadade2d1050'

def verify():
    assert parent.verify()==PARENT_SHA
    f=read(HERE/'FROZEN.json');assert f['parent_frozen_sha256']==PARENT_SHA
    for name,digest in f['sha256'].items():assert sha(ROOT/name)==digest,name
    return sha(HERE/'FROZEN.json')

def contract():
    c=deepcopy(parent.contract());folder=PARENT/'runs/course_01'
    d=parent.checked(folder);audit=read(folder/'audit_result.json')
    assert audit['status']=='AUDITED' and audit['result_sha256']==sha(folder/'result.json')
    for p in [folder/'audit_exit_receipt.json',PARENT/'runs/course_01_launcher/wrapper_exit_receipt.json']:
        receipt=read(p);assert receipt['process_exited'] and receipt['exit_code']==0
    assert d['stop_reason']=='ENTRY_FAILED' and d['deepest_qualified'] is None
    assert d['completed_updates']==2 and d['actor_steps']==16
    train_folder=folder/'frontier350_u0000_train';t=parent.checked(train_folder)
    assert sha(train_folder/'result.json')==TRAIN_SHA
    assert t['actor_hash_after']==SOURCE_ACTOR and t['actor_hash_before']==c['source_actor_hash']
    assert t['completed_updates']==2 and t['actor_steps']==16
    qfolder=folder/'level_00_u0002_qualify';q=parent.checked(qfolder)
    from margin_audit import check_qualification
    check_qualification(qfolder,read(qfolder/'request.json'),q)
    assert sha(qfolder/'result.json')==FAILED_SHA
    assert not q['qualified'] and not q['learning_entry'] and impact_only_failure(q['qualification'])
    assert q['actor_hash']==SOURCE_ACTOR
    source=t['checkpoint'];assert sha(source['path'])==source['sha256']==SOURCE_SHA
    assert source==q['request']['checkpoint']==d['latest_checkpoint']
    c.pop('no_admission_qualification_controller_reward_physics_changes',None)
    c.pop('no_controller_reward_physics_changes',None)
    c.update(source_checkpoint=source,source_profile=deepcopy(c['profile']),source_actor_hash=SOURCE_ACTOR,
             source_parent_frozen_sha256=PARENT_SHA,source_frozen_field='margin_frozen_sha256',
             source35_requalification_required=True,latest_failed_qualification=str(qfolder),
             latest_failed_qualification_sha256=FAILED_SHA,
             total_update_budget=BUDGET,prior_shared_updates=PRIOR_SHARED_UPDATES,
             levels=[list(x) for x in LEVELS],training_objective=OBJECTIVE,
             max_frontier_returns=MAX_FRONTIER_RETURNS,
             revision='bounded high-impact penalty and qualified-frontier return curriculum',
             physical_entry_and_final_gates_unchanged=True,controller_and_physics_unchanged=True,
             original_13_reward_terms_unchanged=True,old_failure_preserved=True)
    return c

def limit_for(started,seconds=43200):
    inherited=parent.limit_for(started,seconds)
    def check():
        assert not (HERE/'STOP').exists(),'Tail-impact STOP requested'
        inherited()
    return check

def checked(folder):
    folder=Path(folder);d=read(folder/'result.json');r=read(folder/'exit_receipt.json')
    assert r['process_exited'] and r['exit_code']==0
    assert d['status']=='COMPLETED' and d['frozen_sha256']==d['final_frozen_sha256']==verify()
    return d

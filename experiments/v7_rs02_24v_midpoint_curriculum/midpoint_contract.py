"""Reuse existing bounded learning entries; recover strict frontier before descent."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'v7_rs02_24v_slot_boundary_recovery'))
import boundary_contract as parent
from midpoint_gates import (checked_levels,wrench_valid,qualification,original_learning_entry,
    impact_only_failure,near_mean_entry,near_native_peak_entry,learning_evidence,learning_entry,
    admission,learning_admission,metrics,zero_qualified,objective,OBJECTIVE,CHECKS,
    MEAN_ENTRY_RELATIVE_MARGIN,NATIVE_PEAK_ENTRY_RELATIVE_MARGIN,ACTOR_LR,MAX_ACCEPTED_KL)
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'v7_rs02_24v_late_precontact_probe'))
from late_contract import LEVELS as OLD_LEVELS
LEVELS=[('uniform23125',0.23125,0.23125)]+OLD_LEVELS[1:]
PRIOR_SHARED_UPDATES=53
BUDGET=128-PRIOR_SHARED_UPDATES
EXPLORATION_OFFSET=39
MAX_FRONTIER_RETURNS=8
MAX_LEVEL_UPDATES=8
MAX_FRONTIER_UPDATES=8

def amount(total,first=False):
    assert 0<=total<BUDGET
    return 1

def decision(rows,c,total,returns,local=0):
    if qualification(rows):return 'ADVANCE'
    if total>=BUDGET:return 'BUDGET_EXHAUSTED'
    if learning_entry(rows,c):return 'LEARN' if local<MAX_LEVEL_UPDATES else 'TARGET_REPAIR_LIMIT'
    if impact_only_failure(rows) and returns<MAX_FRONTIER_RETURNS:return 'FRONTIER'
    return 'ENTRY_FAILED'

def frontier_action(rows,c,total,used,must_update):
    if qualification(rows) and not must_update:return 'FRONTIER_READY'
    if total>=BUDGET:return 'BUDGET_EXHAUSTED'
    if used>=MAX_FRONTIER_UPDATES:return 'FRONTIER_RECOVERY_LIMIT'
    if qualification(rows):return 'TRAIN_QUALIFIED_FRONTIER'
    if learning_entry(rows,c):return 'TRAIN_BOUNDED_RECOVERY'
    return 'RECOVERY_ENTRY_FAILED'

def reuse_key(r):
    if r['mode']!='qualify' or r['global_updates']!=0 or r['checkpoint']!=r['contract']['source_checkpoint']:
        return None
    if r['role']=='target' and r['level_index']==1:return 'source_failed'
    return None

def validate_request(r):
    assert r['mode'] in ('qualify','train') and r['reuse_preflight'] is False
    i=r['level_index'];assert 0<=i<len(LEVELS)
    assert (r['before'],r['after'])==tuple(LEVELS[i][1:])
    assert 0<=r['global_updates']<=BUDGET
    if r['mode']=='train':
        assert r['updates']==1 and r['global_updates']+r['updates']<=BUDGET
        assert r['role'] in ('frontier_train','frontier_recovery_train','target_train')
        assert r['entry_qualification'] and r['entry_result_sha256']
        assert isinstance(r['resume_optimizer'],bool) and not r['reuse_source_evidence']
    else:
        assert r['role'] in ('initial_recovery','frontier_check','frontier_recovery_check','target')
        assert r['reuse_source_evidence']==(reuse_key(r) is not None)
    return True

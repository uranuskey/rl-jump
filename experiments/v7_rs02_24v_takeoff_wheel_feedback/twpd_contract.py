"""Native screen can reject; only unchanged full 45+3x512 gates qualify."""
from twpd_gates import qualification,admission,learning_admission,learning_entry,learning_evidence,objective,metrics,zero_qualified,OBJECTIVE,ACTOR_LR,MAX_ACCEPTED_KL,wrench_valid
from twpd_control import PROFILE
LEVELS=[('uniform000',0.,0.)]
PRIOR_SHARED_UPDATES=54
BUDGET=0
EXPLORATION_OFFSET=40
def reuse_key(r):return None
def native_ready(row):
    return (row['metrics']['worlds']==45 and row['metrics']['passed']==45
        and row['strict_admission']['passed'] and row['audits']['status']=='PASS' and wrench_valid(row))
def validate_request(r):
    assert r['mode']=='qualify' and r['role']=='zero_probe'
    assert r['level_index']==0 and r['before']==r['after']==0. and r['global_updates']==0
    assert r['reuse_preflight'] is False and r['reuse_source_evidence'] is False
    assert r['checkpoint']==r['contract']['source_checkpoint'] and r['contract']['no_training'] is True
    assert r['contract']['takeoff_feedback']==PROFILE and r['contract']['native_first_stop_if_not_strict'] is True
    return True

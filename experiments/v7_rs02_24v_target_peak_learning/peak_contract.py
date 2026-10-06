"""Bounded native-peak learning admission; all final physical gates unchanged."""
from pathlib import Path
import math,sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'v7_rs02_24v_target_mean_learning'))
import target_contract as old
from target_contract import admission,learning_admission,metrics,qualification,wrench_valid,zero_qualified,objective,OBJECTIVE,CHECKS,near_mean_entry,MEAN_ENTRY_RELATIVE_MARGIN
LEVELS=old.LEVELS[1:]
PRIOR_SHARED_UPDATES=36
BUDGET=128-PRIOR_SHARED_UPDATES
MAX_LEVEL_UPDATES=8
EXPLORATION_OFFSET=22
NATIVE_PEAK_ENTRY_RELATIVE_MARGIN=0.0005
ACTOR_LR=2.5e-6
MAX_ACCEPTED_KL=0.005

def near_native_peak_entry(rows,anchors):
    """Learning only: all batches strict; native only worst-impact may miss."""
    try:
        native,batches=rows['native'],rows['batch'];m=native['metrics'];a=native['strict_admission']
        if (m['worlds']!=45 or m['passed']!=45 or native['audits']['status']!='PASS'
            or not wrench_valid(native) or not a['pre_apex_constraint_margin']
            or len(batches)!=3 or [x['repeat'] for x in batches]!=[1,2,3]
            or set(anchors)!={'corrected_reference','original_reference'}
            or set(a['comparisons'])!=set(anchors)):return False
        for row in batches:
            if (row['metrics']['worlds']!=512 or row['metrics']['passed']!=512
                or not row['strict_admission']['passed'] or not wrench_valid(row)):return False
            sa=row['strict_admission']
            if not sa['pre_apex_constraint_margin'] or set(sa['comparisons'])!=set(anchors):return False
            for comp in sa['comparisons'].values():
                if set(comp['checks'])!=CHECKS or not all(comp['checks'].values()):return False
        for name,comp in a['comparisons'].items():
            checks=comp['checks']
            if set(checks)!=CHECKS or any(not v for k,v in checks.items() if k!='worst_impact'):return False
            cap=1.03*anchors[name]['max_force_n'];value=m['max_force_n']
            if not (math.isfinite(value) and math.isfinite(cap) and cap>0
                    and 0<=value<=cap*(1+NATIVE_PEAK_ENTRY_RELATIVE_MARGIN)):return False
        return True
    except (KeyError,TypeError,ValueError):return False

def learning_evidence(rows,c):
    prior=old.learning_evidence(rows,c)
    peak_route=near_native_peak_entry(rows,c['anchors_native'])
    return dict(passed=prior['passed'] or peak_route,parent_learning_evidence=prior,
        original_learning_entry=prior['original_learning_entry'],near_mean_target_entry=prior['near_mean_target_entry'],
        near_native_peak_entry=peak_route,native_peak_relative_allowance=NATIVE_PEAK_ENTRY_RELATIVE_MARGIN,
        route=prior['route'] if prior['passed'] else 'near_native_peak' if peak_route else 'rejected',
        final_qualified=qualification(rows),final_gates_unchanged=True)

def learning_entry(rows,c):return learning_evidence(rows,c)['passed']
def amount(total):
    assert 0<=total<BUDGET
    return 1
def decision(rows,c,total,local):
    if qualification(rows):return 'ADVANCE'
    if total>=BUDGET:return 'BUDGET_EXHAUSTED'
    if local>=MAX_LEVEL_UPDATES:return 'TARGET_REPAIR_LIMIT'
    return 'LEARN' if learning_entry(rows,c) else 'ENTRY_FAILED'
def reuse_key(r):
    return ('source_failed' if r['mode']=='qualify' and r['global_updates']==r['level_index']==0
            and r['checkpoint']==r['contract']['source_checkpoint'] else None)
def validate_request(r):
    assert r['mode'] in ('qualify','train') and r['reuse_preflight'] is False
    i=r['level_index'];assert 0<=i<len(LEVELS)
    assert (r['before'],r['after'])==tuple(LEVELS[i][1:]) and 0<=r['global_updates']<=BUDGET
    if r['mode']=='train':
        assert r['role']=='target_train' and r['updates']==1 and r['global_updates']+1<=BUDGET
        assert r['entry_qualification'] and r['entry_result_sha256'] and isinstance(r['resume_optimizer'],bool)
        assert not r['reuse_source_evidence']
    else:assert r['role']=='target' and r['reuse_source_evidence']==(reuse_key(r) is not None)
    return True

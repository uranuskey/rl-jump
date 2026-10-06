"""Learn at a near-feasible target; preserve every final physical qualification."""
from pathlib import Path
import math,sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'v7_rs02_24v_balanced_impact_learning'))
import balanced_contract as old
from balanced_contract import admission,learning_admission,metrics,qualification,wrench_valid,zero_qualified,objective,OBJECTIVE
LEVELS=old.LEVELS[1:]
PRIOR_SHARED_UPDATES=30
BUDGET=128-PRIOR_SHARED_UPDATES
MAX_LEVEL_UPDATES=16
EXPLORATION_OFFSET=16
MEAN_ENTRY_RELATIVE_MARGIN=.0025
CHECKS={'all_cases','wheel_height','com_height','actual_stroke','mean_impact','worst_impact','recovery','stable_interval','no_rebound'}

def near_mean_entry(rows,anchors):
    """Experimental learning allowance only; no contact/worst-force allowance added."""
    try:
        native,batches=rows['native'],rows['batch']
        if (native['metrics']['worlds']!=45 or native['audits']['status']!='PASS'
            or not native['strict_admission']['passed'] or not wrench_valid(native)
            or len(batches)!=3 or [x['repeat'] for x in batches]!=[1,2,3]
            or set(anchors)!={'corrected_reference','original_reference'}):return False
        for row in batches:
            a=row['strict_admission'];m=row['metrics']
            if (m['worlds']!=512 or m['passed']!=512 or not wrench_valid(row)
                or not a['pre_apex_constraint_margin'] or set(a['comparisons'])!=set(anchors)):return False
            for name,comp in a['comparisons'].items():
                checks=comp['checks']
                if set(checks)!=CHECKS or any(not v for k,v in checks.items() if k!='mean_impact'):return False
                strict_cap=1.02*anchors[name]['mean_force_n']
                value=m['mean_force_n']
                if not (math.isfinite(value) and math.isfinite(strict_cap) and strict_cap>0
                        and 0<=value<=strict_cap*(1+MEAN_ENTRY_RELATIVE_MARGIN)):return False
        return True
    except (KeyError,TypeError,ValueError):return False

def learning_evidence(rows,c):
    original=old.learning_entry(rows)
    mean_route=near_mean_entry(rows,c['anchors_batch'])
    return dict(passed=original or mean_route,original_learning_entry=original,
        near_mean_target_entry=mean_route,mean_relative_allowance=MEAN_ENTRY_RELATIVE_MARGIN,
        route='original_learning_entry' if original else 'near_mean_target' if mean_route else 'rejected',
        final_qualified=qualification(rows),final_gates_unchanged=True)

def learning_entry(rows,c):return learning_evidence(rows,c)['passed']

def amount(total):
    assert 0<=total<BUDGET
    return min(2,BUDGET-total)

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
    assert (r['before'],r['after'])==tuple(LEVELS[i][1:])
    assert 0<=r['global_updates']<=BUDGET
    if r['mode']=='train':
        assert r['role']=='target_train' and 0<r['updates']<=2 and r['global_updates']+r['updates']<=BUDGET
        assert r['entry_qualification'] and r['entry_result_sha256'] and isinstance(r['resume_optimizer'],bool)
        assert not r['reuse_source_evidence']
    else:
        assert r['role']=='target' and r['reuse_source_evidence']==(reuse_key(r) is not None)
    return True

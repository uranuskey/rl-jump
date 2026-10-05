"""Simulation learning may include rare terminated contacts; promotion never does."""
from pathlib import Path
from collections import Counter
import math,sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'v7_rs02_24v_zero_assist_curriculum'))
from zero_contract import admission,qualification,wrench_valid,zero_qualified
LEVELS=[(f'uniform{i:03d}',i/1000,i/1000) for i in range(425,-1,-25)]
BUDGET=128
PRIOR_UPDATES=2
MAX_FAILED_PER_512=2
MAX_TRAINING_PEAK_N=350.

def metrics(summary):
    rows=summary['cases'];avg=lambda k:sum(r['landing_metrics'][k] for r in rows)/len(rows)
    rebounds=[r['landing_metrics']['peak_rebound_vz_mps'] for r in rows]
    return dict(passed=sum(r['passed'] for r in rows),worlds=len(rows),mean_force_n=avg('peak_force_n'),
        max_force_n=max(r['landing_metrics']['peak_force_n'] for r in rows),
        mean_com_stroke_m=avg('first_stop_com_drop_m'),min_com_stroke_m=min(r['landing_metrics']['first_stop_com_drop_m'] for r in rows),
        mean_wheel_cm=sum(r['wheel_cm'] for r in rows)/len(rows),mean_com_cm=sum(r['com_cm'] for r in rows)/len(rows),
        mean_recovery_s=sum(r['success_s']-r['touchdown_s'] for r in rows)/len(rows),
        min_stable_s=min(r['landing_metrics']['best_continuous_stable_s'] for r in rows),
        max_rebound_mps=max(rebounds),reasons=dict(Counter(r['reason'] for r in rows)),
        rebounding_worlds=sum(v>1e-6 for v in rebounds),mean_rebound_mps=sum(rebounds)/len(rows))

def eligible_contact(row):
    """Only a completed takeoff followed by a shallow, bounded self-contact failure."""
    try:
        t=row['terminal_diagnostics'];h=t['leg_height_m'];torque=t['motor_torque_nm'];envelope=t['motor_envelope_nm']
        values=h+torque+envelope+[row['touchdown_s'],row['end_s'],row['landing_metrics']['peak_force_n'],
            t['mimic_q_error_rad'],t['mimic_v_error_rad_s'],t['nonwheel_force_n']]
        return (all(math.isfinite(v) for v in values) and not row['passed'] and row['reason']=='illegal_contact'
            and row['gate_tick']>=0 and t['reward_apex'] and t['phase_before']==3 and t['self_contact']
            and 0<row['touchdown_s']<row['end_s']<=row['touchdown_s']+.15
            and t['nonwheel_force_n']==0 and not t['saturated']
            and len(h)==2 and all(.09<=x<=.23 for x in h)
            and len(torque)==len(envelope)==4 and all(abs(x)<=e+1e-4 and 0<=e<=17.0001 for x,e in zip(torque,envelope))
            and not any(t['leg_overspeed']) and not any(t['wheel_overspeed'])
            and 0<=t['mimic_q_error_rad']<=.001 and 0<=t['mimic_v_error_rad_s']<=.01
            and 0<=row['landing_metrics']['peak_force_n']<=MAX_TRAINING_PEAK_N)
    except (KeyError,TypeError,ValueError):return False

def learning_admission(summary,anchors,pre_apex_v):
    m=metrics(summary);old=admission(m,anchors,pre_apex_v,True)
    if old['passed']:
        return dict(passed=True,route='original_learning_entry',original=old,failed_worlds=0,final_gates_unchanged=True)
    bad=[r for r in summary['cases'] if not r['passed']];good=[r for r in summary['cases'] if r['passed']]
    survivors=admission(metrics({'cases':good}),anchors,pre_apex_v,True) if good else {'passed':False}
    preserved={name:{k:v for k,v in c.get('checks',{}).items() if k not in ('all_cases','worst_impact','stable_interval')}
        for name,c in old['comparisons'].items()}
    checks=dict(fixed_batch=m['worlds']==512,rare_count=1<=len(bad)<=MAX_FAILED_PER_512,
        distinct_worlds=len({r['world'] for r in summary['cases']})==m['worlds'],
        contact_scope=bool(bad) and all(eligible_contact(r) for r in bad),
        surviving_cases=survivors['passed'],pre_apex_constraint_margin=old['pre_apex_constraint_margin'],
        global_height_stroke_mean_impact_recovery_rebound=all(c and all(c.values()) for c in preserved.values()),
        bounded_all_world_peak=math.isfinite(m['max_force_n']) and m['max_force_n']<=MAX_TRAINING_PEAK_N)
    return dict(passed=all(checks.values()),route='rare_posttouch_self_contact',checks=checks,
        original=old,survivors=survivors,failed_worlds=len(bad),failed_world_ids=[r['world'] for r in bad],
        training_failure_limit=MAX_FAILED_PER_512,training_peak_limit_n=MAX_TRAINING_PEAK_N,
        final_gates_unchanged=True,qualification=False)

def learning_entry(rows):
    all_rows=[rows['native']]+rows['batch']
    return (rows['native']['metrics']['worlds']==45 and rows['native']['audits']['status']=='PASS'
        and len(rows['batch'])==3 and [r['repeat'] for r in rows['batch']]==[1,2,3]
        and all(r['metrics']['worlds']==512 for r in rows['batch'])
        and all(r['learning_admission']['passed'] and wrench_valid(r) for r in all_rows))

def decision(rows,completed):
    if qualification(rows):return 'ADVANCE'
    if not learning_entry(rows):return 'ENTRY_FAILED'
    if completed>=BUDGET:return 'BUDGET_EXHAUSTED'
    return 'LEARN'

def chunk_size(completed,level_updates):
    assert 0<=completed<=BUDGET and level_updates>=0
    return min(2 if level_updates==0 else 8,BUDGET-completed)

def validate_reuse(request):
    """Reuse one failed qualification as training evidence, never as a retry/pass."""
    c=request['contract']
    assert request['mode']=='qualify' and request.get('reuse_preflight') is True
    assert request['level_index']==request['global_updates']==0
    assert (request['before'],request['after'])==tuple(LEVELS[0][1:])
    assert request['checkpoint']==c['source_checkpoint']
    proof=c['reused_initial_qualification']
    assert proof['old_qualified'] is False and proof['old_learning_entry'] is False
    return proof

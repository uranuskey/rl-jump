"""Finer withdrawal and a strictly-qualified frontier; final limits unchanged."""
from pathlib import Path
import sys,math
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'v7_rs02_24v_target_peak_learning'))
import peak_contract as parent
from peak_contract import admission,learning_admission,metrics,zero_qualified,objective,OBJECTIVE,CHECKS,MEAN_ENTRY_RELATIVE_MARGIN,NATIVE_PEAK_ENTRY_RELATIVE_MARGIN,ACTOR_LR,MAX_ACCEPTED_KL
from zero_contract import inherited_qualification
LEVELS=[('uniform250',0.25,0.25),('uniform2375',0.2375,0.2375),('uniform23125',0.23125,0.23125)]+parent.LEVELS[1:]
PRIOR_SHARED_UPDATES=39
BUDGET=128-PRIOR_SHARED_UPDATES
EXPLORATION_OFFSET=25
MAX_FRONTIER_RETURNS=8
MAX_LEVEL_UPDATES=8

def checked_levels(before,after):
    assert (before,after) in [(a,b) for _,a,b in LEVELS], 'Undeclared assistance level'
    return before,after

def wrench_valid(row):
    e=row.get('external_wrench',{})
    before,after=row.get('before'),row.get('after')
    try: checked_levels(before,after)
    except (AssertionError,TypeError): return False
    return (e.get('min_sampled_ticks',0)>0
        and e.get('max_abs_linear_force_n')==0
        and e.get('max_abs_root_generalized_force')==0
        and 0<=e.get('max_abs_body_torque_nm',float('inf'))<=20*before+5e-5
        and (before!=0 or e.get('max_abs_body_torque_nm')==0))

def qualification(rows):
    return inherited_qualification(rows) and all(wrench_valid(r) for r in [rows['native']]+rows['batch'])

def original_learning_entry(rows):
    all_rows=[rows['native']]+rows['batch']
    return (rows['native']['metrics']['worlds']==45 and rows['native']['audits']['status']=='PASS'
        and len(rows['batch'])==3 and [r['repeat'] for r in rows['batch']]==[1,2,3]
        and all(r['metrics']['worlds']==512 for r in rows['batch'])
        and all(r['learning_admission']['passed'] and wrench_valid(r) for r in all_rows))

def impact_only_failure(rows):
    if qualification(rows) or rows['native']['audits']['status']!='PASS':return False
    entries=[rows['native']]+rows['batch']
    if rows['native']['metrics']['worlds']!=45 or len(rows['batch'])!=3:return False
    if [r['repeat'] for r in rows['batch']]!=[1,2,3]:return False
    if any(r['metrics']['worlds']!=512 for r in rows['batch']):return False
    for r in entries:
        a=r['strict_admission']
        if not wrench_valid(r) or not a['pre_apex_constraint_margin']:return False
        for c in a['comparisons'].values():
            if not c.get('checks') or any(not v for k,v in c['checks'].items() if k not in ('mean_impact','worst_impact')):return False
    return True

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
    original=original_learning_entry(rows);mean=near_mean_entry(rows,c['anchors_batch']);peak=near_native_peak_entry(rows,c['anchors_native'])
    return dict(passed=original or mean or peak,original_learning_entry=original,near_mean_target_entry=mean,near_native_peak_entry=peak,
        mean_relative_allowance=MEAN_ENTRY_RELATIVE_MARGIN,native_peak_relative_allowance=NATIVE_PEAK_ENTRY_RELATIVE_MARGIN,
        route='original_learning_entry' if original else 'near_mean_target' if mean else 'near_native_peak' if peak else 'rejected',
        final_qualified=qualification(rows),final_gates_unchanged=True)

def learning_entry(rows,c):return learning_evidence(rows,c)['passed']

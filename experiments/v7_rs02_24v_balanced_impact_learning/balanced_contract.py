"""Finer assistance schedule and bulk/tail costs; final physical bounds unchanged."""
from pathlib import Path
import sys,math
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'v7_rs02_24v_tail_impact_learning'))
from tail_contract import admission,learning_admission,metrics,zero_qualified,penalty as tail_penalty
from zero_contract import inherited_qualification
LEVELS=[('uniform275',.275,.275),('uniform2625',.2625,.2625)]+[(f'uniform{i:03d}',i/1000,i/1000) for i in range(250,-1,-25)]
PRIOR_SHARED_UPDATES=14
BUDGET=128-PRIOR_SHARED_UPDATES
MAX_FRONTIER_RETURNS=8
OBJECTIVE=dict(version='bulk_plus_tail_peak_v1',bulk_threshold_n=280.,bulk_weight_per_n=2.,bulk_cap=100.,
    tail_threshold_n=330.,tail_weight=20.,tail_scale_n=10.,tail_cap=100.,
    applies_to='stochastic_PPO_only',base_terms_unchanged=13)

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

def learning_entry(rows):
    all_rows=[rows['native']]+rows['batch']
    return (rows['native']['metrics']['worlds']==45 and rows['native']['audits']['status']=='PASS'
        and len(rows['batch'])==3 and [r['repeat'] for r in rows['batch']]==[1,2,3]
        and all(r['metrics']['worlds']==512 for r in rows['batch'])
        and all(r['learning_admission']['passed'] and wrench_valid(r) for r in all_rows))

def bulk_penalty(peak):
    assert math.isfinite(peak) and peak>=0
    return -min(OBJECTIVE['bulk_cap'],OBJECTIVE['bulk_weight_per_n']*max(0.,peak-OBJECTIVE['bulk_threshold_n']))

def objective(summary):
    rows=summary['cases'];assert [r['world'] for r in rows]==list(range(len(rows)))
    out=[]
    for r in rows:
        base=float(r['reward']);peak=float(r['landing_metrics']['peak_force_n'])
        assert math.isfinite(base)
        bulk,tail=bulk_penalty(peak),tail_penalty(peak)
        out.append(dict(world=r['world'],base_reward=base,peak_force_n=peak,bulk_penalty=bulk,tail_penalty=tail,
            effective_reward=base+bulk+tail,eligible=r['gate_tick']>=0))
    return dict(spec=OBJECTIVE,rows=out)

def amount(total,first):
    assert 0<=total<=BUDGET
    return min(2,BUDGET-total)

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

def decision(rows,total,returns):
    if qualification(rows):return 'ADVANCE'
    if total>=BUDGET:return 'BUDGET_EXHAUSTED'
    if learning_entry(rows):return 'LEARN'
    if impact_only_failure(rows) and returns<MAX_FRONTIER_RETURNS:return 'FRONTIER'
    return 'ENTRY_FAILED'

def reuse_key(request):
    if request['checkpoint']!=request['contract']['source_checkpoint']:return None
    if request['global_updates']!=0:return None
    if request['role']=='initial_frontier' and request['level_index']==0:return 'source_qualified'
    if request['role']=='target' and request['level_index']==2:return 'source_failed'
    return None

def validate_request(r):
    assert r['mode'] in ('qualify','train') and r['reuse_preflight'] is False
    i=r['level_index'];assert 0<=i<len(LEVELS)
    assert (r['before'],r['after'])==tuple(LEVELS[i][1:])
    assert 0<=r['global_updates']<=BUDGET
    if r['mode']=='train':
        assert 0<r['updates']<=2 and r['global_updates']+r['updates']<=BUDGET
        assert r['role'] in ('frontier_train','target_train')
        assert r['entry_qualification'] and r['entry_result_sha256']
        assert isinstance(r['resume_optimizer'],bool) and not r['reuse_source_evidence']
    else:
        assert r['role'] in ('initial_frontier','frontier_check','target')
        assert r['reuse_source_evidence']==(reuse_key(r) is not None)
    return True

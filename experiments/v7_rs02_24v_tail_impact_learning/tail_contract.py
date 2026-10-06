"""Tail-impact training objective; every physical entry and promotion gate unchanged."""
from pathlib import Path
import sys,math
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'v7_rs02_24v_frontier_margin'))
from margin_contract import admission,qualification,learning_admission,learning_entry,metrics,wrench_valid,zero_qualified,impact_only_failure
LEVELS=[(f'uniform{i:03d}',i/1000,i/1000) for i in range(350,-1,-25)]
PRIOR_SHARED_UPDATES=4
BUDGET=128-PRIOR_SHARED_UPDATES
MAX_FRONTIER_RETURNS=3
OBJECTIVE=dict(version='tail_peak_hinge_v1',threshold_n=330.,scale_n=10.,weight=20.,max_penalty=100.,
               applies_to='stochastic_PPO_only',base_terms_unchanged=13)

def penalty(peak):
    assert math.isfinite(peak) and peak>=0
    return -min(OBJECTIVE['max_penalty'],OBJECTIVE['weight']*(max(0.,peak-OBJECTIVE['threshold_n'])/OBJECTIVE['scale_n'])**2)

def objective(summary):
    rows=summary['cases'];assert [r['world'] for r in rows]==list(range(len(rows)))
    out=[]
    for r in rows:
        base=float(r['reward']);peak=float(r['landing_metrics']['peak_force_n'])
        assert math.isfinite(base)
        cost=penalty(peak)
        out.append(dict(world=r['world'],base_reward=base,peak_force_n=peak,tail_penalty=cost,
                        effective_reward=base+cost,eligible=r['gate_tick']>=0))
    return dict(spec=OBJECTIVE,rows=out)

def amount(total,first):
    assert 0<=total<=BUDGET
    return min(2 if first else 8,BUDGET-total)

def decision(rows,total,returns):
    if qualification(rows):return 'ADVANCE'
    if total>=BUDGET:return 'BUDGET_EXHAUSTED'
    if learning_entry(rows):return 'LEARN'
    if impact_only_failure(rows) and returns<MAX_FRONTIER_RETURNS:return 'FRONTIER'
    return 'ENTRY_FAILED'

def validate_request(r):
    assert r['mode'] in ('qualify','train') and not r.get('reuse_preflight')
    i=r['level_index'];assert 0<=i<len(LEVELS)
    assert (r['before'],r['after'])==tuple(LEVELS[i][1:])
    assert 0<=r['global_updates']<=BUDGET
    if r['mode']=='train':
        assert 0<r['updates']<=8 and r['global_updates']+r['updates']<=BUDGET
        assert r['role'] in ('frontier_train','target_train')
        assert r['entry_qualification'] and r['entry_result_sha256']
        assert isinstance(r['resume_optimizer'],bool)
    else:assert r['role'] in ('initial_frontier','frontier_check','target')
    return True

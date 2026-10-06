"""Finite policy-step repair; physical and final qualification functions unchanged."""
from pathlib import Path
import math,sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'v7_rs02_24v_tail_impact_learning'))
from tail_contract import admission,learning_admission,metrics,qualification,learning_entry,zero_qualified,wrench_valid

FRACTIONS=(.75,.5,.25)
LEVELS=[(f'uniform{i:03d}',i/1000,i/1000) for i in range(250,-1,-25)]
PRIOR_SHARED_UPDATES=14
NEW_PPO_UPDATES=0

def blend_parameter(old,new,fraction):
    assert math.isfinite(fraction) and 0<fraction<1
    return old+fraction*(new-old)

def validate_request(r):
    assert r['mode']=='qualify' and r['global_updates']==0 and r['reuse_preflight'] is False
    i=r['level_index'];assert 0<=i<len(LEVELS)
    assert (r['before'],r['after'])==tuple(LEVELS[i][1:])
    candidate=r['candidate'];j=candidate['index']
    assert 0<=j<len(FRACTIONS) and candidate['fraction']==FRACTIONS[j]
    assert r['checkpoint']==candidate['checkpoint']
    assert r['role']==('filter' if i==0 else 'withdraw')
    assert candidate['actor_hash'] and candidate['checkpoint']['sha256']
    return True

def replay(events):
    """Exact predeclared finite search; no skipped/repeated candidate or level."""
    level=candidate=0;selected=None;promotions=[];stop=None
    for e in events:
        assert stop is None,'Event after terminal decision'
        assert e['level_index']==level and e['candidate_index']==candidate
        assert type(e['qualified']) is bool
        if e['qualified']:
            selected=candidate
            promotions.append((level,candidate))
            level+=1
            if level==len(LEVELS):stop='ZERO_ASSISTANCE_REACHED'
        elif level==0:
            candidate+=1
            if candidate==len(FRACTIONS):stop='NO_FEASIBLE_STEP'
        else:stop='TARGET_FAILED'
    return dict(level_index=level,candidate_index=candidate,selected=selected,promotions=promotions,stop_reason=stop)

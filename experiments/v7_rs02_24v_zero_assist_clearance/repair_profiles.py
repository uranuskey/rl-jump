"""Finite motor-offset screen, with a minimum measured tracking improvement."""
from copy import deepcopy
import math

WINDOWS=(1.25,1.375,1.5,1.75,2.0)
MIN_IMPROVEMENT_MM=.5

def candidates(base):
    assert base['slot_action_bound']==WINDOWS[0]
    out=[]
    for w in WINDOWS:
        p=deepcopy(base)
        p.update(name=f'zero_slot_window_{int(w*1000):04d}',slot_action_bound=w)
        out.append(p)
    return out

def validate_profile(base,p):
    assert set(p)==set(base)
    assert p['slot_action_bound'] in WINDOWS
    assert p['name']==f"zero_slot_window_{int(p['slot_action_bound']*1000):04d}"
    assert all(p[k]==v for k,v in base.items() if k not in ('name','slot_action_bound'))

def choose(rows):
    assert tuple(r['profile']['slot_action_bound'] for r in rows)==WINDOWS
    errors=[r['max_slot_error_mm'] for r in rows]
    assert all(math.isfinite(e) and e>=0 for e in errors)
    for r in rows[1:]:
        if (r['strict_admission']['passed'] and r['external_valid']
                and r['max_slot_error_mm']<=errors[0]-MIN_IMPROVEMENT_MM):
            return r
    return None

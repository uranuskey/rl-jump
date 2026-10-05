"""Finite residual-range comparison; no change to gains or motor envelope."""
from copy import deepcopy
BOUNDS=(1.,1.25,1.5,2.)
def candidates(base):
    rows=[]
    for bound in BOUNDS:
        p=deepcopy(base)
        p.update(name='slot_allocation_'+str(int(bound*100)),slot_action_bound=bound)
        rows.append(p)
    return rows
def validate_profile(base,p):
    assert set(p)==set(base)|{'slot_action_bound'}
    assert all(base[k]==p[k] for k in base if k!='name')
    assert p['slot_action_bound'] in BOUNDS
    return True
def choose(rows):
    # A material tracking improvement is required before independent qualification.
    ceiling=rows[0]['max_slot_error_mm']-1.
    eligible=[r for r in rows[1:] if r['strict_admission']['passed']
              and r['max_slot_error_mm']<=ceiling]
    return eligible[0] if eligible else None

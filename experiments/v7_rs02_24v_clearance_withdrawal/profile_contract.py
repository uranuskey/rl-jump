"""Bounded gain-only slot-tracking candidates; original guards and trajectory remain."""
from copy import deepcopy
CANDIDATES=[('original',3000.,32.,40.),('slot125',3750.,40.,50.),
            ('slot150',4500.,48.,60.),('slot175',5250.,56.,70.),
            ('damped150',4500.,64.,60.),('slot200',6000.,64.,80.)]

def candidates(base):
    out=[]
    for name,k,d,cap in CANDIDATES:
        p=deepcopy(base)
        p.update(name='clearance_'+name,slot_kx=k,slot_dx=d,slot_force_cap_n=cap)
        if name=='original':p=deepcopy(base)
        out.append(p)
    return out

def validate_profile(original,changed):
    allowed={'name','slot_kx','slot_dx','slot_force_cap_n'}
    assert set(original)==set(changed)
    assert all(original[k]==changed[k] for k in original if k not in allowed)
    assert 3000<=changed['slot_kx']<=6000 and 32<=changed['slot_dx']<=64
    assert 40<=changed['slot_force_cap_n']<=80
    return True

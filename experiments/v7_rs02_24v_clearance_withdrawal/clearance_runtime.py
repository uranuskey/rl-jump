"""Declared controller-profile transfer, immutable physics and historical evidence."""
from copy import deepcopy
from pathlib import Path
import time
import diagnostic_runtime as diagnostic
import search_profiles
from diagnostic_runtime import read,write,sha,now,metrics,exclusive
from profile_contract import candidates,validate_profile
from clearance_contract import LEVELS,BUDGET,PRIOR_UPDATES
from curriculum_runtime import exploration
from compliant_runtime import resource_limit
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def verify():
    assert diagnostic.verify()=='43f0fcf6bde80f2c3f67f1ba54916269c2bbf103bc64be612c156aaeaa679665'
    assert search_profiles.verify()=='055cc1fb6a27df53b24cfd2f5925e232cdf9137ebbd07505be017b35731bfd76'
    f=read(HERE/'FROZEN.json')
    for name,digest in f['sha256'].items():assert sha(ROOT/name)==digest,name
    return sha(HERE/'FROZEN.json')

def contract():
    c,ck=diagnostic.source();c=deepcopy(c)
    folder=HERE/'runs/search_01';s=read(folder/'result.json')
    receipt=read(HERE/'runs/search_01_launcher/exit_receipt.json')
    assert receipt['process_exited'] and receipt['exit_code']==0
    assert s['status']=='SCREENED' and s['source_checkpoint']==ck and s['selected'] is not None
    assert s['frozen_sha256']==s['final_frozen_sha256']==search_profiles.verify()
    expected=candidates(c['profile'])
    assert [x['profile'] for x in s['candidates']]==expected
    # Recompute screen admissions and the declared least-gain selection.
    from transfer_contract import admission
    screen=read(folder/'screen.json');assert sha(folder/'screen.json')==s['screen_sha256']
    from collections import Counter
    for i,x in enumerate(s['candidates']):
        rows=screen['cases'][45*i:45*(i+1)]
        m=metrics(dict(cases=rows,reasons=dict(Counter(r['reason'] for r in rows))))
        assert m==x['metrics']
        pre=max(r['constraint_residual_maxima']['pre_apex_v_rad_s'] for r in rows)
        for key,learn in [('learning_admission',True),('strict_admission',False)]:
            assert admission(m,c['anchors_native'],pre,learn)==x[key]
    eligible=[x for x in s['candidates'][1:] if x['strict_admission']['passed']
              and x['max_slot_error_mm']<s['candidates'][0]['max_slot_error_mm']]
    assert eligible and s['selected']==eligible[0]
    profile=s['selected']['profile'];validate_profile(c['profile'],profile)
    c.update(source_checkpoint=ck,source_profile=c['profile'],profile=profile,
        profile_search=dict(folder=str(folder),sha256=sha(folder/'result.json')),
        prior_updates=PRIOR_UPDATES,total_update_budget=BUDGET,
        levels=[list(x) for x in LEVELS],retained_previous_qualification='adaptive_withdrawal/course_01/uniform550')
    return c

def limit_for(started,seconds=43200):
    old=resource_limit(started,seconds)
    def limit():
        assert not (HERE/'STOP').exists(),'Clearance withdrawal STOP requested'
        old()
    return limit

def checked(folder):
    folder=Path(folder);d=read(folder/'result.json');r=read(folder/'exit_receipt.json')
    assert r['process_exited'] and r['exit_code']==0
    assert d['status']=='COMPLETED' and d['frozen_sha256']==d['final_frozen_sha256']==verify()
    return d

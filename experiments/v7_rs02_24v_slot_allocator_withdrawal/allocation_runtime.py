"""Frozen allocator transfer and a shared remaining PPO budget."""
from copy import deepcopy
from pathlib import Path
import allocation_bootstrap as boot
from allocation_bootstrap import HERE,ROOT,read,write,sha,now,metrics,exclusive,limit_for
from allocation_profiles import candidates,validate_profile,choose
from allocation_contract import LEVELS,BUDGET,PRIOR_UPDATES
from curriculum_runtime import exploration
def verify():
    search_hash=boot.verify_search()
    f=read(HERE/'FROZEN.json')
    assert f['search_frozen_sha256']==search_hash
    for name,digest in f['sha256'].items():assert sha(ROOT/name)==digest,name
    return sha(HERE/'FROZEN.json')
def contract():
    c,ck=boot.source();c=deepcopy(c)
    folder=HERE/'runs/search_01';s=read(folder/'result.json')
    receipt=read(HERE/'runs/search_01_launcher/exit_receipt.json')
    assert receipt['process_exited'] and receipt['exit_code']==0
    assert s['status']=='SCREENED' and s['source_checkpoint']==ck
    assert s['frozen_sha256']==s['final_frozen_sha256']==boot.verify_search()
    expected=candidates(c['profile'])
    assert [r['profile'] for r in s['candidates']]==expected
    from transfer_contract import admission
    from collections import Counter
    screen=read(folder/'screen.json');assert sha(folder/'screen.json')==s['screen_sha256']
    for i,r in enumerate(s['candidates']):
        rows=screen['cases'][45*i:45*(i+1)]
        m=metrics(dict(cases=rows,reasons=dict(Counter(x['reason'] for x in rows))))
        assert m==r['metrics']
        pre=max(x['constraint_residual_maxima']['pre_apex_v_rad_s'] for x in rows)
        for key,learning in [('strict_admission',False),('learning_admission',True)]:
            assert admission(m,c['anchors_native'],pre,learning)==r[key]
    assert s['selected']==choose(s['candidates']) and s['selected'] is not None,'No admissible allocator candidate'
    selected=s['selected']['profile'];validate_profile(c['profile'],selected)
    c.update(profile=selected,source_checkpoint=ck,prior_updates=PRIOR_UPDATES,
        total_update_budget=BUDGET,levels=[list(x) for x in LEVELS],
        allocator_search=dict(folder=str(folder),sha256=sha(folder/'result.json')),
        retained_previous_qualification='clearance_withdrawal/course_01/uniform550',
        revision='slot-specific bounded residual window; motor envelope and FIFO unchanged')
    return c
def checked(folder):
    folder=Path(folder);d=read(folder/'result.json');r=read(folder/'exit_receipt.json')
    assert r['process_exited'] and r['exit_code']==0
    assert d['status']=='COMPLETED' and d['frozen_sha256']==d['final_frozen_sha256']==verify()
    return d

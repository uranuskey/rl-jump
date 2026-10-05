"""Freeze a screened motor-offset change; strict qualifications are unchanged."""
from pathlib import Path
import repair_bootstrap as boot
from repair_bootstrap import HERE,ROOT,read,write,sha,now,metrics,exclusive,exploration,verify,limit_for
from repair_contract import LEVELS,BUDGET,PRIOR_UPDATES

def contract():
    c,ck=boot.source();folder=HERE/'runs/screen_01'
    from repair_screen_audit import inspect
    d=inspect(folder,c,ck);a=read(folder/'screen_audit.json')
    assert a['status']=='SCREEN_AUDITED' and a['result_sha256']==sha(folder/'result.json')
    receipt=read(folder/'audit_exit_receipt.json')
    assert receipt['process_exited'] and receipt['exit_code']==0
    assert d['selected'] is not None,'No improved admissible screen candidate; parent repair required'
    c.update(profile=d['selected']['profile'],source_checkpoint=ck,prior_updates=PRIOR_UPDATES,
        total_update_budget=BUDGET,levels=[list(x) for x in LEVELS],
        clearance_screen=dict(folder=str(folder),sha256=sha(folder/'result.json')),
        revision='bounded motor offset selection plus deterministic pose evidence; physical limits unchanged')
    return c

def checked(folder):
    folder=Path(folder);d=read(folder/'result.json');r=read(folder/'exit_receipt.json')
    assert r['process_exited'] and r['exit_code']==0
    assert d['status']=='COMPLETED' and d['frozen_sha256']==d['final_frozen_sha256']==verify()
    return d

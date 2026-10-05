"""A separate frozen continuation of the fully qualified uniform50% controller."""
from pathlib import Path
from copy import deepcopy
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PARENT=HERE.parent/'v7_rs02_24v_slot_allocator_withdrawal'
sys.path.insert(0,str(PARENT))
import allocation_runtime_v2 as parent
from allocation_runtime_v2 import read,write,sha,now,metrics,exclusive,exploration
from zero_contract import LEVELS,BUDGET,PRIOR_UPDATES
PARENT_SHA='86258351f0ca2142272543adad66666e7c85c286670c697e0fd6e52f487af49c'

def verify():
    assert parent.verify()==PARENT_SHA
    f=read(HERE/'FROZEN.json');assert f['parent_frozen_sha256']==PARENT_SHA
    for name,digest in f['sha256'].items():assert sha(ROOT/name)==digest,name
    return sha(HERE/'FROZEN.json')

def contract():
    c=deepcopy(parent.contract());folder=PARENT/'runs/course_02'
    d=parent.checked(folder);a=read(folder/'audit_result.json')
    assert a['status']=='AUDITED' and a['result_sha256']==sha(folder/'result.json')
    assert read(folder/'audit_exit_receipt.json')['exit_code']==0
    wrapper=read(PARENT/'runs/course_02_launcher/wrapper_exit_receipt.json')
    assert wrapper['process_exited'] and wrapper['exit_code']==0
    assert d['deepest_qualified']=='uniform500' and d['stop_reason']=='PLANNED_FLOOR_REACHED'
    last=d['qualified_levels'][-1];assert last['before']==last['after']==.5
    assert sha(last['checkpoint']['path'])==last['checkpoint']['sha256']
    from allocation_audit_v2 import check_qualification
    qfolder=Path(last['qualification_folder']);q=parent.checked(qfolder)
    check_qualification(qfolder,read(qfolder/'request.json'),q)
    assert q['qualified'] and sha(qfolder/'result.json')==last['qualification_result_sha256']
    c.update(source_checkpoint=last['checkpoint'],levels=[list(x) for x in LEVELS],
        total_update_budget=BUDGET,prior_updates=PRIOR_UPDATES,
        qualified_source=dict(folder=str(folder),result_sha256=sha(folder/'result.json'),
            audit_sha256=sha(folder/'audit_result.json')),
        retained_previous_qualification='slot_allocator_withdrawal/course_02/uniform500',
        revision='uniform assistance to zero; unchanged allocation/physics/reward; all-body wrench measurements',
        autonomous_repair_authorized=True,
        external_wrench_completion='zero actual body wrench and free-base generalized force throughout native and three512')
    return c

def limit_for(started,seconds=43200):
    inherited=parent.limit_for(started,seconds)
    def check():
        assert not (HERE/'STOP').exists(),'Zero assistance course STOP requested'
        inherited()
    return check

def checked(folder):
    folder=Path(folder);d=read(folder/'result.json');r=read(folder/'exit_receipt.json')
    assert r['process_exited'] and r['exit_code']==0
    assert d['status']=='COMPLETED' and d['frozen_sha256']==d['final_frozen_sha256']==verify()
    return d


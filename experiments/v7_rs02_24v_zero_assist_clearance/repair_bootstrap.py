"""Preserve the completed failed course and immutable 45% fallback."""
from pathlib import Path
from copy import deepcopy
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PARENT=HERE.parent/'v7_rs02_24v_zero_assist_curriculum'
sys.path.insert(0,str(PARENT))
import zero_runtime as parent
from zero_runtime import read,write,sha,now,metrics,exclusive,exploration
PARENT_SHA='27d91b18a3dcbbdedbb1b2e7d386bc61be99dd6dca9a8306542c99be3f7532e1'
PARENT_RESULT='1a092b3a8b37d4b3ec66e865217bb090eba741d7bf6a1bb08994e9f52b1f6d2a'

def verify():
    assert parent.verify()==PARENT_SHA
    f=read(HERE/'FROZEN.json');assert f['parent_frozen_sha256']==PARENT_SHA
    for name,digest in f['sha256'].items():assert sha(ROOT/name)==digest,name
    return sha(HERE/'FROZEN.json')

def source():
    c=deepcopy(parent.contract());folder=PARENT/'runs/course_01'
    d=parent.checked(folder);a=read(folder/'audit_result.json')
    assert a['status']=='AUDITED' and a['result_sha256']==sha(folder/'result.json')==PARENT_RESULT
    assert read(folder/'audit_exit_receipt.json')['exit_code']==0
    receipt=read(PARENT/'runs/course_01_launcher/wrapper_exit_receipt.json')
    assert receipt['process_exited'] and receipt['exit_code']==0
    assert d['deepest_qualified']=='uniform450' and d['stop_reason']=='ENTRY_FAILED'
    assert d['completed_updates']==d['actor_steps']==0
    last=d['qualified_levels'][-1];assert last['before']==last['after']==.45
    ck=last['checkpoint'];assert sha(ck['path'])==ck['sha256']==c['source_checkpoint']['sha256']
    from zero_audit import check_qualification
    qfolder=Path(last['qualification_folder']);q=parent.checked(qfolder)
    check_qualification(qfolder,read(qfolder/'request.json'),q)
    assert q['qualified'] and sha(qfolder/'result.json')==last['qualification_result_sha256']
    failed=Path(d['events'][-1]['folder']);f=parent.checked(failed)
    assert f['request']['before']==f['request']['after']==.425
    assert not f['qualified'] and not f['learning_entry']
    c.update(retained_previous_qualification=str(qfolder),
        retained_failure=dict(folder=str(failed),sha256=sha(failed/'result.json')))
    return c,ck

def limit_for(started,seconds=43200):
    inherited=parent.limit_for(started,seconds)
    def check():
        assert not (HERE/'STOP').exists(),'Zero assistance clearance STOP requested'
        inherited()
    return check

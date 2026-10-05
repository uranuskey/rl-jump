"""New learning disposition over immutable failed physics evidence."""
from pathlib import Path
from copy import deepcopy
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PARENT=HERE.parent/'v7_rs02_24v_zero_assist_clearance'
sys.path.insert(0,str(PARENT))
import repair_runtime as parent
from repair_runtime import read,write,sha,now,exclusive,exploration
from rare_contract import LEVELS,BUDGET,PRIOR_UPDATES,metrics,learning_admission
PARENT_SHA='6db46199d2df6522ff3faa6ec9778f70af4875543b38ef06df7e1e56af1d2677'
PARENT_RESULT='868001f7d3fddb081972844109155af0dcd413df5ba1c204b4a864f5f239d5cc'
FAILED_RESULT='5469ba65d6d4e00235e672fcde2b8cb6910f9370059e5d2a5466e4ce2593245a'

def verify():
    assert parent.verify()==PARENT_SHA
    f=read(HERE/'FROZEN.json');assert f['parent_frozen_sha256']==PARENT_SHA
    for name,digest in f['sha256'].items():assert sha(ROOT/name)==digest,name
    return sha(HERE/'FROZEN.json')

def contract():
    c=deepcopy(parent.contract());folder=PARENT/'runs/course_02'
    d=parent.checked(folder);a=read(folder/'audit_result.json')
    assert a['status']=='AUDITED' and a['result_sha256']==sha(folder/'result.json')==PARENT_RESULT
    assert read(folder/'audit_exit_receipt.json')['exit_code']==0
    receipt=read(PARENT/'runs/course_02_launcher/wrapper_exit_receipt.json')
    assert receipt['process_exited'] and receipt['exit_code']==0
    assert d['deepest_qualified']=='uniform450' and d['stop_reason']=='ENTRY_FAILED'
    assert d['completed_updates']==d['actor_steps']==0
    last=d['qualified_levels'][-1];assert last['before']==last['after']==.45
    assert last['checkpoint']==c['source_checkpoint']
    from repair_audit import check_qualification
    qfolder=Path(last['qualification_folder']);q=parent.checked(qfolder)
    check_qualification(qfolder,read(qfolder/'request.json'),q)
    assert q['qualified'] and sha(qfolder/'result.json')==last['qualification_result_sha256']
    failed=Path(d['events'][-1]['folder']);f=parent.checked(failed)
    check_qualification(failed,read(failed/'request.json'),f)
    assert sha(failed/'result.json')==FAILED_RESULT
    assert f['request']['before']==f['request']['after']==.425
    assert not f['qualified'] and not f['learning_entry']
    for native,row in [(True,f['qualification']['native'])]+[(False,x) for x in f['qualification']['batch']]:
        s=read(Path(row['evidence_dir'])/row['summary'])
        assert metrics(s)==row['metrics']==parent.metrics(s)
        anchors=c['anchors_native'] if native else c['anchors_batch']
        assert learning_admission(s,anchors,row['pre_apex_v_rad_s'])['passed']
    c.update(retained_previous_qualification=str(qfolder),
        reused_initial_qualification=dict(folder=str(failed),result_sha256=FAILED_RESULT,old_qualified=False,old_learning_entry=False),
        total_update_budget=BUDGET,prior_updates=PRIOR_UPDATES,levels=[list(x) for x in LEVELS],
        revision='simulation learning entry admits at most2 bounded post-touch self contacts in512; strict promotion unchanged',
        simulation_training_contact_failure_limit=2,simulation_training_peak_limit_n=350.,
        no_controller_reward_physics_changes=True,old_failure_preserved=True)
    return c

def limit_for(started,seconds=43200):
    inherited=parent.limit_for(started,seconds)
    def check():
        assert not (HERE/'STOP').exists(),'Rare-contact learning STOP requested'
        inherited()
    return check

def checked(folder):
    folder=Path(folder);d=read(folder/'result.json');r=read(folder/'exit_receipt.json')
    assert r['process_exited'] and r['exit_code']==0
    assert d['status']=='COMPLETED' and d['frozen_sha256']==d['final_frozen_sha256']==verify()
    return d

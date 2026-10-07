"""Isolated pre-apex wheel feedback following the preserved zero-assist failure."""
from pathlib import Path
import hashlib,json,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PARENT=HERE.parent/'v7_rs02_24v_zero_assist_snapshot'
sys.path.insert(0,str(PARENT))
import zsnap_runtime as parent
from zsnap_runtime import read,write,sha,now,exclusive,metrics,exploration
from twpd_control import PROFILE
PARENT_SHA='f2e791ffc50a7ba303b05c5acf173307ff62ffe000959874baf2c6e07ef79910'

def verify():
    assert parent.verify()==PARENT_SHA
    f=read(HERE/'FROZEN.json');assert f['parent_frozen_sha256']==PARENT_SHA
    for name,digest in f['sha256'].items():assert sha(ROOT/name)==digest,name
    return sha(HERE/'FROZEN.json')

def contract():
    c=parent.contract()
    run=PARENT/'runs/course_01';folder=run/'level_00_u0000_zero_probe'
    r=read(folder/'result.json')
    assert r['status']=='ERROR_STOPPED' and 'zero-size array' in r['error']
    assert r['request']['before']==r['request']['after']==0 and r['request']['checkpoint']==c['source_checkpoint']
    receipts=[]
    for path in [folder/'exit_receipt.json',run/'exit_receipt.json',PARENT/'runs/course_01_launcher/wrapper_exit_receipt.json']:
        e=read(path);assert e['process_exited'] and e['exit_code']==1
        receipts.append(dict(path=str(path),sha256=sha(path),exit_code=1))
    assert not (run/'audit_result.json').exists()
    expected={'native.json':'be959ffda93918e9fc6dab20a004c18aca9f3ea35b10d503576fc8e0c53990e6',
        'native_traces.npz':'49b01987feb3c9aa9395f396bce36ae0bdfa53066e3eb724243ebca1d98b2afc',
        'native_poses.npz':'5aaf85601ebc1751b5647f7060f5ef0c722d5d450bb0b8f3a0918e1ae71b7647'}
    for name,digest in expected.items():assert sha(folder/name)==digest
    s=read(folder/'native.json');assert s['worlds']==45 and s['passed']==0
    identity=dict(actor=c['source_actor_hash'],landing_profile=c['profile'],takeoff_feedback=PROFILE)
    c.update(takeoff_feedback=PROFILE,controller_identity=identity,
        controller_id=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        zero_failure=dict(folder=str(folder),files=expected,receipts=receipts,actual_worlds=45,original_exit_code=1,original_audited=False),
        controller_and_physics_unchanged=False,physics_unchanged=True,takeoff_network_unchanged=True,
        landing_actor_apex_gate_unchanged=True,standing_actor_before_handoff_unchanged=True,
        native_first_stop_if_not_strict=True,revision='One fixed causal bounded wheel pitch PD through existing FIFO and real motor limiter; no PPO')
    return c

def checked(folder):
    folder=Path(folder);r=read(folder/'result.json');e=read(folder/'exit_receipt.json')
    assert e['process_exited'] and e['exit_code']==0 and r['status']=='COMPLETED'
    assert r['frozen_sha256']==r['final_frozen_sha256']==verify()
    return r

def limit_for(started,seconds=43200):
    inherited=parent.limit_for(started,seconds)
    def check():
        assert not (HERE/'STOP').exists(),'Wheel feedback STOP requested'
        inherited()
    return check

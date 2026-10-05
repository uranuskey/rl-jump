"""An independent curriculum; prior controlled-stop evidence stays immutable."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PARENT=HERE.parent/'v7_rs02_24v_rotor6125_learning'
sys.path.insert(0,str(PARENT))
import transfer_runtime as parent
from transfer_runtime import read,write,sha,now,metrics,exclusive,exploration
from compliant_runtime import resource_limit
from curriculum_contract import LEVELS,BUDGET

PARENT_SHA='c977dd472dfe668233a96719a73f22179a8310a955c6216d11afe437da46b357'


def verify():
    assert parent.verify()==PARENT_SHA
    f=read(HERE/'FROZEN.json'); assert f['parent_transfer_sha256']==PARENT_SHA
    for path,digest in f['sha256'].items():
        assert sha(ROOT/path)==digest,'Adaptive curriculum source changed: '+path
    return sha(HERE/'FROZEN.json')


def contract():
    c=parent.contract()
    pre=PARENT/'runs/train_01_preflight'; smoke=PARENT/'runs/train_01_smoke'
    p=parent.checked(pre,'PREFLIGHT_COMPLETED'); s=parent.checked(smoke,'SMOKE_COMPLETED')
    assert s['completed_updates']==2 and s['actor_steps']==16
    assert p['native']['strict_admission']['passed'] and p['batch']['strict_admission']['passed']
    old=PARENT/'runs/train_01'
    stopped=read(old/'result.json'); receipt=read(old/'exit_receipt.json')
    assert stopped['status']=='ERROR_STOPPED' and stopped['error']=="AssertionError('Rotor6125 adaptation STOP requested')"
    assert stopped['completed_updates']==0 and stopped['actor_steps']==0
    assert receipt['process_exited'] and receipt['exit_code']==1
    c.update(reused_preflight=dict(folder=str(pre),sha256=sha(pre/'result.json')),
        reused_smoke=dict(folder=str(smoke),sha256=sha(smoke/'result.json'),updates=2,actor_steps=16),
        prior_controlled_stop=dict(folder=str(old),sha256=sha(old/'result.json'),exit_sha256=sha(old/'exit_receipt.json')),
        levels=[list(x) for x in LEVELS],total_update_budget=BUDGET)
    c['source_validated_before']=c.pop('before');c['source_validated_after']=c.pop('after')
    return c


def limit_for(started,seconds=43200):
    old=resource_limit(started,seconds)
    def limit():
        assert not (HERE/'STOP').exists(),'Adaptive withdrawal STOP requested'
        old()
    return limit


def checked(folder):
    folder=Path(folder); d=read(folder/'result.json'); r=read(folder/'exit_receipt.json')
    assert r['process_exited'] and r['exit_code']==0
    assert d['status']=='COMPLETED' and d['frozen_sha256']==d['final_frozen_sha256']==verify()
    return d

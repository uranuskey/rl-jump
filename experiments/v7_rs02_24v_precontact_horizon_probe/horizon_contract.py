"""Predeclared controller timing candidates; every physical gate is inherited."""
from copy import deepcopy
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'v7_rs02_24v_frontier_admission_recovery'))
import recovery_contract as parent
from recovery_contract import qualification,admission,learning_admission,metrics,wrench_valid,zero_qualified,OBJECTIVE
LEVELS=parent.LEVELS[1:]
HORIZONS=(0.035,0.040,0.045)
PRIOR_SHARED_UPDATES=44
REMAINING_SHARED_UPDATES=84

def profile(source,index):
    assert 0<=index<len(HORIZONS)
    p=deepcopy(source)
    assert p['name']=='zero_slot_window_1500' and p['horizon_s']==0.030
    p.update(name=f'zero_slot_precontact_h{round(HORIZONS[index]*1000):02d}',horizon_s=HORIZONS[index])
    return p

def controller_id(actor,p):
    return hashlib.sha256(json.dumps(dict(actor_hash=actor,profile=p),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def validate_request(r):
    assert r['mode']=='qualify' and r['role'] in ('candidate','advance')
    assert r['global_updates']==0 and not r['reuse_source_evidence'] and not r['reuse_preflight']
    i,j=r['candidate_index'],r['level_index'];assert 0<=i<len(HORIZONS) and 0<=j<len(LEVELS)
    assert (r['before'],r['after'])==tuple(LEVELS[j][1:])
    c=r['contract'];assert c['profile']==profile(c['source_profile'],i)
    assert r['checkpoint']==c['source_checkpoint']
    assert r['controller_id']==controller_id(c['source_actor_hash'],c['profile'])
    assert (r['role']=='candidate')==(j==0)
    return True

def transition(index,level,passed):
    if passed:return ('COMPLETE',index,level) if level==len(LEVELS)-1 else ('ADVANCE',index,level+1)
    if level==0 and index+1<len(HORIZONS):return 'NEXT_CANDIDATE',index+1,0
    return ('NO_PROFILE_QUALIFIED' if level==0 else 'TARGET_FAILED'),index,level

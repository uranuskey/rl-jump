"""Retain the failed v1 audit and freeze the corrected event integration separately."""
from pathlib import Path
import allocation_runtime as original
from allocation_runtime import HERE,ROOT,read,write,sha,now,metrics,exclusive,limit_for,exploration
ORIGINAL_FROZEN='61f454374af9ff420780ee0c7fb6236628b35f28f866dcc1eec6b0e3ccd3acb3'
def verify():
    assert original.verify()==ORIGINAL_FROZEN
    f=read(HERE/'FROZEN_V2.json')
    assert f['original_frozen_sha256']==ORIGINAL_FROZEN
    for name,digest in f['sha256'].items():assert sha(ROOT/name)==digest,name
    return sha(HERE/'FROZEN_V2.json')
def contract():
    c=original.contract()
    c['integration_revision']='restore inherited causal apex bookkeeping before allocation control'
    c['preserved_failed_course']='course_01'
    return c
def checked(folder):
    folder=Path(folder);d=read(folder/'result.json');r=read(folder/'exit_receipt.json')
    assert r['process_exited'] and r['exit_code']==0
    assert d['status']=='COMPLETED' and d['frozen_sha256']==d['final_frozen_sha256']==verify()
    return d

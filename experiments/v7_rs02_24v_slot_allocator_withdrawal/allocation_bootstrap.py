"""Verify the completed parent and a finite immutable screening protocol."""
import allocation_paths as paths
from allocation_paths import HERE,ROOT,read,write,sha,now,metrics,exclusive,previous
PARENT_SHA='533a5dcd77e9b9ff1bd11b53e6298d1e5183f9806c99e41c681e2f4902274f41'
def verify_search():
    assert previous.verify()==PARENT_SHA
    f=read(HERE/'SEARCH_FROZEN.json')
    assert f['parent_frozen_sha256']==PARENT_SHA
    for name,digest in f['sha256'].items():assert sha(ROOT/name)==digest,name
    return sha(HERE/'SEARCH_FROZEN.json')
def source():
    c=previous.contract()
    run=previous.HERE/'runs/course_01'
    d=previous.checked(run);a=read(run/'audit_result.json')
    assert a['status']=='AUDITED' and a['result_sha256']==sha(run/'result.json')
    assert read(run/'audit_exit_receipt.json')['exit_code']==0
    assert d['deepest_qualified']=='uniform550' and d['stop_reason']=='ENTRY_FAILED'
    assert d['completed_updates']==0 and d['actor_steps']==0
    ck=d['qualified_levels'][-1]['checkpoint']
    assert sha(ck['path'])==ck['sha256']==c['source_checkpoint']['sha256']
    return c,ck
def limit_for(started,seconds=43200):
    from compliant_runtime import resource_limit
    limit=resource_limit(started,seconds)
    def check():
        assert not (HERE/'STOP').exists(),'Slot allocation STOP requested'
        limit()
    return check

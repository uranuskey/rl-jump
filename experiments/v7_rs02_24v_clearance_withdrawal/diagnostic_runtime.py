"""Read-only physical diagnosis over the completed, frozen assistance course."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'v7_rs02_24v_adaptive_withdrawal'))
import curriculum_runtime as parent
from curriculum_runtime import read,write,sha,now,metrics,exclusive

PARENT_SHA='1e94416d624d0e48632977f679f110d5d23c8d1c3324f97a00cf82ea812a4a09'

def verify():
    assert parent.verify()==PARENT_SHA
    f=read(HERE/'DIAGNOSTICS_FROZEN.json')
    assert f['parent_course_sha256']==PARENT_SHA
    for name,digest in f['sha256'].items():
        assert sha(ROOT/name)==digest,name
    return sha(HERE/'DIAGNOSTICS_FROZEN.json')

def source():
    run=parent.HERE/'runs/course_01'
    d=parent.checked(run);a=read(run/'audit_result.json')
    assert a['status']=='AUDITED' and a['result_sha256']==sha(run/'result.json')
    assert d['deepest_qualified']=='uniform550' and d['stop_reason']=='ENTRY_FAILED'
    assert d['completed_updates']==2 and d['actor_steps']==16
    ck=d['qualified_levels'][-1]['checkpoint']
    assert sha(ck['path'])==ck['sha256']
    return d['contract'],ck

"""Independent profile/identity, original gates and ordered-course audit."""
import argparse
from pathlib import Path
import late_runtime as rt
from late_contract import LEVELS,HORIZONS,qualification,validate_request,controller_id,zero_qualified,transition

def check_qualification(folder,req,r):
    from boundary_row_audit import check_row
    validate_request(req);assert r['request']==req and req['contract']==rt.case_contract(req['candidate_index'])
    assert r['mode']=='qualify' and r['actor_hash']==rt.SOURCE_ACTOR
    assert rt.sha(req['checkpoint']['path'])==req['checkpoint']['sha256']==rt.SOURCE_SHA
    assert r['controller_id']==req['controller_id']==controller_id(r['actor_hash'],req['contract']['profile'])
    rows=r['qualification'];assert rows['native']['metrics']['worlds']==45
    assert len(rows['batch'])==3 and [x['repeat'] for x in rows['batch']]==[1,2,3]
    for native,row in [(True,rows['native'])]+[(False,x) for x in rows['batch']]:
        assert row['metrics']['worlds']==(45 if native else 512) and row['sample_kind']=='deterministic_evaluation'
        assert (row['before'],row['after'])==(req['before'],req['after']) and Path(row['evidence_dir'])==folder
        assert rt.read(folder/row['summary'])['slot_profile']==req['contract']['profile']
        check_row(folder,row,req['contract'],native)
    assert r['new_physical_trials']==1581 and r['planned_physical_trials']==1581
    assert r['completed_updates']==r['actor_steps']==0 and r['training_disabled'] and not r['learning_entry']
    assert r['qualified']==qualification(rows)
    out=dict(status='AUDITED',utc=rt.now(),result_sha256=rt.sha(folder/'result.json'),actor_hash=r['actor_hash'],controller_id=r['controller_id'],
        profile=req['contract']['profile'],qualified=r['qualified'],new_physical_trials=1581,completed_updates=0,actor_steps=0)
    path=folder/'qualification_audit.json'
    if path.exists():
        old=rt.read(path);assert {k:v for k,v in old.items() if k!='utc'}=={k:v for k,v in out.items() if k!='utc'}
    else:rt.write(path,out)
    return out

def audit(run):
    d=rt.checked(run);assert d['contract']==rt.contract()
    index=level=0;seen=set();promotions=[];selected=None;stop=None
    for pos,event in enumerate(d['events']):
        assert stop is None
        folder=Path(event['folder']);r=rt.checked(folder);req=r['request']
        assert event['result_sha256']==rt.sha(folder/'result.json') and event['mode']=='qualify'
        assert (req['candidate_index'],req['level_index'])==(index,level)
        key=(req['controller_id'],req['before'],req['after']);assert key not in seen;seen.add(key)
        check_qualification(folder,req,r)
        if r['qualified']:
            selected=index
            promotions.append(dict(name=LEVELS[level][0],before=req['before'],after=req['after'],profile=req['contract']['profile'],
                checkpoint=req['checkpoint'],actor_hash=r['actor_hash'],controller_id=r['controller_id'],at_global_update=0,
                qualification_folder=str(folder),qualification_result_sha256=event['result_sha256']))
        action,index,level=transition(index,level,r['qualified'])
        assert event['decision']==action
        if action not in ('NEXT_CANDIDATE','ADVANCE'):
            assert pos==len(d['events'])-1;stop=action
    assert stop==d['stop_reason'] and promotions==d['qualified_levels'] and selected==d['selected_candidate_index']
    assert d['completed_updates']==d['actor_steps']==0 and d['new_physical_trials']==1581*len(d['events'])
    assert d['zero_assistance_qualified']==zero_qualified(promotions)
    assert (promotions[-1]['name'] if promotions else None)==d['deepest_qualified']
    assert (stop=='COMPLETE')==d['zero_assistance_qualified']
    out=dict(status='AUDITED',utc=rt.now(),result_sha256=rt.sha(run/'result.json'),frozen_sha256=rt.verify(),
        completed_updates=0,actor_steps=0,updates_in_128_budget=44,shared_remaining=84,qualified_levels=promotions,
        stop_reason=stop,deepest_qualified=d['deepest_qualified'],selected_candidate_index=selected,
        profile_change_only=True,final_physical_gates_unchanged=True,new_physical_trials=d['new_physical_trials'],
        zero_assistance_qualified=d['zero_assistance_qualified'],hardware_qualified=False)
    rt.write(run/'audit_result.json',out);return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)

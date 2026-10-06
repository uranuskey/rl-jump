"""Recompute weights, deterministic gates and every finite-search transition."""
import argparse
from pathlib import Path
import filter_runtime as rt
from filter_contract import LEVELS,FRACTIONS,replay,qualification,learning_entry,zero_qualified,validate_request
from filter_candidates import verify_candidate

def check_qualification(folder,request,result):
    from rare_row_audit import check_row
    validate_request(request)
    assert result['request']==request and result['mode']=='qualify'
    assert result['actor_hash']==request['candidate']['actor_hash']
    verify_candidate(request['candidate'],request['contract'])
    rows=result['qualification'];assert 'reused_qualification' not in result
    assert rows['native']['metrics']['worlds']==45 and len(rows['batch'])==3
    for native,row in [(True,rows['native'])]+[(False,x) for x in rows['batch']]:
        check_row(Path(row['evidence_dir']),row,request['contract'],native)
        assert (row['before'],row['after'])==(request['before'],request['after'])
        assert row['sample_kind']=='deterministic_evaluation'
    assert qualification(rows)==result['qualified'] and learning_entry(rows)==result['learning_entry']
    out=dict(status='AUDITED',utc=rt.now(),result_sha256=rt.sha(folder/'result.json'),
             qualified=result['qualified'],learning_entry=result['learning_entry'],actor_hash=result['actor_hash'])
    path=folder/'qualification_audit.json'
    if path.exists():
        old=rt.read(path);assert {k:v for k,v in old.items() if k!='utc'}=={k:v for k,v in out.items() if k!='utc'}
    else:rt.write(path,out)
    return out

def promotion(candidate,index,folder):
    return dict(name=LEVELS[index][0],before=LEVELS[index][1],after=LEVELS[index][2],
        candidate_index=candidate['index'],fraction=candidate['fraction'],actor_hash=candidate['actor_hash'],
        checkpoint=candidate['checkpoint'],qualification_folder=str(folder),
        qualification_result_sha256=rt.sha(folder/'result.json'),at_global_update=0,level_updates=0)

def audit(run):
    d=rt.checked(run);assert d['contract']==rt.contract()
    assert d['completed_updates']==d['actor_steps']==0
    assert d['candidates']==rt.read(run/'candidates.json') and len(d['candidates'])==len(FRACTIONS)
    assert [c['index'] for c in d['candidates']]==list(range(len(FRACTIONS)))
    for candidate in d['candidates']:verify_candidate(candidate,d['contract'])
    assert len({c['actor_hash'] for c in d['candidates']})==len(FRACTIONS)
    events=[];promotions=[];tested=set()
    for event in d['events']:
        schedule=replay(events);assert schedule['stop_reason'] is None
        folder=Path(event['folder']);r=rt.checked(folder);req=rt.read(folder/'request.json')
        candidate=d['candidates'][schedule['candidate_index']];i=schedule['level_index']
        assert req==r['request'] and req['contract']==d['contract'] and req['candidate']==candidate
        assert req['level_index']==i and event['result_sha256']==rt.sha(folder/'result.json')
        assert event['candidate_index']==candidate['index'] and event['level_index']==i
        assert event['qualified']==r['qualified'] and event['actor_hash']==candidate['actor_hash']
        assert (i,r['actor_hash']) not in tested;tested.add((i,r['actor_hash']))
        check_qualification(folder,req,r);events.append(event)
        if r['qualified']:promotions.append(promotion(candidate,i,folder))
    end=replay(events);assert end['stop_reason']==d['stop_reason'] and end['stop_reason'] is not None
    assert promotions==d['qualified_levels']
    assert d['deepest_qualified']==(promotions[-1]['name'] if promotions else None)
    assert d['selected_candidate']==(d['candidates'][end['selected']] if end['selected'] is not None else None)
    assert d['zero_assistance_qualified']==zero_qualified(promotions)
    assert d['zero_assistance_qualified']==(d['stop_reason']=='ZERO_ASSISTANCE_REACHED')
    out=dict(status='AUDITED',utc=rt.now(),result_sha256=rt.sha(run/'result.json'),frozen_sha256=rt.verify(),
             completed_updates=0,actor_steps=0,prior_shared_updates=14,remaining_shared_updates=114,
             deepest_qualified=d['deepest_qualified'],stop_reason=d['stop_reason'],selected_candidate=d['selected_candidate'],
             qualified_levels=promotions,zero_assistance_qualified=d['zero_assistance_qualified'],hardware_qualified=False,
             interpretation='finite post-training actor step selection, not new PPO learning',
             physical_and_final_gates_unchanged=True,old_failures_preserved=True,estimated_24V=True)
    rt.write(run/'audit_result.json',out);return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    print(audit(args.run),flush=True)

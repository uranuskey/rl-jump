"""Reconstruct the complete qualification before allowing each transition."""
import argparse
from pathlib import Path
import margin_runtime as rt
from margin_contract import LEVELS,BUDGET,qualification,learning_entry,decision,chunk_size,zero_qualified,validate_request,warmup_request,WARMUP_UPDATES,PRIOR_UPDATES


def check_qualification(folder,request,result):
    from rare_row_audit import check_row
    assert result['request']==request and result['mode']=='qualify'
    rows=result['qualification']; c=request['contract']
    validate_request(request)
    assert 'reused_qualification' not in result
    assert rows['native']['metrics']['worlds']==45
    for native,row in [(True,rows['native'])]+[(False,x) for x in rows['batch']]:
        check_row(Path(row['evidence_dir']),row,c,native)
        assert (row['before'],row['after'])==(request['before'],request['after'])
        assert row['sample_kind']=='deterministic_evaluation'
    assert qualification(rows)==result['qualified']
    assert learning_entry(rows)==result['learning_entry']
    out=dict(status='AUDITED',utc=rt.now(),result_sha256=rt.sha(folder/'result.json'),
        qualified=result['qualified'],learning_entry=result['learning_entry'],actor_hash=result['actor_hash'])
    path=folder/'qualification_audit.json'
    if path.exists():
        old=rt.read(path)
        assert {k:v for k,v in old.items() if k!='utc'}=={k:v for k,v in out.items() if k!='utc'}
    else: rt.write(path,out)
    return out

def check_training(folder,request,result,contract,expected_updates,expected_actor):
    from rare_row_audit import check_row
    assert result['completed_updates']==request['updates']==expected_updates
    assert result['actor_hash_before']==expected_actor and result['actor_hash_after']!=expected_actor
    counted=0
    assert len(result['updates'])==expected_updates
    for index,entry in enumerate(result['updates'],1):
        assert entry['global_update']==request['global_updates']+index
        assert entry['sample']['sample_kind']=='stochastic_training'
        check_row(folder,entry['sample'],contract)
        counted+=entry['stats']['actor_steps']
    assert counted==result['actor_steps'] and counted>0
    assert rt.sha(result['checkpoint']['path'])==result['checkpoint']['sha256']
    return counted


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True); args=p.parse_args()
    run=args.run; d=rt.read(run/'result.json'); receipt=rt.read(run/'exit_receipt.json')
    assert receipt['process_exited'] and receipt['exit_code']==0 and d['status']=='COMPLETED'
    assert d['contract']==rt.contract() and d['frozen_sha256']==d['final_frozen_sha256']==rt.verify()
    total=steps=level=level_updates=0; incumbent=d['contract']['source_checkpoint']
    deepest=None; expected_mode='qualify'; failed_actor=None; promotions=[]; tested=set()
    expected_actor=d['contract']['source_actor_hash']
    assert d['events'] and d['events'][0]['mode']=='warmup_train'
    for event_index,event in enumerate(d['events']):
        folder=Path(event['folder']); r=rt.checked(folder); req=rt.read(folder/'request.json')
        assert event['result_sha256']==rt.sha(folder/'result.json')
        assert req==r['request'];validate_request(req)
        if event['mode']=='warmup_train':
            assert event_index==0 and req==warmup_request(d['contract'],d['frozen_sha256'])
            steps=check_training(folder,req,r,d['contract'],WARMUP_UPDATES,expected_actor)
            total=WARMUP_UPDATES;incumbent=r['checkpoint'];expected_actor=r['actor_hash_after']
            continue
        assert event_index>0
        assert req==r['request'] and req['mode']==expected_mode
        assert req['level_index']==level and req['checkpoint']==incumbent
        assert (req['before'],req['after'])==tuple(LEVELS[level][1:])
        assert req['global_updates']==total
        if req['mode']=='qualify':
            assert r['actor_hash']==expected_actor
            assert r['actor_hash'] not in tested;tested.add(r['actor_hash'])
            check_qualification(folder,req,r)
            action=decision(r['qualification'],total); assert event['decision']==action
            if action=='ADVANCE':
                deepest=LEVELS[level][0]
                promotions.append(dict(name=deepest,before=req['before'],after=req['after'],checkpoint=incumbent,
                    at_global_update=total,level_updates=level_updates,qualification_folder=str(folder),
                    qualification_result_sha256=rt.sha(folder/'result.json')))
                level+=1;level_updates=0;failed_actor=None;tested=set()
            elif action=='LEARN':
                expected_mode='train'; failed_actor=r['actor_hash']
            else:
                assert event==d['events'][-1] and d['stop_reason']==action
        else:
            assert req['updates']==chunk_size(total,level_updates)==r['completed_updates']
            assert req['resume_optimizer']==(level_updates>0)
            assert r['actor_hash_after'] not in tested
            counted=check_training(folder,req,r,d['contract'],req['updates'],failed_actor)
            steps+=counted; total+=r['completed_updates']; level_updates+=r['completed_updates']
            incumbent=r['checkpoint']; assert rt.sha(incumbent['path'])==incumbent['sha256']
            expected_actor=r['actor_hash_after']
            expected_mode='qualify'
    assert total==d['completed_updates']<=BUDGET and steps==d['actor_steps']
    assert deepest==d['deepest_qualified']
    assert promotions==d['qualified_levels']
    zero=zero_qualified(promotions)
    assert d['zero_assistance_qualified']==zero
    if level==len(LEVELS): assert d['stop_reason']=='ZERO_ASSISTANCE_REACHED' and zero
    rt.write(run/'audit_result.json',dict(status='AUDITED',utc=rt.now(),frozen_sha256=rt.verify(),
        result_sha256=rt.sha(run/'result.json'),completed_updates=total,actor_steps=steps,
        deepest_qualified=deepest,qualified_levels=d['qualified_levels'],stop_reason=d['stop_reason'],
        shared_update_budget=BUDGET,prior_updates=PRIOR_UPDATES,combined_updates=total+PRIOR_UPDATES,
        updates_in_128_budget=total+d['contract']['previous_course_updates'],
        warmup_updates=WARMUP_UPDATES,retained_qualified_strength=.35,early_graduation=True,all_failed_attempts_retained=True,
        estimated_24V=True,zero_assistance_qualified=zero,hardware_qualified=False))


if __name__=='__main__': main()

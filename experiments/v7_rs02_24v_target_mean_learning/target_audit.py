"""Recompute physical gates, every PPO objective and the complete curriculum ledger."""
import argparse,math
from pathlib import Path
import target_runtime as rt
from target_contract import LEVELS,BUDGET,OBJECTIVE,PRIOR_SHARED_UPDATES,qualification,learning_entry,learning_evidence,decision,amount,zero_qualified,validate_request,objective,reuse_key,EXPLORATION_OFFSET

def check_qualification(folder,request,result):
    from balanced_row_audit import check_row
    assert result['request']==request and result['mode']=='qualify'
    validate_request(request);rows=result['qualification'];c=request['contract']
    assert rows['native']['metrics']['worlds']==45
    if request['reuse_source_evidence']:
        key=reuse_key(request);assert key is not None
        old=rt.source_evidence(c,key)
        assert result['reused_qualification']==c[key] and result['new_physical_trials']==0
        assert rows==old['qualification'] and result['actor_hash']==old['actor_hash']
        assert result['original_source_learning_entry']==old['learning_entry'] is False
        assert (old['request']['before'],old['request']['after'])==(request['before'],request['after'])
    else:
        assert 'reused_qualification' not in result and result['new_physical_trials']==45+3*512
    for native,row in [(True,rows['native'])]+[(False,x) for x in rows['batch']]:
        check_row(Path(row['evidence_dir']),row,c,native)
        assert (row['before'],row['after'])==(request['before'],request['after'])
        assert row['sample_kind']=='deterministic_evaluation'
    assert qualification(rows)==result['qualified'] and learning_entry(rows,c)==result['learning_entry']
    assert learning_evidence(rows,c)==result['learning_evidence']
    out=dict(status='AUDITED',utc=rt.now(),result_sha256=rt.sha(folder/'result.json'),qualified=result['qualified'],
             learning_entry=result['learning_entry'],learning_evidence=result['learning_evidence'],actor_hash=result['actor_hash'])
    path=folder/'qualification_audit.json'
    if path.exists():
        old=rt.read(path);assert {k:v for k,v in old.items() if k!='utc'}=={k:v for k,v in out.items() if k!='utc'}
    else:rt.write(path,out)
    return out

def check_training(folder,req,r,actor):
    from balanced_row_audit import check_row
    import numpy as np
    assert r['completed_updates']==req['updates'] and r['actor_hash_before']==actor
    assert r['actor_hash_after']!=actor and r['actor_steps']>0 and len(r['updates'])==req['updates']
    steps=0
    for offset,e in enumerate(r['updates'],1):
        assert e['global_update']==req['global_updates']+offset
        row=e['sample'];assert row['sample_kind']=='stochastic_training'
        check_row(folder,row,req['contract'])
        stored=rt.read(folder/e['objective']['file'])
        assert rt.sha(folder/e['objective']['file'])==e['objective']['sha256']
        expected=objective(rt.read(folder/row['summary']))
        assert stored==expected
        values=np.asarray([x['effective_reward'] for x in stored['rows'] if x['eligible']],dtype=np.float32)*np.float32(.01)
        assert len(values)==e['stats']['eligible_samples']
        assert math.isclose(float(values.mean()),e['stats']['mean_return'],abs_tol=2e-5,rel_tol=2e-5)
        assert math.isfinite(e['stats']['max_accepted_kl']) and e['stats']['max_accepted_kl']<=.03
        ck=e['checkpoint'];assert rt.sha(ck['path'])==ck['sha256'] and ck['global_update']==e['global_update']
        import torch
        from slot_learning import Policy
        from target_worker import actor_hash
        state=torch.load(ck['path'],map_location='cpu',weights_only=True)
        assert state['target_frozen_sha256']==r['frozen_sha256'] and state['global_update']==e['global_update']
        policy=Policy('cpu');policy.load_state_dict(state['model_state_dict'],strict=True)
        assert actor_hash(policy)==e['actor_hash_after']
        assert state['profile']==req['contract']['profile'] and state['voltage_v']==24 and state['action_dim']==16
        assert state['level_index']==req['level_index'] and state['before']==req['before'] and state['after']==req['after']
        rt.exploration(policy,EXPLORATION_OFFSET+e['global_update']-1)
        assert torch.equal(policy.std,state['model_state_dict']['std'])
        steps+=e['stats']['actor_steps']
    assert steps==r['actor_steps'] and r['checkpoint']==r['updates'][-1]['checkpoint']
    assert r['actor_hash_after']==r['updates'][-1]['actor_hash_after']
    assert rt.sha(r['checkpoint']['path'])==r['checkpoint']['sha256']
    return steps

def audit(run):
    d=rt.checked(run);assert d['contract']==rt.contract()
    total=steps=target=local=0;last_training=None;tested=set();promotions=[];entry=None;stopped=False
    checkpoint=d['contract']['source_checkpoint'];actor=d['contract']['source_actor_hash'];expected='qualify'
    for number,event in enumerate(d['events']):
        assert not stopped and target<len(LEVELS)
        folder=Path(event['folder']);r=rt.checked(folder);req=rt.read(folder/'request.json');validate_request(req)
        assert req==r['request'] and rt.sha(folder/'result.json')==event['result_sha256']
        assert req['mode']==event['mode']==expected and req['role']==event['role']
        assert req['checkpoint']==checkpoint and req['global_updates']==total and req['level_index']==target
        assert rt.sha(checkpoint['path'])==checkpoint['sha256']
        if expected=='qualify':
            assert r['actor_hash']==actor and (target,actor) not in tested
            tested.add((target,actor));check_qualification(folder,req,r);entry=(folder,r)
            action=decision(r['qualification'],d['contract'],total,local);assert event['decision']==action
            if action=='ADVANCE':
                promotions.append(dict(name=LEVELS[target][0],before=req['before'],after=req['after'],checkpoint=checkpoint,
                    at_global_update=total,level_updates=local,qualification_folder=str(folder),qualification_result_sha256=rt.sha(folder/'result.json')))
                target+=1;local=0
            elif action=='LEARN':expected='train'
            else:
                assert number==len(d['events'])-1 and d['stop_reason']==action;stopped=True
        else:
            ef,q=entry;assert q['learning_entry']
            assert req['entry_qualification']==str(ef) and req['entry_result_sha256']==rt.sha(ef/'result.json')
            assert req['updates']==amount(total) and req['resume_optimizer']==(last_training==target)
            steps+=check_training(folder,req,r,actor);total+=r['completed_updates'];local+=r['completed_updates']
            checkpoint=r['checkpoint'];actor=r['actor_hash_after'];last_training=target;expected='qualify'
            assert all((i,actor) not in tested for i in range(len(LEVELS)))
    assert total==d['completed_updates']<=BUDGET and steps==d['actor_steps']
    assert promotions==d['qualified_levels'] and d['deepest_qualified']==(promotions[-1]['name'] if promotions else None)
    assert d['zero_assistance_qualified']==zero_qualified(promotions)
    if target==len(LEVELS):assert d['stop_reason']=='ZERO_ASSISTANCE_REACHED' and d['zero_assistance_qualified']
    else:assert stopped
    result=dict(status='AUDITED',utc=rt.now(),result_sha256=rt.sha(run/'result.json'),frozen_sha256=rt.verify(),
        completed_updates=total,actor_steps=steps,prior_shared_updates=PRIOR_SHARED_UPDATES,
        shared_update_budget=BUDGET,updates_in_128_budget=total+PRIOR_SHARED_UPDATES,
        stop_reason=d['stop_reason'],deepest_qualified=d['deepest_qualified'],qualified_levels=promotions,
        training_objective=OBJECTIVE,learning_entry_changed=True,final_gates_unchanged=True,
        old_failures_preserved=True,zero_assistance_qualified=d['zero_assistance_qualified'],hardware_qualified=False)
    rt.write(run/'audit_result.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();audit(a.run)
if __name__=='__main__':main()

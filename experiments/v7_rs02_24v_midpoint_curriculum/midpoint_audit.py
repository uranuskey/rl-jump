"""Recompute physical gates, every PPO objective and the complete curriculum ledger."""
import argparse,math
from pathlib import Path
import midpoint_runtime as rt
from midpoint_contract import LEVELS,BUDGET,OBJECTIVE,PRIOR_SHARED_UPDATES,qualification,learning_entry,learning_evidence,decision,amount,zero_qualified,validate_request,objective,reuse_key,EXPLORATION_OFFSET,ACTOR_LR,MAX_ACCEPTED_KL,MAX_FRONTIER_UPDATES,frontier_action

def check_qualification(folder,request,result):
    from midpoint_row_audit import check_row
    assert result['request']==request and result['mode']=='qualify'
    validate_request(request);rows=result['qualification'];c=request['contract']
    assert rows['native']['metrics']['worlds']==45
    assert len(rows['batch'])==3 and [r['repeat'] for r in rows['batch']]==[1,2,3]
    if request['reuse_source_evidence']:
        key=reuse_key(request);assert key is not None
        old=rt.source_evidence(c,key)
        assert result['reused_qualification']==c[key] and result['new_physical_trials']==0
        assert rows==old['qualification'] and result['actor_hash']==old['actor_hash']
        assert result['original_source_learning_entry']==old['learning_entry']
        assert result['original_source_qualified']==old['qualified']
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
    from midpoint_row_audit import check_row
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
        assert math.isfinite(e['stats']['max_accepted_kl']) and e['stats']['max_accepted_kl']<=MAX_ACCEPTED_KL
        ck=e['checkpoint'];assert rt.sha(ck['path'])==ck['sha256'] and ck['global_update']==e['global_update']
        import torch
        from slot_learning import Policy
        from midpoint_worker import actor_hash
        state=torch.load(ck['path'],map_location='cpu',weights_only=True)
        assert state['midpoint_frozen_sha256']==r['frozen_sha256'] and state['global_update']==e['global_update']
        policy=Policy('cpu');policy.load_state_dict(state['model_state_dict'],strict=True)
        assert actor_hash(policy)==e['actor_hash_after']
        assert state['profile']==req['contract']['profile'] and state['voltage_v']==24 and state['action_dim']==16
        assert 0<e['stats']['actor_lr_next']<=ACTOR_LR and 0<e['stats']['accepted_actor_lr_min']<=ACTOR_LR
        assert all(0<g['lr']<=ACTOR_LR for g in state['actor_optimizer']['param_groups'])
        assert state['level_index']==req['level_index'] and state['before']==req['before'] and state['after']==req['after']
        rt.exploration(policy,EXPLORATION_OFFSET+e['global_update']-1)
        assert torch.equal(policy.std,state['model_state_dict']['std'])
        steps+=e['stats']['actor_steps']
    assert steps==r['actor_steps'] and r['checkpoint']==r['updates'][-1]['checkpoint']
    assert r['actor_hash_after']==r['updates'][-1]['actor_hash_after']
    assert rt.sha(r['checkpoint']['path'])==r['checkpoint']['sha256']
    return steps

def audit(run):
    d=rt.read(run/'result.json');receipt=rt.read(run/'exit_receipt.json')
    assert receipt['process_exited'] and receipt['exit_code']==0 and d['status']=='COMPLETED'
    assert d['contract']==rt.contract() and d['frozen_sha256']==d['final_frozen_sha256']==rt.verify()
    total=steps=target=returns=used=0;local={};known={};tested=set();promotions=[]
    checkpoint=d['contract']['source_checkpoint'];actor=d['contract']['source_actor_hash']
    expected='initial_recovery';last_training=None;stopped=False;must_update=False
    def promote(i,folder,req):
        promotions.append(dict(name=LEVELS[i][0],before=req['before'],after=req['after'],checkpoint=checkpoint,
            at_global_update=total,level_updates=local.get(i,0),qualification_folder=str(folder),
            qualification_result_sha256=rt.sha(folder/'result.json')))
    for number,event in enumerate(d['events']):
        assert not stopped and event['role']==expected
        frontier=0 if target==0 else target-1
        if event['mode']=='route':
            assert expected=='frontier_route'
            entry=known.get((frontier,actor));assert entry is not None
            folder,q=entry
            assert event['folder']==str(folder) and event['result_sha256']==rt.sha(folder/'result.json')
            assert event['level_index']==frontier and event['target_index']==target
            assert event['at_global_update']==total and event['frontier_updates']==used and event['must_update']==must_update
            action=frontier_action(q['qualification'],d['contract'],total,used,must_update)
            assert event['decision']==action
            if action=='FRONTIER_READY':
                assert q['qualified'] and not must_update
                if target==0:promote(0,folder,q['request']);target=1;used=returns=0
                expected='target'
            elif action=='TRAIN_QUALIFIED_FRONTIER':expected='frontier_train'
            elif action=='TRAIN_BOUNDED_RECOVERY':expected='frontier_recovery_train'
            else:
                assert number==len(d['events'])-1 and d['stop_reason']==action;stopped=True
            continue
        folder=Path(event['folder']);r=rt.checked(folder);req=rt.read(folder/'request.json');validate_request(req)
        assert req==r['request'] and event['result_sha256']==rt.sha(folder/'result.json')
        assert event['role']==req['role'] and req['checkpoint']==checkpoint and req['global_updates']==total
        i=req['level_index'];role=req['role']
        assert i==(target if role in ('target','target_train') else frontier)
        if req['mode']=='qualify':
            assert event['mode']=='qualify' and r['actor_hash']==actor and (i,actor) not in tested
            tested.add((i,actor));known[(i,actor)]=(folder,r);check_qualification(folder,req,r)
            if role in ('initial_recovery','frontier_check','frontier_recovery_check'):
                expected='frontier_route'
            else:
                action=decision(r['qualification'],d['contract'],total,returns,local.get(i,0))
                assert event['decision']==action
                if action=='ADVANCE':promote(i,folder,req);target+=1;used=returns=0;expected='target'
                elif action=='LEARN':expected='target_train'
                elif action=='FRONTIER':
                    returns+=1;must_update=True
                    expected='frontier_route' if (i-1,actor) in known else 'frontier_check'
                else:
                    assert number==len(d['events'])-1 and d['stop_reason']==action;stopped=True
        else:
            assert event['mode']=='train'
            entry=known.get((i,actor));assert entry is not None
            entryfolder,q=entry
            assert req['entry_qualification']==str(entryfolder) and req['entry_result_sha256']==rt.sha(entryfolder/'result.json')
            assert q['learning_entry']
            if role=='frontier_train':assert q['qualified']
            if role=='frontier_recovery_train':assert not q['qualified']
            assert req['updates']==amount(total) and req['resume_optimizer']==(last_training==i)
            steps+=check_training(folder,req,r,actor)
            total+=r['completed_updates'];local[i]=local.get(i,0)+r['completed_updates']
            checkpoint=r['checkpoint'];actor=r['actor_hash_after'];last_training=i
            assert all((j,actor) not in tested for j in range(len(LEVELS)))
            if role in ('frontier_train','frontier_recovery_train'):
                used+=1;assert used<=MAX_FRONTIER_UPDATES;must_update=False;expected='frontier_recovery_check'
            else:expected='target'
    assert total==d['completed_updates']<=BUDGET and steps==d['actor_steps']
    assert promotions==d['qualified_levels']
    assert (promotions[-1]['name'] if promotions else None)==d['deepest_qualified']
    assert d['zero_assistance_qualified']==zero_qualified(promotions)
    if target==len(LEVELS):assert d['stop_reason']=='ZERO_ASSISTANCE_REACHED' and d['zero_assistance_qualified']
    else:assert stopped,'Unexpected unfinished successful ledger'
    out=dict(status='AUDITED',utc=rt.now(),result_sha256=rt.sha(run/'result.json'),frozen_sha256=rt.verify(),
        completed_updates=total,actor_steps=steps,shared_update_budget=BUDGET,prior_shared_updates=PRIOR_SHARED_UPDATES,
        updates_in_128_budget=total+PRIOR_SHARED_UPDATES,deepest_qualified=d['deepest_qualified'],stop_reason=d['stop_reason'],
        qualified_levels=promotions,training_objective=OBJECTIVE,physical_and_final_gates_unchanged=True,
        existing_learning_routes_unchanged=True,frontier_strict_before_descent=True,old_failures_preserved=True,
        estimated_24V=True,zero_assistance_qualified=d['zero_assistance_qualified'],hardware_qualified=False)
    rt.write(run/'audit_result.json',out)
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();audit(a.run)
if __name__=='__main__':main()

"""Learn a margin at a strictly qualified higher-assistance frontier."""
import argparse,re,sys,subprocess,traceback
from pathlib import Path
import timing_runtime as rt
from timing_contract import LEVELS,BUDGET,OBJECTIVE,amount,decision,qualification,zero_qualified,validate_request,reuse_key,frontier_action
from timing_audit import check_qualification

def worker(run,request,result):
    validate_request(request)
    name=f"level_{request['level_index']:02d}_u{request['global_updates']:04d}_{request['role']}"
    folder=run/name;assert not folder.exists();folder.mkdir()
    rt.write(folder/'request.json',request)
    result.update(stage=name,active_worker_folder=str(folder));rt.write(run/'progress.json',result)
    started=rt.now()
    with (folder/'stdout.log').open('w',encoding='utf-8') as stdout,(folder/'stderr.log').open('w',encoding='utf-8') as stderr:
        child=subprocess.Popen([sys.executable,'-X','faulthandler','-u',str(rt.HERE/'timing_worker.py'),'--request',str(folder/'request.json')],
                               stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        result['active_worker_pid']=child.pid;rt.write(run/'progress.json',result)
        code=child.wait()
    rt.write(folder/'exit_receipt.json',dict(process_exited=True,exit_code=code,pid=child.pid,utc_start=started,utc_end=rt.now()))
    assert code==0,('Recovery worker failed',name,code)
    r=rt.checked(folder);assert r['request']==request
    return folder,r

def execute(run,result):
    checkpoint=result['contract']['source_checkpoint'];actor=result['contract']['source_actor_hash']
    total=steps=0;last_training_level=None;known={};tested=set();level_updates={}
    result.update(events=[],qualified_levels=[],deepest_qualified=None,completed_updates=0,actor_steps=0)
    def request(i,mode,role,**extra):
        _,before,after=LEVELS[i]
        r=dict(mode=mode,role=role,contract=result['contract'],frozen_sha256=result['frozen_sha256'],
            level_index=i,before=before,after=after,checkpoint=checkpoint,global_updates=total,reuse_preflight=False,**extra)
        r['reuse_source_evidence']=mode=='qualify' and reuse_key(r) is not None
        return r
    def qualify(i,role):
        key=(i,actor);assert key not in tested,'Identical actor at same level cannot be repeatedly tested'
        folder,q=worker(run,request(i,'qualify',role),result)
        assert q['actor_hash']==actor;tested.add(key)
        check_qualification(folder,q['request'],q)
        result['events'].append(dict(folder=str(folder),mode='qualify',role=role,result_sha256=rt.sha(folder/'result.json')))
        known[key]=(folder,q);rt.write(run/'progress.json',result)
        return folder,q
    def train(i,role,entry):
        nonlocal checkpoint,actor,total,steps,last_training_level
        folder,q=entry
        assert q['actor_hash']==actor and q['request']['level_index']==i
        assert q['learning_entry']
        if role=='frontier_train':assert q['qualified']
        if role=='frontier_recovery_train':assert not q['qualified']
        req=request(i,'train',role,updates=amount(total),resume_optimizer=last_training_level==i,
            entry_qualification=str(folder),entry_result_sha256=rt.sha(folder/'result.json'))
        tf,t=worker(run,req,result)
        assert t['actor_hash_before']==actor and t['actor_hash_after']!=actor and t['actor_steps']>0
        assert t['completed_updates']==1
        actor=t['actor_hash_after'];assert all((j,actor) not in tested for j in range(len(LEVELS)))
        checkpoint=t['checkpoint'];total+=1;steps+=t['actor_steps'];last_training_level=i
        level_updates[i]=level_updates.get(i,0)+1
        result['events'].append(dict(folder=str(tf),mode='train',role=role,result_sha256=rt.sha(tf/'result.json')))
        result.update(completed_updates=total,actor_steps=steps,latest_checkpoint=checkpoint)
        rt.write(run/'progress.json',result)
    def restore_frontier(i,target,entry,must_update,used):
        while True:
            folder,q=entry
            action=frontier_action(q['qualification'],result['contract'],total,used,must_update)
            result['events'].append(dict(mode='route',role='frontier_route',folder=str(folder),
                result_sha256=rt.sha(folder/'result.json'),level_index=i,target_index=target,
                at_global_update=total,frontier_updates=used,must_update=must_update,decision=action))
            rt.write(run/'progress.json',result)
            if action=='FRONTIER_READY':
                assert q['qualified'];return entry,used
            if action not in ('TRAIN_QUALIFIED_FRONTIER','TRAIN_BOUNDED_RECOVERY'):
                result['stop_reason']=action;return None,used
            role='frontier_train' if action=='TRAIN_QUALIFIED_FRONTIER' else 'frontier_recovery_train'
            train(i,role,entry);used+=1;must_update=False
            # Every changed frontier actor must pass its own complete strict gate before descent.
            entry=qualify(i,'frontier_recovery_check')
    def promote(i,entry):
        folder,q=entry;assert q['qualified']
        row=dict(name=LEVELS[i][0],before=LEVELS[i][1],after=LEVELS[i][2],checkpoint=checkpoint,
            at_global_update=total,level_updates=level_updates.get(i,0),qualification_folder=str(folder),
            qualification_result_sha256=rt.sha(folder/'result.json'))
        result['qualified_levels'].append(row);result['deepest_qualified']=row['name']
        rt.write(run/'progress.json',result)
        print(dict(event='qualified_and_advance',**row),flush=True)
    entry=qualify(0,'initial_recovery')
    entry,_=restore_frontier(0,0,entry,False,0)
    if entry is None:return
    promote(0,entry)
    for i in range(1,len(LEVELS)):
        returns=frontier_used=0
        while True:
            entry=qualify(i,'target');q=entry[1]
            action=decision(q['qualification'],result['contract'],total,returns,level_updates.get(i,0))
            result['events'][-1]['decision']=action;rt.write(run/'progress.json',result)
            if action=='ADVANCE':promote(i,entry);break
            if action=='LEARN':train(i,'target_train',entry);continue
            if action=='FRONTIER':
                returns+=1;frontier=i-1
                prior=known.get((frontier,actor))
                if prior is None:prior=qualify(frontier,'frontier_check')
                prior,frontier_used=restore_frontier(frontier,i,prior,True,frontier_used)
                if prior is None:return
                continue
            result['stop_reason']=action;return
    result['stop_reason']='ZERO_ASSISTANCE_REACHED'
    result['zero_assistance_qualified']=zero_qualified(result['qualified_levels'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
    assert re.fullmatch('[A-Za-z0-9_-]+',a.run_id)
    run=rt.HERE/'runs'/a.run_id;assert not run.exists();run.mkdir(parents=True)
    f,c=rt.verify(),rt.contract()
    r=dict(status='RUNNING',utc_start=rt.now(),frozen_sha256=f,contract=c,num_envs=512,
           training_objective=OBJECTIVE,shared_update_budget=BUDGET,early_graduation=True,
           fixed_takeoff_network=True,zero_assistance_qualified=False,hardware_qualified=False)
    rt.write(run/'progress.json',r)
    try:
        execute(run,r);r.update(status='COMPLETED',final_frozen_sha256=rt.verify())
    except BaseException as error:
        r.update(status='ERROR_STOPPED',error=repr(error),traceback=traceback.format_exc())
        print(r['traceback'],flush=True)
    finally:
        r['utc_end']=rt.now()
        if not r['zero_assistance_qualified']:
            rt.write(run/'parent_repair_request.json',dict(status='PARENT_ACTION_REQUIRED',cause=r.get('error',r.get('stop_reason')),
                deepest_qualified=r.get('deepest_qualified'),last_worker_folder=r.get('active_worker_folder'),goal_complete=False,
                autonomous_repair_authorized=True,next_action='Diagnose the unchanged strict gates and continue a new frozen repair.'))
        rt.write(run/'result.json',r);rt.write(run/'progress.json',r)
    return int(r['status']!='COMPLETED')

if __name__=='__main__':raise SystemExit(main())

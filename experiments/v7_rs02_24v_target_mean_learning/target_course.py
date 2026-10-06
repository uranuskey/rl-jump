"""Reuse the failed source once, learn at that target, and promote only strict passes."""
import argparse,re,sys,subprocess,traceback
from pathlib import Path
import target_runtime as rt
from target_contract import LEVELS,BUDGET,OBJECTIVE,amount,decision,zero_qualified,validate_request,reuse_key
from target_audit import check_qualification

def worker(run,request,result):
    validate_request(request)
    name=f"level_{request['level_index']:02d}_u{request['global_updates']:04d}_{request['role']}"
    folder=run/name;assert not folder.exists();folder.mkdir()
    rt.write(folder/'request.json',request)
    result.update(stage=name,active_worker_folder=str(folder));rt.write(run/'progress.json',result)
    started=rt.now()
    with (folder/'stdout.log').open('w',encoding='utf-8') as stdout,(folder/'stderr.log').open('w',encoding='utf-8') as stderr:
        child=subprocess.Popen([sys.executable,'-X','faulthandler','-u',str(rt.HERE/'target_worker.py'),'--request',str(folder/'request.json')],
            stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        result['active_worker_pid']=child.pid;rt.write(run/'progress.json',result);code=child.wait()
    rt.write(folder/'exit_receipt.json',dict(process_exited=True,exit_code=code,pid=child.pid,utc_start=started,utc_end=rt.now()))
    assert code==0,('Target worker failed',name,code)
    r=rt.checked(folder);assert r['request']==request
    return folder,r

def execute(run,result):
    checkpoint=result['contract']['source_checkpoint'];actor=result['contract']['source_actor_hash']
    total=steps=0;last_training=None;tested=set()
    result.update(events=[],qualified_levels=[],deepest_qualified=None,completed_updates=0,actor_steps=0)
    def request(i,mode,**extra):
        _,before,after=LEVELS[i]
        r=dict(mode=mode,role='target' if mode=='qualify' else 'target_train',contract=result['contract'],
            frozen_sha256=result['frozen_sha256'],level_index=i,before=before,after=after,checkpoint=checkpoint,
            global_updates=total,reuse_preflight=False,**extra)
        r['reuse_source_evidence']=reuse_key(r) is not None
        return r
    for i,level in enumerate(LEVELS):
        local=0
        while True:
            assert (i,actor) not in tested,'No repeated qualification for an unchanged failed actor'
            folder,q=worker(run,request(i,'qualify'),result);tested.add((i,actor))
            assert q['actor_hash']==actor;check_qualification(folder,q['request'],q)
            action=decision(q['qualification'],result['contract'],total,local)
            result['events'].append(dict(folder=str(folder),mode='qualify',role='target',decision=action,result_sha256=rt.sha(folder/'result.json')))
            rt.write(run/'progress.json',result)
            if action=='ADVANCE':
                promotion=dict(name=level[0],before=level[1],after=level[2],checkpoint=checkpoint,
                    at_global_update=total,level_updates=local,qualification_folder=str(folder),
                    qualification_result_sha256=rt.sha(folder/'result.json'))
                result['qualified_levels'].append(promotion);result['deepest_qualified']=level[0]
                rt.write(run/'progress.json',result);print(dict(event='qualified_and_advance',**promotion),flush=True);break
            if action!='LEARN':result['stop_reason']=action;return
            assert q['learning_entry']
            req=request(i,'train',updates=amount(total),resume_optimizer=last_training==i,
                entry_qualification=str(folder),entry_result_sha256=rt.sha(folder/'result.json'))
            tf,t=worker(run,req,result)
            assert t['actor_hash_before']==actor and t['actor_hash_after']!=actor and t['actor_steps']>0
            assert t['completed_updates']==req['updates']
            actor=t['actor_hash_after'];assert all((j,actor) not in tested for j in range(len(LEVELS)))
            checkpoint=t['checkpoint'];total+=t['completed_updates'];local+=t['completed_updates'];steps+=t['actor_steps'];last_training=i
            result['events'].append(dict(folder=str(tf),mode='train',role='target_train',result_sha256=rt.sha(tf/'result.json')))
            result.update(completed_updates=total,actor_steps=steps,latest_checkpoint=checkpoint);rt.write(run/'progress.json',result)
    result['stop_reason']='ZERO_ASSISTANCE_REACHED';result['zero_assistance_qualified']=zero_qualified(result['qualified_levels'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();assert re.fullmatch('[A-Za-z0-9_-]+',a.run_id)
    run=rt.HERE/'runs'/a.run_id;assert not run.exists();run.mkdir(parents=True)
    f,c=rt.verify(),rt.contract()
    r=dict(status='RUNNING',utc_start=rt.now(),frozen_sha256=f,contract=c,num_envs=512,
        training_objective=OBJECTIVE,shared_update_budget=BUDGET,early_graduation=True,
        fixed_takeoff_network=True,zero_assistance_qualified=False,hardware_qualified=False)
    rt.write(run/'progress.json',r)
    try:execute(run,r);r.update(status='COMPLETED',final_frozen_sha256=rt.verify())
    except BaseException as error:
        r.update(status='ERROR_STOPPED',error=repr(error),traceback=traceback.format_exc());print(r['traceback'],flush=True)
    finally:
        r['utc_end']=rt.now()
        if not r['zero_assistance_qualified']:
            rt.write(run/'parent_repair_request.json',dict(status='PARENT_ACTION_REQUIRED',cause=r.get('error',r.get('stop_reason')),
                deepest_qualified=r.get('deepest_qualified'),last_worker_folder=r.get('active_worker_folder'),goal_complete=False,
                autonomous_repair_authorized=True,next_action='Diagnose unchanged final gates and continue a new frozen repair.'))
        rt.write(run/'result.json',r);rt.write(run/'progress.json',r)
    return int(r['status']!='COMPLETED')
if __name__=='__main__':raise SystemExit(main())

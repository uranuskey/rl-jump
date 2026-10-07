"""Once-only fixed timing candidates, then automatic strict assistance descent."""
import argparse,re,subprocess,sys,traceback
from pathlib import Path
import horizon_runtime as rt
from horizon_contract import LEVELS,validate_request,controller_id,transition,zero_qualified
from horizon_audit import check_qualification

def execute(run,r):
    index=level=0;seen=set()
    r.update(events=[],qualified_levels=[],selected_candidate_index=None,deepest_qualified=None,completed_updates=0,actor_steps=0,new_physical_trials=0)
    while True:
        c=rt.case_contract(index);_,before,after=LEVELS[level]
        req=dict(mode='qualify',role='candidate' if level==0 else 'advance',candidate_index=index,level_index=level,
            before=before,after=after,contract=c,checkpoint=c['source_checkpoint'],frozen_sha256=r['frozen_sha256'],
            global_updates=0,reuse_source_evidence=False,reuse_preflight=False,controller_id=controller_id(c['source_actor_hash'],c['profile']))
        validate_request(req);key=(req['controller_id'],before,after);assert key not in seen;seen.add(key)
        name=f'profile_{index:02d}_level_{level:02d}_{req["role"]}'
        folder=run/name;assert not folder.exists();folder.mkdir();rt.write(folder/'request.json',req)
        r.update(stage=name,active_worker_folder=str(folder));rt.write(run/'progress.json',r)
        started=rt.now()
        with (folder/'stdout.log').open('w',encoding='utf-8') as stdout,(folder/'stderr.log').open('w',encoding='utf-8') as stderr:
            child=subprocess.Popen([sys.executable,'-X','faulthandler','-u',str(rt.HERE/'horizon_worker.py'),'--request',str(folder/'request.json')],
                stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            r['active_worker_pid']=child.pid;rt.write(run/'progress.json',r);code=child.wait()
        rt.write(folder/'exit_receipt.json',dict(process_exited=True,exit_code=code,pid=child.pid,utc_start=started,utc_end=rt.now()))
        assert code==0,('Worker failed',name,code)
        q=rt.checked(folder);check_qualification(folder,req,q)
        event=dict(mode='qualify',folder=str(folder),result_sha256=rt.sha(folder/'result.json'))
        r['events'].append(event);r['new_physical_trials']+=q['new_physical_trials']
        if q['qualified']:
            r['selected_candidate_index']=index
            r['qualified_levels'].append(dict(name=LEVELS[level][0],before=before,after=after,profile=c['profile'],checkpoint=req['checkpoint'],
                actor_hash=q['actor_hash'],controller_id=q['controller_id'],at_global_update=0,
                qualification_folder=str(folder),qualification_result_sha256=event['result_sha256']))
            r['deepest_qualified']=LEVELS[level][0]
        action,index,level=transition(index,level,q['qualified']);event['decision']=action;rt.write(run/'progress.json',r)
        if action in ('NEXT_CANDIDATE','ADVANCE'):continue
        r['stop_reason']=action;r['zero_assistance_qualified']=zero_qualified(r['qualified_levels']);return

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();assert re.fullmatch('[A-Za-z0-9_-]+',a.run_id)
    run=rt.HERE/'runs'/a.run_id;assert not run.exists();run.mkdir(parents=True)
    r=dict(status='RUNNING',utc_start=rt.now(),frozen_sha256=rt.verify(),contract=rt.contract(),zero_assistance_qualified=False,hardware_qualified=False)
    try:
        execute(run,r);r.update(status='COMPLETED',final_frozen_sha256=rt.verify())
    except BaseException as error:
        r.update(status='ERROR_STOPPED',error=repr(error),traceback=traceback.format_exc());print(r['traceback'],flush=True)
    finally:
        r['utc_end']=rt.now()
        if not r['zero_assistance_qualified']:
            rt.write(run/'parent_repair_request.json',dict(status='PARENT_ACTION_REQUIRED',cause=r.get('error',r.get('stop_reason')),
                deepest_qualified=r.get('deepest_qualified'),preserved_previous_deepest='uniform250',last_worker_folder=r.get('active_worker_folder'),
                goal_complete=False,autonomous_repair_authorized=True,next_action='Diagnose preserved timing candidates; continue a new frozen repair without replaying failed controllers.'))
        rt.write(run/'result.json',r);rt.write(run/'progress.json',r)
    return int(r['status']!='COMPLETED')
if __name__=='__main__':raise SystemExit(main())

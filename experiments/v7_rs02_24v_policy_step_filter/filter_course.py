"""Use the first fully qualified finite candidate, then withdraw assistance."""
import argparse,re,subprocess,sys,traceback
from pathlib import Path
import filter_runtime as rt
from filter_contract import LEVELS,replay,zero_qualified,validate_request
from filter_candidates import prepare
from filter_audit import check_qualification,promotion

def worker(run,request,result):
    validate_request(request)
    name=f"candidate_{request['candidate']['index']:02d}_level_{request['level_index']:02d}_qualify"
    folder=run/name;assert not folder.exists();folder.mkdir()
    rt.write(folder/'request.json',request)
    result.update(stage=name,active_worker_folder=str(folder));rt.write(run/'progress.json',result)
    started=rt.now()
    with (folder/'stdout.log').open('w',encoding='utf-8') as out,(folder/'stderr.log').open('w',encoding='utf-8') as err:
        child=subprocess.Popen([sys.executable,'-X','faulthandler','-u',str(rt.HERE/'filter_worker.py'),'--request',str(folder/'request.json')],
            stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        result['active_worker_pid']=child.pid;rt.write(run/'progress.json',result);code=child.wait()
    rt.write(folder/'exit_receipt.json',dict(process_exited=True,exit_code=code,pid=child.pid,utc_start=started,utc_end=rt.now()))
    assert code==0,('Policy filter worker failed',name,code)
    return folder,rt.checked(folder)

def execute(run,result):
    result.update(candidates=prepare(run,result['contract']),events=[],qualified_levels=[],deepest_qualified=None,
                  selected_candidate=None,completed_updates=0,actor_steps=0)
    tested=set()
    while True:
        schedule=replay(result['events'])
        if schedule['stop_reason']:
            result['stop_reason']=schedule['stop_reason'];break
        index=schedule['level_index'];candidate=result['candidates'][schedule['candidate_index']]
        key=(index,candidate['actor_hash']);assert key not in tested;tested.add(key)
        req=dict(mode='qualify',role='filter' if index==0 else 'withdraw',contract=result['contract'],
            frozen_sha256=result['frozen_sha256'],candidate=candidate,checkpoint=candidate['checkpoint'],
            level_index=index,before=LEVELS[index][1],after=LEVELS[index][2],global_updates=0,reuse_preflight=False)
        folder,q=worker(run,req,result);check_qualification(folder,req,q)
        event=dict(folder=str(folder),level_index=index,candidate_index=candidate['index'],actor_hash=candidate['actor_hash'],
                   qualified=q['qualified'],result_sha256=rt.sha(folder/'result.json'))
        result['events'].append(event)
        if q['qualified']:
            row=promotion(candidate,index,folder)
            result['qualified_levels'].append(row);result['deepest_qualified']=row['name']
            result['selected_candidate']=candidate
            print(dict(event='qualified_and_advance',**row),flush=True)
        rt.write(run/'progress.json',result)
    result['zero_assistance_qualified']=zero_qualified(result['qualified_levels'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
    assert re.fullmatch('[A-Za-z0-9_-]+',a.run_id)
    run=rt.HERE/'runs'/a.run_id;assert not run.exists();run.mkdir(parents=True)
    f,c=rt.verify(),rt.contract()
    r=dict(status='RUNNING',utc_start=rt.now(),frozen_sha256=f,contract=c,num_envs=512,early_graduation=True,
           completed_updates=0,actor_steps=0,zero_assistance_qualified=False,hardware_qualified=False,
           fallback_qualified=c['source_evidence']['level_03_u0002_target'])
    rt.write(run/'progress.json',r)
    try:
        execute(run,r);r.update(status='COMPLETED',final_frozen_sha256=rt.verify())
    except BaseException as error:
        r.update(status='ERROR_STOPPED',error=repr(error),traceback=traceback.format_exc());print(r['traceback'],flush=True)
    finally:
        r['utc_end']=rt.now()
        if not r['zero_assistance_qualified']:
            rt.write(run/'parent_repair_request.json',dict(status='PARENT_ACTION_REQUIRED',cause=r.get('error',r.get('stop_reason')),
                deepest_qualified=r.get('deepest_qualified'),last_worker_folder=r.get('active_worker_folder'),goal_complete=False,
                autonomous_repair_authorized=True,next_action='Inspect finite step evidence, then continue a new frozen repair.'))
        rt.write(run/'result.json',r);rt.write(run/'progress.json',r)
    return int(r['status']!='COMPLETED')

if __name__=='__main__':raise SystemExit(main())

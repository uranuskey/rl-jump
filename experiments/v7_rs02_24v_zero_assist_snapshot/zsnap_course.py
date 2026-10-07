"""One complete native45 + exactly three512 diagnostic at zero assistance."""
import argparse,re,sys,subprocess,traceback
from pathlib import Path
import zsnap_runtime as rt
from zsnap_contract import validate_request
from zsnap_audit import check_qualification
def worker(run,request,result):
    validate_request(request)
    name=f"level_{request['level_index']:02d}_u{request['global_updates']:04d}_{request['role']}"
    folder=run/name;assert not folder.exists();folder.mkdir()
    rt.write(folder/'request.json',request)
    result.update(stage=name,active_worker_folder=str(folder));rt.write(run/'progress.json',result)
    started=rt.now()
    with (folder/'stdout.log').open('w',encoding='utf-8') as stdout,(folder/'stderr.log').open('w',encoding='utf-8') as stderr:
        child=subprocess.Popen([sys.executable,'-X','faulthandler','-u',str(rt.HERE/'zsnap_worker.py'),'--request',str(folder/'request.json')],
                               stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        result['active_worker_pid']=child.pid;rt.write(run/'progress.json',result)
        code=child.wait()
    rt.write(folder/'exit_receipt.json',dict(process_exited=True,exit_code=code,pid=child.pid,utc_start=started,utc_end=rt.now()))
    assert code==0,('Recovery worker failed',name,code)
    r=rt.checked(folder);assert r['request']==request
    return folder,r
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();assert re.fullmatch('[A-Za-z0-9_-]+',a.run_id)
 run=rt.HERE/'runs'/a.run_id;assert not run.exists();run.mkdir(parents=True)
 f,c=rt.verify(),rt.contract()
 r=dict(status='RUNNING',utc_start=rt.now(),frozen_sha256=f,contract=c,num_envs=512,completed_updates=0,actor_steps=0,
   zero_assistance_qualified=False,hardware_qualified=False,events=[],deepest_qualified=None)
 rt.write(run/'progress.json',r)
 try:
  req=dict(mode='qualify',role='zero_probe',contract=c,frozen_sha256=f,level_index=0,before=0.0,after=0.0,
      checkpoint=c['source_checkpoint'],global_updates=0,reuse_preflight=False,reuse_source_evidence=False)
  folder,q=worker(run,req,r);assert q['actor_hash']==c['source_actor_hash'];check_qualification(folder,req,q)
  r['events'].append(dict(mode='qualify',role='zero_probe',folder=str(folder),result_sha256=rt.sha(folder/'result.json')))
  r.update(zero_assistance_qualified=q['qualified'],deepest_qualified='uniform000' if q['qualified'] else None,
     stop_reason='ZERO_ASSISTANCE_REACHED' if q['qualified'] else 'ZERO_PROBE_FAILED',status='COMPLETED',final_frozen_sha256=rt.verify())
 except BaseException as error:
  r.update(status='ERROR_STOPPED',error=repr(error),traceback=traceback.format_exc());print(r['traceback'],flush=True)
 finally:
  r['utc_end']=rt.now()
  if not r['zero_assistance_qualified']:
   rt.write(run/'parent_repair_request.json',dict(status='PARENT_ACTION_REQUIRED',cause=r.get('error',r.get('stop_reason')),
      goal_complete=False,autonomous_repair_authorized=True,next_action='Use the one zero-assistance diagnostic, preserve failed evidence and continue concrete repair; never repeat this actor/profile/level.'))
  rt.write(run/'result.json',r);rt.write(run/'progress.json',r)
 return int(r['status']!='COMPLETED')
if __name__=='__main__':raise SystemExit(main())

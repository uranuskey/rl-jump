"""Evaluate first, learn only when needed, and withdraw immediately on qualification."""
import argparse
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import clearance_runtime as rt
from clearance_contract import LEVELS,BUDGET,decision,chunk_size
from clearance_audit import check_qualification


def worker(run,request,result):
    name=f"level_{request['level_index']:02d}_u{request['global_updates']:04d}_{request['mode']}"
    folder=run/name; assert not folder.exists(); folder.mkdir()
    rt.write(folder/'request.json',request)
    result.update(stage=name,active_worker_folder=str(folder)); rt.write(run/'progress.json',result)
    started=rt.now()
    with (folder/'stdout.log').open('w',encoding='utf-8') as stdout,(folder/'stderr.log').open('w',encoding='utf-8') as stderr:
        child=subprocess.Popen([sys.executable,'-X','faulthandler','-u',str(rt.HERE/'clearance_worker.py'),
            '--request',str(folder/'request.json')],stdout=stdout,stderr=stderr,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        result['active_worker_pid']=child.pid; rt.write(run/'progress.json',result)
        code=child.wait()
    rt.write(folder/'exit_receipt.json',dict(process_exited=True,exit_code=code,pid=child.pid,
        utc_start=started,utc_end=rt.now()))
    assert code==0,('Curriculum worker failed',name,code)
    row=rt.checked(folder); assert row['request']==request
    return folder,row


def execute(run,result):
    checkpoint=result['contract']['source_checkpoint']; global_updates=actor_steps=0
    result.update(events=[],qualified_levels=[],deepest_qualified=None,completed_updates=0,actor_steps=0)
    for index,(name,before,after) in enumerate(LEVELS):
        local_updates=0; tested=set()
        while True:
            assert not (rt.HERE/'STOP').exists(),'Adaptive withdrawal STOP requested'
            req=dict(mode='qualify',contract=result['contract'],frozen_sha256=result['frozen_sha256'],
                level_index=index,before=before,after=after,checkpoint=checkpoint,global_updates=global_updates,
                reuse_preflight=False)
            folder,q=worker(run,req,result)
            assert q['actor_hash'] not in tested,'Identical policy cannot be requalified until it passes'
            tested.add(q['actor_hash'])
            check_qualification(folder,req,q)
            action=decision(q['qualification'],global_updates)
            result['events'].append(dict(folder=str(folder),mode='qualify',decision=action,result_sha256=rt.sha(folder/'result.json')))
            rt.write(run/'progress.json',result)
            if action=='ADVANCE':
                chosen=dict(name=name,before=before,after=after,checkpoint=checkpoint,
                    at_global_update=global_updates,level_updates=local_updates,qualification_folder=str(folder),
                    qualification_result_sha256=rt.sha(folder/'result.json'))
                result['qualified_levels'].append(chosen); result['deepest_qualified']=name
                rt.write(run/'progress.json',result)
                print(dict(event='qualified_and_advance',**chosen),flush=True)
                break
            if action!='LEARN':
                result['stop_reason']=action; return
            amount=chunk_size(global_updates,local_updates)
            req={**req,'mode':'train','reuse_preflight':False,'updates':amount,'resume_optimizer':local_updates>0}
            folder,t=worker(run,req,result)
            assert t['actor_hash_before']==q['actor_hash'] and t['actor_hash_after']!=q['actor_hash']
            assert t['actor_hash_after'] not in tested,'Training returned an already failed deterministic actor'
            assert t['completed_updates']==amount
            global_updates+=amount; local_updates+=amount; actor_steps+=t['actor_steps']
            checkpoint=t['checkpoint']
            result['events'].append(dict(folder=str(folder),mode='train',result_sha256=rt.sha(folder/'result.json')))
            result.update(completed_updates=global_updates,actor_steps=actor_steps,latest_checkpoint=checkpoint)
            rt.write(run/'progress.json',result)
    result['stop_reason']='PLANNED_FLOOR_REACHED'


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
    assert re.fullmatch('[A-Za-z0-9_-]+',a.run_id)
    run=rt.HERE/'runs'/a.run_id; assert not run.exists();run.mkdir(parents=True)
    f,c=rt.verify(),rt.contract()
    result=dict(status='RUNNING',utc_start=rt.now(),frozen_sha256=f,contract=c,
        shared_update_budget=BUDGET,prior_updates=2,num_envs=512,early_graduation=True,
        fixed_takeoff_network=True,reward='unchanged 13-term landing score',
        zero_assistance_qualified=False,hardware_qualified=False)
    rt.write(run/'progress.json',result)
    try:
        execute(run,result)
        result.update(status='COMPLETED',final_frozen_sha256=rt.verify())
        assert result['final_frozen_sha256']==f
    except BaseException as error:
        result.update(status='ERROR_STOPPED',error=repr(error),traceback=traceback.format_exc())
        print(result['traceback'],flush=True)
    finally:
        result['utc_end']=rt.now();rt.write(run/'result.json',result);rt.write(run/'progress.json',result)
    return int(result['status']!='COMPLETED')


if __name__=='__main__':raise SystemExit(main())

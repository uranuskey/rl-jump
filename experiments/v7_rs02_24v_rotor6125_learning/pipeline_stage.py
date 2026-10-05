"""Real process exits at preflight, smoke, train and independent evaluation."""
import argparse
import re
import time
import traceback
import transfer_runtime as rt
from transfer_engine import preflight, learn, evaluate
from transfer_contract import BEFORE, AFTER, VARIANT


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-id',required=True)
    p.add_argument('--stage',choices=['preflight','smoke','train','evaluate'],required=True)
    args=p.parse_args()
    assert re.fullmatch('[A-Za-z0-9_-]+',args.run_id)
    f,c=rt.verify(),rt.contract()
    root=rt.HERE/'runs'; train=root/args.run_id
    suffix={'preflight':'_preflight','smoke':'_smoke','train':'','evaluate':'_eval'}[args.stage]
    out=root/(args.run_id+suffix)
    assert not out.exists(), 'Use a fresh run id'
    prerequisites={}
    if args.stage in ('smoke','train','evaluate'):
        q=root/(args.run_id+'_preflight'); rt.checked(q,'PREFLIGHT_COMPLETED')
        prerequisites['preflight']=dict(path=str(q/'result.json'),sha256=rt.sha(q/'result.json'))
    if args.stage in ('train','evaluate'):
        q=root/(args.run_id+'_smoke'); s=rt.checked(q,'SMOKE_COMPLETED')
        assert s['completed_updates']==2 and s['actor_steps']>0
        prerequisites['smoke']=dict(path=str(q/'result.json'),sha256=rt.sha(q/'result.json'))
    trained=None
    if args.stage=='evaluate':
        trained=rt.checked(train,'TRAIN_COMPLETED')
        assert trained['completed_updates']==128 and trained['actor_steps']>0
        prerequisites['train']=dict(path=str(train/'result.json'),sha256=rt.sha(train/'result.json'))
    with rt.exclusive(512) as resources:
        out.mkdir(parents=True)
        start=time.monotonic()
        result=dict(status='RUNNING',stage=args.stage,utc_start=rt.now(),frozen_sha256=f,
            contract=c,resources=resources,prerequisites=prerequisites,
            num_envs=45 if args.stage=='smoke' else 512,update_budget=2 if args.stage=='smoke' else 128 if args.stage=='train' else 0,
            before=BEFORE,after=AFTER,constraint_variant=VARIANT,voltage_v=24,
            estimated_motor_curve=True,physics_hz=400,reward='unchanged 13-term landing score',
            selection='latest strict-passing checkpoint; no impact-improvement ranking',
            zero_assistance_qualified=False,hardware_qualified=False,
            fixed_takeoff_weights=True,landing_actor_active_only_after_apex=True,
            no_automatic_next_assistance_level=True)
        rt.write(out/'progress.json',result)
        try:
            limit=rt.limit_for(start)
            if args.stage=='preflight': preflight(out,result,limit)
            elif args.stage in ('smoke','train'):
                learn(out,result,limit,result['update_budget'],result['num_envs'])
            else: evaluate(out,result,limit,trained)
            result['final_frozen_sha256']=rt.verify()
            assert result['final_frozen_sha256']==f
        except BaseException as error:
            result.update(status='ERROR_STOPPED',error=repr(error),traceback=traceback.format_exc())
            print(result['traceback'],flush=True)
        finally:
            result.update(utc_end=rt.now(),wall_s=time.monotonic()-start)
            rt.write(out/'result.json',result); rt.write(out/'progress.json',result)
    return int(result['status']=='ERROR_STOPPED')


if __name__=='__main__':
    raise SystemExit(main())

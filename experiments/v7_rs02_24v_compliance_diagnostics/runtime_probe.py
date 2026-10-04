"""Bounded 512-world native-crash diagnosis using unchanged v3 physics/control."""
import argparse
import faulthandler
from pathlib import Path
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'v7_rs02_24v_compliant_landing_v3'
sys.path.insert(0,str(SOURCE))
import compliant_paths
from compliant_runtime import verify,sha,write,read,exclusive,resource_limit,admission,now


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-id',required=True)
    p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=3,choices=(1,3))
    args=p.parse_args()
    assert __import__('re').fullmatch('[A-Za-z0-9_-]+',args.run_id)
    out=HERE/'runs'/args.run_id
    assert not out.exists()
    out.mkdir(parents=True)
    faulthandler.enable(all_threads=True)
    frozen=verify()
    validation=read(args.validation)
    assert validation['status']=='CONTROLLER_ADMITTED' and validation['frozen_sha256']==frozen
    result=dict(status='STARTING',frozen_sha256=frozen,script_sha256=sha(Path(__file__)),
                num_envs=512,repeats=args.repeats,utc_start=now(),evaluations=[])
    write(out/'progress.json',result)
    started=time.monotonic()
    try:
        with exclusive(512) as resource:
            result['resource']=resource
            import torch
            from compliant_env import CompliantEnv
            from compliant_learning import Policy,FrozenLaunch
            from standing_policy import JumpPolicy
            from compliant_rollout import trial,compact
            torch.set_num_threads(1)
            torch.manual_seed(104052)
            torch.cuda.manual_seed_all(104052)
            env=CompliantEnv(512,abort_dir=out/'aborts')
            standing=JumpPolicy(env.device).eval().requires_grad_(False)
            launch,policy=FrozenLaunch(env.device),Policy(env.device)
            seed=validation['seed']
            assert sha(seed['path'])==seed['sha256']
            policy.load_state_dict(torch.load(seed['path'],map_location='cpu',weights_only=True)['model_state_dict'],strict=True)
            check=resource_limit(started,1800)
            for repeat in range(args.repeats):
                count=0
                def limit():
                    nonlocal count
                    check()
                    if count%25==0:
                        result.update(status='ROLLOUT',repeat=repeat,control_step=count,utc_progress=now())
                        write(out/'progress.json',result)
                        print(dict(repeat=repeat,control_step=count),flush=True)
                    count+=1
                *_,summary=trial(env,standing,launch,policy,limit)
                write(out/f'trial_{repeat:02d}.json',summary)
                result['evaluations'].append(compact(summary))
                assert admission(summary,validation['baseline']), '512-world admission failed'
            result['status']='PROBE_PASSED'
            assert verify()==frozen
    except BaseException as error:
        result.update(status='ERROR_STOPPED',error=repr(error),traceback=traceback.format_exc())
        print(result['traceback'],flush=True)
    result.update(utc_end=now(),wall_s=time.monotonic()-started)
    write(out/'result.json',result)
    write(out/'progress.json',result)
    return int(result['status']!='PROBE_PASSED')


if __name__=='__main__':
    raise SystemExit(main())

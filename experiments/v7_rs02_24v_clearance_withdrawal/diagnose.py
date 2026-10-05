"""Bounded 512-world replay for collision diagnosis, never a qualification retry."""
import argparse,time,traceback
from pathlib import Path
import diagnostic_runtime as rt

def execute(out,result):
    import torch,numpy as np,mujoco
    from contact_env import ContactEnv
    from slot_learning import Policy,FrozenLaunch
    from standing_policy import JumpPolicy
    from environment import reset_cases
    from slot_control import kinematics,target
    from compliant_runtime import resource_limit
    import warp as wp
    torch.set_num_threads(1);torch.manual_seed(105070);torch.cuda.manual_seed_all(105070)
    contract,ck=rt.source();result['source_checkpoint']=ck
    state=torch.load(ck['path'],map_location='cpu',weights_only=True)
    env=ContactEnv(512,[contract['profile']],before=.525,after=.525,variant='rotor5ms',
                   fast_backend=True,proof=True,abort_dir=out/'aborts')
    standing=JumpPolicy(env.device).eval().requires_grad_(False)
    launch=FrozenLaunch(env.device)
    policy=Policy(env.device).eval().requires_grad_(False)
    policy.load_state_dict(state['model_state_dict'],strict=True)
    limit=resource_limit(time.monotonic(),1800)
    result['replays']=[]
    # Fixed upper bound; stop once useful failed-contact evidence is obtained.
    for repeat in range(1,4):
        env.assist_strength.fill_(.525);reset_cases(env)
        for step in range(80):
            limit();assert not (rt.HERE/'STOP').exists()
            with torch.no_grad():
                if step==30:
                    obs=env.plan_observation().clone()
                    env.lock_launch(launch(obs));env.lock_parameters(policy.distribution(obs).mean)
                teacher=standing.actor(env.obs) if step<30 else torch.zeros(env.n,6,device=env.device)
                env.step(teacher,auto_reset=False)
        env.assert_launch_fixed();env.assert_parameters_fixed()
        seen=env.contact_seen.cpu().numpy()
        counts=wp.to_torch(env.contact_count).cpu().numpy()
        assert np.all(counts[seen]>0) and np.all(counts<=16),'Incomplete exact GPU contact capture'
        result['replays'].append(dict(repeat=repeat,worlds=512,until_s=1.6,self_contact_worlds=int(seen.sum())))
        rt.write(out/'progress.json',result)
        if not seen.any():continue
        worlds=np.flatnonzero(seen);q=env.contact_q.cpu().numpy();v=env.contact_v.cpu().numpy()
        pairs=wp.to_torch(env.contact_pairs).cpu().numpy()
        values=wp.to_torch(env.contact_values).cpu().numpy()
        np.savez_compressed(out/'failed_states.npz',worlds=worlds,q=q[worlds],v=v[worlds],
            pairs=pairs[worlds],values=values[worlds],counts=counts[worlds])
        s=kinematics(env.contact_q[:,env.qids[:4]],env.contact_v[:,env.main[:4]])
        tx,sl=target(s['height'])
        s={k:x.cpu().numpy() for k,x in s.items()};tx=tx.cpu().numpy()
        terminal={k:x.cpu().numpy() for k,x in env.terminal.items()}
        data=mujoco.MjData(env.m);records=[]
        for w in worlds:
            gpu=[]
            for j in range(counts[w]):
                ga,gb=map(int,pairs[w,j])
                gpu.append(dict(geom1=env.m.geom(ga).name,geom2=env.m.geom(gb).name,
                    body1=env.m.body(int(env.m.geom_bodyid[ga])).name,body2=env.m.body(int(env.m.geom_bodyid[gb])).name,
                    penetration_mm=float(-1000*values[w,j,0]),normal_force_n=float(values[w,j,1])))
            data.qpos[:]=q[w];data.qvel[:]=v[w];mujoco.mj_forward(env.m,data)
            cpu=[]
            for c in data.contact:
                ga,gb=int(c.geom1),int(c.geom2)
                if env.m.geom_bodyid[ga] and env.m.geom_bodyid[gb] and c.dist<.001:
                    cpu.append(dict(geom1=env.m.geom(ga).name,geom2=env.m.geom(gb).name,
                                    signed_distance_mm=float(1000*c.dist)))
            records.append(dict(world=int(w),case=int(w%45),time_s=float(terminal['ticks'][w]*.0025),
                gpu_contacts=gpu,cpu_static_contacts=cpu,height_mm=(1000*s['height'][w]).tolist(),
                cad_dx_mm=(-1000*s['x'][w]).tolist(),target_cad_dx_mm=(-1000*tx[w]).tolist(),
                vx_mps=s['vx'][w].tolist(),vh_mps=s['vh'][w].tolist(),
                qpos=q[w].tolist(),qvel=v[w].tolist()))
        result.update(records=records,physics_receipt=env.physics_receipt,
            failed_states_sha256=rt.sha(out/'failed_states.npz'),reproduced=True)
        break
    result.setdefault('reproduced',False)
    result.update(status='DIAGNOSED',qualification=False,training_updates=0,
                  original_guards_unchanged=True,no_state_projection=True)
    assert result['reproduced'],'No failed contacts within bounded diagnostic replays'

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);args=p.parse_args()
    assert args.run_id.replace('_','').isalnum()
    out=rt.HERE/'runs'/args.run_id;assert not out.exists();out.mkdir(parents=True)
    result=dict(status='RUNNING',frozen_sha256=rt.verify(),utc_start=rt.now(),mode='diagnosis_only')
    rt.write(out/'progress.json',result);code=0
    with rt.exclusive(512) as resources:
        result['resources']=resources
        try:execute(out,result)
        except BaseException as e:
            result.update(status='ERROR_STOPPED',error=repr(e),traceback=traceback.format_exc())
            print(result['traceback'],flush=True);code=1
        finally:
            result.update(utc_end=rt.now(),final_frozen_sha256=rt.verify())
            rt.write(out/'result.json',result);rt.write(out/'progress.json',result)
    return code
if __name__=='__main__':raise SystemExit(main())

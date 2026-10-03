"""Continue the stopped 2048-world job at 512 worlds, with 128 total updates."""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import re
import shutil
import time
import traceback
import bootstrap
from bootstrap import HERE
from runtime import exclusive, write, sha
from resume_state import read, restore, source_state, verify_resume


def execute(args, out, result, source, limit):
    import torch
    from fast_env import FastLandingEnv
    from landing import LandingEnv
    from fast_rollout import trial
    from rollout import prefix_check, compact, qualifies
    from standing_policy import JumpPolicy
    from learning import FrozenLaunch, Policy, PPO
    from environment import reset_cases
    from native_checks import readback, selective_reset
    torch.set_num_threads(1)
    torch.manual_seed(104027)
    torch.cuda.manual_seed_all(104027)
    preflight = out/'preflight'
    preflight.mkdir()
    env = FastLandingEnv(45, abort_dir=preflight/'aborts')
    standing = JumpPolicy(env.device).eval().requires_grad_(False)
    launch = FrozenLaunch(env.device)
    reset_cases(env)
    with torch.no_grad():
        env.step(standing.actor(env.obs), auto_reset=False)
    original_sensors, fast_sensors = LandingEnv.sensors(env), env.sensors()
    assert all(torch.equal(original_sensors[k], fast_sensors[k]) for k in original_sensors)
    result['readback'] = readback(env, world=1)
    result['selective_reset'] = selective_reset(env)
    result['prefix_proof'] = prefix_check(env, standing, launch, limit, preflight)
    write(preflight/'prefix_proof.json', result['prefix_proof'])
    result['same_state_sensor_equivalence'] = True
    print(json.dumps(dict(event='fast_native_preflight', status='PASS', worlds=45)), flush=True)
    del original_sensors, fast_sensors, env
    gc.collect()
    torch.cuda.empty_cache()
    env = FastLandingEnv(512, abort_dir=out/'aborts')
    launch_state = {k: v.clone() for k, v in launch.state_dict().items()}
    policy = Policy(env.device)
    ppo = PPO(policy)

    def save(name, update):
        path = out/name
        temporary = path.with_suffix('.pt.tmp')
        torch.save(dict(model_state_dict=policy.state_dict(), actor_optimizer=ppo.actor_optimizer.state_dict(),
            critic_optimizer=ppo.critic_optimizer.state_dict(), update=update, voltage_v=24, assist_strength=.625,
            frozen_sha256=result['frozen_sha256'], resume_frozen_sha256=result['resume_frozen_sha256'],
            num_envs=512, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), temporary)
        temporary.replace(path)
        return dict(path=str(path), sha256=sha(path), update=update)

    # Preserve the actual original zero-action model and inherited history.
    shutil.copy2(args.source/'initial.pt', out/'initial.pt')
    initial = dict(path=str(out/'initial.pt'), sha256=sha(out/'initial.pt'), update=0)
    state = torch.load(out/'initial.pt', map_location='cpu', weights_only=True)
    assert state['update']==0 and state['frozen_sha256']==result['frozen_sha256']
    policy.load_state_dict(state['model_state_dict'])
    for i in range(1, args.expected_update+1):
        shutil.copy2(args.source/f'update_{i:04d}.json', out/f'update_{i:04d}.json')
    _, baseline = trial(env, standing, launch, policy, limit)
    write(out/'baseline.json', baseline)
    assert qualifies(baseline), '512-world zero-action baseline failed'
    result.update(baseline=compact(baseline), selected=dict(checkpoint=initial, evaluation=compact(baseline)),
                  historical_selected=source['selected'], evaluations=list(source['evaluations']),
                  frozen_launch_unchanged=True)
    candidate = source['selected']['checkpoint']
    if candidate['update']>0:
        policy.load_state_dict(torch.load(candidate['path'], map_location='cpu', weights_only=True)['model_state_dict'])
        _, evaluation = trial(env, standing, launch, policy, limit)
        entry = dict(update=candidate['update'], checkpoint=candidate, evaluation=compact(evaluation),
                     qualifies=qualifies(evaluation), scope='historical best re-evaluated at 512 worlds')
        write(out/'historical_best_512.json', dict(**entry, cases=evaluation['cases']))
        result['evaluations'].append(entry)
        if entry['qualifies'] and evaluation['mean_return']>baseline['mean_return']:
            result['selected'] = entry
    result['restore'] = restore(policy, ppo, args.checkpoint, args.expected_update, result['frozen_sha256'])
    print(json.dumps(dict(event='resume_ready', **result['restore'], baseline=compact(baseline))), flush=True)
    write(out/'progress.json', {**result, 'status':'RUNNING'})
    for update in range(args.expected_update+1, 129):
        started = time.monotonic()
        batch, summary = trial(env, standing, launch, policy, limit, stochastic=True, collect=True)
        rollout_s = time.monotonic()-started
        if batch is None:
            raise RuntimeError('No post-apex training transitions')
        ppo_started = time.monotonic()
        stats = ppo.update(batch)
        del batch
        ppo_s = time.monotonic()-ppo_started
        assert all(torch.equal(v, launch_state[k]) for k, v in launch.state_dict().items())
        result['completed_updates'] = update
        result['actor_steps'] += stats['actor_steps']
        result['landing_transitions'] += stats['eligible_samples']
        row = dict(update=update, num_envs=512, wall_s=time.monotonic()-started,
                   rollout_s=rollout_s, ppo_s=ppo_s, **stats, trial=compact(summary))
        # A completed update is saved before periodic evaluation or its STOP check.
        result['latest_checkpoint'] = save('latest.pt', update)
        write(out/f'trials_{update:04d}.json', summary)
        write(out/f'update_{update:04d}.json', row)
        write(out/'progress.json', {**result, 'status':'RUNNING', 'last_update':row})
        print(json.dumps(row), flush=True)
        if update%8==0 or update==128:
            evaluation_started = time.monotonic()
            checkpoint = save(f'model_{update:04d}.pt', update)
            _, evaluation = trial(env, standing, launch, policy, limit)
            entry = dict(update=update, checkpoint=checkpoint, evaluation=compact(evaluation),
                         qualifies=qualifies(evaluation), wall_s=time.monotonic()-evaluation_started)
            write(out/f'evaluation_{update:04d}.json', dict(**entry, cases=evaluation['cases']))
            result['evaluations'].append(entry)
            if entry['qualifies'] and evaluation['mean_return']>result['selected']['evaluation']['mean_return']:
                result['selected'] = entry
            print(json.dumps(dict(event='evaluation', **entry)), flush=True)
        final_frozen, final_resume = verify_resume()
        write(out/'progress.json', {**result, 'status':'RUNNING', 'last_update':row,
             'final_frozen_sha256':final_frozen, 'final_resume_frozen_sha256':final_resume})
    result.update(final_checkpoint=save('final.pt', 128), status='BUDGET_COMPLETED',
                  final_frozen_sha256=final_frozen, final_resume_frozen_sha256=final_resume,
                  final_training_assist=.625)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--expected-update', type=int, required=True)
    p.add_argument('--run-id', required=True)
    args = p.parse_args()
    assert re.fullmatch('[A-Za-z0-9_-]+', args.run_id)
    frozen, resume_frozen = verify_resume()
    source = source_state(args.source, args.checkpoint, args.expected_update, frozen)
    out = HERE/'runs'/args.run_id
    if out.exists():
        raise RuntimeError('Use a fresh run id')
    with exclusive(512) as resource:
        out.mkdir(parents=True)
        started = time.monotonic()
        result = dict(status='STARTING', mode='train', num_envs=512, resource=resource,
            frozen_sha256=frozen, resume_frozen_sha256=resume_frozen,
            utc_start=datetime.now(timezone.utc).isoformat(), voltage_v=24, assist_strength=.625,
            control_hz=50, action_dim=7, hardware_used=False, completed_updates=args.expected_update,
            actor_steps=source['actor_steps'], landing_transitions=source['landing_transitions'],
            resumed_from=dict(run=str(args.source), result_sha256=sha(args.source/'result.json'),
                checkpoint=dict(path=str(args.checkpoint), sha256=sha(args.checkpoint), update=args.expected_update)),
            environment_schedule=[dict(first_update=1, last_update=args.expected_update, num_envs=2048),
                                  dict(first_update=args.expected_update+1, last_update=128, num_envs=512)],
            bitwise_trajectory_continuation=False,
            trajectory_note='Model, both Adam optimizers and RNG restored; worlds reset and batch size changes.')
        write(out/'progress.json', result)
        def limit():
            import torch
            if (HERE/'STOP').exists() or (HERE.parent/'v7_jump_in_place/STOP').exists():
                raise RuntimeError('STOP requested')
            if time.monotonic()-started>43200 or torch.cuda.mem_get_info()[0]<512*1024**2:
                raise RuntimeError('Training resource bound')
        try:
            execute(args, out, result, source, limit)
        except BaseException as error:
            result.update(status='ERROR_STOPPED', error=repr(error), traceback=traceback.format_exc())
            print(result['traceback'], flush=True)
        finally:
            result.update(wall_s=time.monotonic()-started, utc_end=datetime.now(timezone.utc).isoformat())
            write(out/'result.json', result)
            write(out/'progress.json', result)
            print(json.dumps({k:result.get(k) for k in ('status','completed_updates','actor_steps','wall_s')}), flush=True)
    return int(result['status']=='ERROR_STOPPED')


if __name__=='__main__':
    raise SystemExit(main())

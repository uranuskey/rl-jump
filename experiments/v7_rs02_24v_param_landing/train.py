"""Bounded method-1 PPO: 512 episode plans per update, 128 updates total."""
import argparse
import json
from pathlib import Path
import re
import time
import traceback
import path_setup
from path_setup import HERE
from param_runtime import exclusive, now, read, resource_limit, sha, verify, write


def execute(args, out, result, limit):
    import torch
    from param_env import ParameterEnv
    from param_learning import Policy, PPO, FrozenLaunch
    from param_rollout import trial, compact, qualifies
    from param_prefix import prefix_check
    from standing_policy import JumpPolicy
    from environment import reset_cases
    from native_checks import readback, selective_reset
    torch.set_num_threads(1)
    torch.manual_seed(104031)
    torch.cuda.manual_seed_all(104031)
    env = ParameterEnv(args.num_envs, abort_dir=out/'aborts')
    standing = JumpPolicy(env.device).eval().requires_grad_(False)
    launch = FrozenLaunch(env.device)
    launch_state = {k:v.clone() for k,v in launch.state_dict().items()}
    reset_cases(env)
    with torch.no_grad():
        env.step(standing.actor(env.obs), auto_reset=False)
    result['readback'] = readback(env, world=1)
    result['selective_reset'] = selective_reset(env)
    if args.mode=='smoke':
        result['prefix_proof'] = prefix_check(env, standing, launch, limit, out)
        write(out/'prefix_proof.json', result['prefix_proof'])
        print(json.dumps(dict(event='prefix_proof', status=result['prefix_proof']['status'])), flush=True)
    else:
        result['prefix_proof'] = read(args.smoke_result)['prefix_proof']
        result['smoke_result_sha256'] = sha(args.smoke_result)
    policy = Policy(env.device)
    ppo = PPO(policy)

    def save(name, update):
        path = out/name
        temporary = path.with_suffix('.pt.tmp')
        torch.save(dict(task='fixed_jump_13d_parameter_landing_v1', model_state_dict=policy.state_dict(),
            actor_optimizer=ppo.actor_optimizer.state_dict(), critic_optimizer=ppo.critic_optimizer.state_dict(),
            update=update, num_envs=args.num_envs, voltage_v=24, assist_strength=.625,
            frozen_sha256=result['frozen_sha256'], torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all()), temporary)
        temporary.replace(path)
        return dict(path=str(path), sha256=sha(path), update=update)

    initial = save('initial.pt', 0)
    *_, baseline = trial(env, standing, launch, policy, limit)
    write(out/'baseline.json', baseline)
    assert qualifies(baseline), 'Untrained parameter controller failed the validated landing baseline'
    result.update(completed_updates=0, actor_steps=0, executed_landing_plans=0,
        selected=dict(checkpoint=initial, evaluation=compact(baseline)), baseline=compact(baseline),
        evaluations=[], frozen_launch_unchanged=True)
    print(json.dumps(dict(event='baseline', **compact(baseline))), flush=True)
    write(out/'progress.json', {**result, 'status':'RUNNING'})
    budget = 2 if args.mode=='smoke' else 128
    for update in range(1, budget+1):
        started = time.monotonic()
        obs, action, reward, eligible, summary = trial(env, standing, launch, policy, limit, stochastic=True)
        rollout_s = time.monotonic()-started
        if not bool(eligible.any()):
            raise RuntimeError('No episode reached the landing parameter activation gate')
        ppo_started = time.monotonic()
        stats = ppo.update(obs, action, reward, eligible)
        ppo_s = time.monotonic()-ppo_started
        assert all(torch.equal(value, launch_state[key]) for key, value in launch.state_dict().items())
        result['completed_updates'] = update
        result['actor_steps'] += stats['actor_steps']
        result['executed_landing_plans'] += stats['eligible_samples']
        row = dict(update=update, num_envs=args.num_envs, wall_s=time.monotonic()-started,
            rollout_s=rollout_s, ppo_s=ppo_s, **stats, trial=compact(summary))
        result['latest_checkpoint'] = save('latest.pt', update)
        write(out/f'trials_{update:04d}.json', summary)
        write(out/f'update_{update:04d}.json', row)
        write(out/'progress.json', {**result, 'status':'RUNNING', 'last_update':row})
        print(json.dumps(row), flush=True)
        if update%8==0 or update==budget:
            checkpoint = save(f'model_{update:04d}.pt', update)
            *_, evaluation = trial(env, standing, launch, policy, limit)
            entry = dict(update=update, checkpoint=checkpoint, evaluation=compact(evaluation),
                         qualifies=qualifies(evaluation), zero_pass_warning=evaluation['passed']==0)
            write(out/f'evaluation_{update:04d}.json', dict(**entry, cases=evaluation['cases']))
            result['evaluations'].append(entry)
            result['latest_evaluation'] = entry
            if entry['qualifies'] and evaluation['mean_return']>result['selected']['evaluation']['mean_return']:
                result['selected'] = entry
            print(json.dumps(dict(event='evaluation', **entry)), flush=True)
        write(out/'progress.json', {**result, 'status':'RUNNING', 'last_update':row, 'final_frozen_sha256':verify()})
    result.update(final_checkpoint=save('final.pt', budget), final_frozen_sha256=verify(),
                  status='SMOKE_COMPLETED' if args.mode=='smoke' else 'BUDGET_COMPLETED')
    assert result['actor_steps']>0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('smoke','train'), required=True)
    p.add_argument('--num-envs', type=int, choices=(45,512), required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--smoke-result', type=Path)
    args = p.parse_args()
    assert re.fullmatch('[A-Za-z0-9_-]+', args.run_id)
    assert args.num_envs==(45 if args.mode=='smoke' else 512)
    frozen = verify()
    if args.mode=='train':
        smoke = read(args.smoke_result)
        receipt = read(args.smoke_result.parent/'exit_receipt.json')
        assert smoke['status']=='SMOKE_COMPLETED' and smoke['prefix_proof']['status']=='PASS'
        assert smoke['frozen_sha256']==smoke['final_frozen_sha256']==receipt['frozen_sha256']==frozen
        assert receipt['exit_code']==0 and receipt['process_exited'] and receipt['stage']=='smoke'
    out = HERE/'runs'/args.run_id
    if out.exists():
        raise RuntimeError('Use a fresh run id')
    with exclusive(args.num_envs) as resource:
        out.mkdir(parents=True)
        started = time.monotonic()
        result = dict(status='STARTING', mode=args.mode, method='episodic_13d_landing_parameters',
            num_envs=args.num_envs, resource=resource, frozen_sha256=frozen, utc_start=now(),
            voltage_v=24, assist_strength=.625, action_dim=13, observation_dim=17,
            parameter_samples_per_episode=1, initialization='validated landing curve; new actor and critic',
            control_hz=50, physics_hz=400, hardware_used=False)
        write(out/'progress.json', result)
        try:
            execute(args, out, result, resource_limit(started, 1800 if args.mode=='smoke' else 43200))
        except BaseException as error:
            result.update(status='ERROR_STOPPED', error=repr(error), traceback=traceback.format_exc())
            print(result['traceback'], flush=True)
        finally:
            result.update(wall_s=time.monotonic()-started, utc_end=now())
            write(out/'result.json', result)
            write(out/'progress.json', result)
            print(json.dumps({k:result.get(k) for k in ('status','completed_updates','actor_steps','wall_s')}), flush=True)
    return int(result['status']=='ERROR_STOPPED')


if __name__=='__main__':
    raise SystemExit(main())

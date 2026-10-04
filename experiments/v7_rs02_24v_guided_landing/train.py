"""Guided warm continuation: physical proposals then 128 new PPO updates."""
import argparse
import json
from pathlib import Path
import re
import time
import traceback
import guided_paths
from guided_paths import HERE
from guided_runtime import exclusive, now, read, resource_limit, sha, verify, write


def execute(args, out, result, limit):
    import torch
    from guided_env import GuidedEnv as ParameterEnv
    from guided_learning import Policy, PPO, FrozenLaunch, load_parent, set_exploration, apply_offset
    from guided_rollout import trial, compact, qualifies, better, force
    from guided_search import search
    from param_prefix import prefix_check
    from standing_policy import JumpPolicy
    from environment import reset_cases
    from native_checks import readback, selective_reset
    torch.set_num_threads(1)
    torch.manual_seed(104042)
    torch.cuda.manual_seed_all(104042)
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
    load_parent(policy,args.checkpoint)
    ppo = PPO(policy)

    def save(name, update):
        path = out/name
        temporary = path.with_suffix('.pt.tmp')
        torch.save(dict(task='guided_parameter_cushion_v1', model_state_dict=policy.state_dict(),
            actor_optimizer=ppo.actor_optimizer.state_dict(), critic_optimizer=ppo.critic_optimizer.state_dict(),
            update=update, num_envs=args.num_envs, voltage_v=24, assist_strength=.625,
            frozen_sha256=result['frozen_sha256'], parent_checkpoint_sha256=sha(args.checkpoint), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all()), temporary)
        temporary.replace(path)
        return dict(path=str(path), sha256=sha(path), update=update)

    initial = save('parent_initial.pt', 0)
    *_, baseline = trial(env, standing, launch, policy, limit)
    write(out/'baseline.json', baseline)
    assert qualifies(baseline), 'Inherited update-120 actor failed the landing baseline'
    result.update(completed_updates=0, actor_steps=0, executed_landing_plans=0,
        selected=dict(checkpoint=initial, evaluation=compact(baseline)), baseline=compact(baseline),
        evaluations=[], frozen_launch_unchanged=True)
    print(json.dumps(dict(event='baseline', **compact(baseline))), flush=True)
    write(out/'progress.json', {**result, 'status':'RUNNING'})
    result['parent_checkpoint']=dict(path=str(args.checkpoint),sha256=sha(args.checkpoint),update=120)
    result['parent_initial_checkpoint']=initial
    if args.mode=='train':
        result['guide_search']=search(env,standing,launch,policy,limit,out)
        offset=result['guide_search']['chosen']['offset']
        parent_bias=policy.actor[-1].bias.detach().clone()
        apply_offset(policy,offset)
        *_,guided=trial(env,standing,launch,policy,limit)
        write(out/'guided_validation.json',guided)
        accepted=qualifies(guided) and force(guided)<force(baseline)-.5
        result['guide_search']['validated_accepted']=accepted
        if accepted:
            seed=save('guided_initial.pt',0)
            result['selected']=dict(checkpoint=seed,evaluation=compact(guided))
            result['guided_baseline']=compact(guided)
        else:
            with torch.no_grad():
                policy.actor[-1].bias.copy_(parent_bias)
            seed=save('guided_initial.pt',0)
            result['guided_baseline']=compact(baseline)
        result['guided_checkpoint']=seed
        write(out/'guide_search.json',result['guide_search'])
        print(json.dumps(dict(event='guidance_validated',accepted=accepted,
            proposal=result['guide_search']['chosen']['name'],baseline_force_n=force(baseline),
            guided_force_n=force(guided),passed=guided['passed'])),flush=True)
    else:
        result['guided_checkpoint']=save('guided_initial.pt',0)
        result['guided_baseline']=compact(baseline)
    write(out/'progress.json',{**result,'status':'RUNNING'})
    budget = 2 if args.mode=='smoke' else 128
    for update in range(1, budget+1):
        set_exploration(policy,update-1)
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
            if better(evaluation,result['selected']['evaluation']):
                result['selected'] = entry
            print(json.dumps(dict(event='evaluation', **entry)), flush=True)
            if entry['zero_pass_warning']:
                raise RuntimeError('Latest deterministic evaluation has zero passing worlds')
        write(out/'progress.json', {**result, 'status':'RUNNING', 'last_update':row, 'final_frozen_sha256':verify()})
    result.update(final_checkpoint=save('final.pt', budget), final_frozen_sha256=verify(),
                  status='SMOKE_COMPLETED' if args.mode=='smoke' else 'BUDGET_COMPLETED')
    assert result['actor_steps']>0
    if args.mode=='smoke':
        assert result['latest_evaluation']['qualifies'], 'Smoke latest policy must retain all 45 landings'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('smoke','train'), required=True)
    p.add_argument('--num-envs', type=int, choices=(45,512), required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--smoke-result', type=Path)
    p.add_argument('--checkpoint',type=Path,required=True)
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
        result = dict(status='STARTING', mode=args.mode, method='guided_episodic_13d_landing_parameters',
            num_envs=args.num_envs, resource=resource, frozen_sha256=frozen, utc_start=now(),
            voltage_v=24, assist_strength=.625, action_dim=13, observation_dim=17,
            parameter_samples_per_episode=1, initialization='audited update-120 actor; new critic and Adam; physical proposal search before on-policy PPO',
            control_hz=50, physics_hz=400, hardware_used=False)
        write(out/'progress.json', result)
        try:
            execute(args, out, result, resource_limit(started, 2400 if args.mode=='smoke' else 43200))
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

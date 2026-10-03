"""Fixed-jump, continuous-landing PPO: bounded smoke or 128 full episodes batches."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
import bootstrap
from bootstrap import HERE
from runtime import exclusive, verify, write, sha


def execute(args, out, result, limit):
    import torch
    from landing import LandingEnv
    from standing_policy import JumpPolicy
    from learning import FrozenLaunch, Policy, PPO
    from environment import reset_cases
    from native_checks import readback, selective_reset
    from rollout import trial, prefix_check, compact, qualifies
    torch.set_num_threads(1)
    torch.manual_seed(104027)
    torch.cuda.manual_seed_all(104027)
    env = LandingEnv(args.num_envs, abort_dir=out/'aborts')
    standing = JumpPolicy(env.device).eval().requires_grad_(False)
    launch = FrozenLaunch(env.device)
    launch_state = {k: v.clone() for k, v in launch.state_dict().items()}
    reset_cases(env)
    with torch.no_grad():
        env.step(standing.actor(env.obs), auto_reset=False)
    result['readback'] = readback(env, world=1)
    result['selective_reset'] = selective_reset(env)
    if args.mode == 'smoke':
        result['prefix_proof'] = prefix_check(env, standing, launch, limit, out)
        write(out/'prefix_proof.json', result['prefix_proof'])
        print(json.dumps(dict(event='prefix_proof', **result['prefix_proof'])), flush=True)
    else:
        result['prefix_proof'] = json.loads(args.smoke_result.read_text())['prefix_proof']
    policy = Policy(env.device)
    ppo = PPO(policy)

    def save(name, update):
        path = out/name
        torch.save(dict(model_state_dict=policy.state_dict(), actor_optimizer=ppo.actor_optimizer.state_dict(),
            critic_optimizer=ppo.critic_optimizer.state_dict(), update=update, voltage_v=24, assist_strength=.625,
            frozen_sha256=result['frozen_sha256'], torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all()), path)
        return dict(path=str(path), sha256=sha(path), update=update)

    initial = save('initial.pt', 0)
    _, baseline = trial(env, standing, launch, policy, limit)
    write(out/'baseline.json', baseline)
    assert qualifies(baseline), 'Fixed-jump zero-action baseline failed'
    result.update(completed_updates=0, actor_steps=0, landing_transitions=0,
        selected=dict(checkpoint=initial, evaluation=compact(baseline)), evaluations=[],
        baseline=compact(baseline), frozen_launch_unchanged=True)
    print(json.dumps(dict(event='baseline', **compact(baseline))), flush=True)
    write(out/'progress.json', {**result, 'status':'RUNNING'})
    budget = 2 if args.mode=='smoke' else 128
    for update in range(1, budget+1):
        started = time.monotonic()
        batch, summary = trial(env, standing, launch, policy, limit, stochastic=True, collect=True)
        if batch is None:
            raise RuntimeError('No post-apex training transitions')
        stats = ppo.update(batch)
        del batch
        assert all(torch.equal(v, launch_state[k]) for k, v in launch.state_dict().items())
        result['completed_updates'] = update
        result['actor_steps'] += stats['actor_steps']
        result['landing_transitions'] += stats['eligible_samples']
        row = dict(update=update, wall_s=time.monotonic()-started, **stats, trial=compact(summary))
        write(out/f'trials_{update:04d}.json', summary)
        write(out/f'update_{update:04d}.json', row)
        print(json.dumps(row), flush=True)
        if update%8==0 or update==budget:
            checkpoint = save(f'model_{update:04d}.pt', update)
            _, evaluation = trial(env, standing, launch, policy, limit)
            entry = dict(update=update, checkpoint=checkpoint, evaluation=compact(evaluation),
                         qualifies=qualifies(evaluation))
            write(out/f'evaluation_{update:04d}.json', dict(**entry, cases=evaluation['cases']))
            result['evaluations'].append(entry)
            if entry['qualifies'] and evaluation['mean_return']>result['selected']['evaluation']['mean_return']:
                result['selected'] = entry
            print(json.dumps(dict(event='evaluation', **entry)), flush=True)
        result['latest_checkpoint'] = save('latest.pt', update)
        write(out/'progress.json', {**result, 'status':'RUNNING', 'last_update':row,
                                    'final_frozen_sha256':verify()})
    result.update(final_checkpoint=save('final.pt', budget),
        status='SMOKE_COMPLETED' if args.mode=='smoke' else 'BUDGET_COMPLETED',
        final_frozen_sha256=verify(), final_training_assist=.625)
    assert result['actor_steps']>0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('smoke', 'train'), required=True)
    p.add_argument('--num-envs', type=int, choices=(45, 256, 2048), required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--smoke-result', type=Path)
    args = p.parse_args()
    if args.mode=='smoke' and args.num_envs!=45:
        raise ValueError('Paired native smoke uses the original 45 conditions')
    frozen = verify()
    if args.mode=='train':
        smoke = json.loads(args.smoke_result.read_text(encoding='utf-8-sig'))
        receipt = json.loads((args.smoke_result.parent/'exit_receipt.json').read_text(encoding='utf-8-sig'))
        assert smoke['status']=='SMOKE_COMPLETED' and smoke['prefix_proof']['status']=='PASS'
        assert smoke['frozen_sha256']==smoke['final_frozen_sha256']==frozen
        assert receipt['exit_code']==0 and receipt['process_exited'] and receipt['frozen_sha256']==frozen
    out = HERE/'runs'/args.run_id
    if out.exists():
        raise RuntimeError('Use a fresh run id')
    with exclusive(args.num_envs) as resource:
        out.mkdir(parents=True)
        start = time.monotonic()
        result = dict(status='STARTING', mode=args.mode, num_envs=args.num_envs, resource=resource,
            frozen_sha256=frozen, utc_start=datetime.now(timezone.utc).isoformat(), voltage_v=24,
            assist_strength=.625, control_hz=50, action_dim=7, hardware_used=False)
        def limit():
            import torch
            if (HERE/'STOP').exists() or (HERE.parent/'v7_jump_in_place/STOP').exists():
                raise RuntimeError('STOP requested')
            if time.monotonic()-start > (43200 if args.mode=='train' else 1800):
                raise RuntimeError('Wall time bound')
            if torch.cuda.mem_get_info()[0]<512*1024**2:
                raise RuntimeError('GPU reserve')
        try:
            execute(args, out, result, limit)
        except BaseException as error:
            result.update(status='ERROR_STOPPED', error=repr(error), traceback=traceback.format_exc())
            print(result['traceback'], flush=True)
        finally:
            result.update(wall_s=time.monotonic()-start, utc_end=datetime.now(timezone.utc).isoformat())
            write(out/'result.json', result)
            write(out/'progress.json', result)
            print(json.dumps({k: result.get(k) for k in ('status', 'completed_updates', 'actor_steps', 'wall_s')}), flush=True)
    return int(result['status']=='ERROR_STOPPED')


if __name__=='__main__':
    raise SystemExit(main())

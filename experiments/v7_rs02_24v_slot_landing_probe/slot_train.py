"""Fresh Adam, fixed takeoff, one landing-plan sample per complete episode."""
import argparse
import json
from pathlib import Path
import re
import time
import traceback
from slot_training_runtime import *


def execute(args, out, result, contract, limit):
    import torch
    from slot_env import SlotEnv
    from slot_learning import Policy, FrozenLaunch, exploration
    from slot_ppo import PPO
    from compliant_rollout import trial, compact
    from standing_policy import JumpPolicy
    torch.set_num_threads(1)
    torch.manual_seed(105057)
    torch.cuda.manual_seed_all(105057)
    env = SlotEnv(args.num_envs, [contract['profile']], proof=True, abort_dir=out/'aborts')
    standing = JumpPolicy(env.device).eval().requires_grad_(False)
    launch, policy = FrozenLaunch(env.device), Policy(env.device)
    source = contract['source_checkpoint']
    state = torch.load(source['path'], map_location='cpu', weights_only=True)
    assert sha(source['path']) == SOURCE_SHA and state['update'] == 56
    assert state['frozen_sha256'] == SOURCE_FROZEN and state['action_dim'] == 16
    assert state['voltage_v'] == 24 and state['assist_strength'] == .625
    policy.load_state_dict(state['model_state_dict'], strict=True)
    exploration(policy, 0)
    ppo = PPO(policy)
    launch_state = {k:v.clone() for k,v in launch.state_dict().items()}

    def save(name, update):
        path = out/name
        temporary = path.with_suffix('.pt.tmp')
        torch.save(dict(task='slot_tracking_compliant_landing', model_state_dict=policy.state_dict(),
            actor_optimizer=ppo.actor_optimizer.state_dict(), critic_optimizer=ppo.critic_optimizer.state_dict(),
            update=update, num_envs=args.num_envs, voltage_v=24, assist_strength=.625, action_dim=16,
            frozen_sha256=SOURCE_FROZEN, training_frozen_sha256=result['training_frozen_sha256'],
            source_checkpoint_sha256=SOURCE_SHA, profile=contract['profile'], air_shift_in_actor_m=-.005,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), temporary)
        temporary.replace(path)
        return dict(path=str(path), sha256=sha(path), update=update)

    initial = save('initial.pt', 0)
    *_, baseline = trial(env, standing, launch, policy, limit)
    initial_metrics = metrics(baseline)
    write(out/'baseline.json', baseline)
    assert env.proof_samples > 0
    assert seed_qualified(initial_metrics, contract['baseline']), 'Seed failed full batch admission'
    result.update(completed_updates=0, actor_steps=0, executed_landing_plans=0,
        initial_checkpoint=initial, source_checkpoint=source, baseline=compact(baseline),
        initial_metrics=initial_metrics, selected=dict(checkpoint=initial, metrics=initial_metrics,
            evaluation=compact(baseline), update=0), evaluations=[], prefix_proof_samples=env.proof_samples,
        frozen_launch_unchanged=True)
    write(out/'progress.json', result)
    print(json.dumps(dict(event='deterministic_seed', metrics=initial_metrics)), flush=True)
    if args.mode == 'validate':
        result['status'] = 'BATCH_VALIDATED'
        return
    budget = 2 if args.mode == 'smoke' else 128
    for update in range(1, budget+1):
        exploration(policy, update-1)
        started = time.monotonic()
        obs, action, reward, eligible_mask, summary = trial(env, standing, launch, policy, limit, stochastic=True)
        rollout_s = time.monotonic()-started
        assert bool(eligible_mask.any()), 'No executed landing plans'
        ppo_started = time.monotonic()
        stats = ppo.update(obs, action, reward, eligible_mask)
        assert all(torch.equal(v, launch_state[k]) for k,v in launch.state_dict().items())
        result['completed_updates'] = update
        result['actor_steps'] += stats['actor_steps']
        result['executed_landing_plans'] += stats['eligible_samples']
        row = dict(update=update, num_envs=args.num_envs, wall_s=time.monotonic()-started,
            rollout_s=rollout_s, ppo_s=time.monotonic()-ppo_started, **stats,
            trial=compact(summary), metrics=metrics(summary), sample_kind='stochastic_training',
            exploration_std=policy.std.cpu().tolist())
        result['latest_checkpoint'] = save('latest.pt', update)
        result['last_update'] = row
        write(out/f'trials_{update:04d}.json', summary)
        write(out/f'update_{update:04d}.json', row)
        write(out/'progress.json', result)
        print(json.dumps(row), flush=True)
        if update>=8:
            recent=[read(out/f'update_{i:04d}.json') for i in range(update-7,update+1)]
            assert sum(r['actor_steps'] for r in recent)>0, 'Eight updates without an accepted actor step'
        if update%8 == 0 or update == budget:
            checkpoint = save(f'model_{update:04d}.pt', update)
            *_, evaluation = trial(env, standing, launch, policy, limit)
            measured = metrics(evaluation)
            entry = dict(update=update, checkpoint=checkpoint, evaluation=compact(evaluation),
                metrics=measured, seed_qualified=seed_qualified(measured, contract['baseline']),
                sample_kind='deterministic_evaluation')
            write(out/f'evaluation_{update:04d}.json', dict(**entry, cases=evaluation['cases']))
            result['evaluations'].append(entry)
            result['latest_evaluation'] = entry
            if selected_better(measured, result['selected']['metrics'], contract['baseline']):
                result['selected'] = entry
            print(json.dumps(dict(event='evaluation', **entry)), flush=True)
            assert measured['passed'] > 0, 'Latest deterministic evaluation has zero passing worlds'
        result['final_training_frozen_sha256'] = verify_training()
        result['final_frozen_sha256'] = verify()
        assert result['final_training_frozen_sha256'] == result['training_frozen_sha256']
        write(out/'progress.json', result)
    if args.mode == 'smoke':
        assert result['latest_evaluation']['seed_qualified'], 'Smoke no longer satisfies seed admission'
    assert result['actor_steps'] > 0
    result.update(final_checkpoint=save('final.pt', budget),
                  status='SMOKE_COMPLETED' if args.mode == 'smoke' else 'BUDGET_COMPLETED')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('validate', 'smoke', 'train'), required=True)
    p.add_argument('--num-envs', type=int, choices=(45,512), required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--smoke-result', type=Path)
    p.add_argument('--validation-result', type=Path)
    args = p.parse_args()
    assert re.fullmatch('[A-Za-z0-9_-]+', args.run_id)
    assert args.num_envs == (45 if args.mode == 'smoke' else 512)
    frozen = verify_training()
    contract = seed_contract(args.probe)
    if args.mode == 'train':
        for path, status, stage in ((args.smoke_result,'SMOKE_COMPLETED','smoke'),
                                    (args.validation_result,'BATCH_VALIDATED','validate')):
            prior = checked_result(path, status, stage)
            receipt_identity(prior, frozen)
            assert prior['contract'] == contract
    out = HERE/'runs'/args.run_id
    assert not out.exists(), 'Use a fresh run id'
    with exclusive(args.num_envs) as resource:
        out.mkdir(parents=True)
        started = time.monotonic()
        result = dict(status='RUNNING', mode=args.mode, num_envs=args.num_envs, resource=resource,
            utc_start=now(), frozen_sha256=SOURCE_FROZEN, training_frozen_sha256=frozen, contract=contract,
            voltage_v=24, assist_strength=.625, action_dim=16, observation_dim=17, physics_hz=400,
            initialization='selected56 weights; actor air shift; approved slot profile; fresh Adam',
            fixed_takeoff=True, zero_assistance_qualified=False, hardware_qualified=False)
        write(out/'progress.json', result)
        try:
            execute(args, out, result, contract, limit_for(started, 43200 if args.mode=='train' else 3600))
            result['final_frozen_sha256'] = verify()
            result['final_training_frozen_sha256'] = verify_training()
            receipt_identity(result, frozen)
        except BaseException as error:
            result.update(status='ERROR_STOPPED', error=repr(error), traceback=traceback.format_exc())
            print(result['traceback'], flush=True)
        finally:
            result.update(utc_end=now(), wall_s=time.monotonic()-started)
            write(out/'result.json', result)
            write(out/'progress.json', result)
            print(json.dumps({k:result.get(k) for k in ('status','completed_updates','actor_steps','wall_s')}),flush=True)
    return int(result['status']=='ERROR_STOPPED')


if __name__ == '__main__':
    raise SystemExit(main())

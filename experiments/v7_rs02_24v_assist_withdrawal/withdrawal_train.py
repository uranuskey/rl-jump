"""A single admitted assistance level, 512 worlds, 128 new landing PPO updates."""
import argparse
import json
from pathlib import Path
import re
import time
import traceback
from withdrawal_runtime import (HERE, verify, source_contract, load_source, metrics,
                                read, write, sha, now, exclusive, limit_for)
from withdrawal_training_runtime import verify_training, probe_contract, checked_result, exploration
from withdrawal_contract import SOURCE_SHA, SOURCE_LEVEL, PARENT_PHYSICS_SHA, admission, better


def execute(args, out, result, contract, anchor, limit):
    import torch
    from slot_env import SlotEnv
    from slot_learning import Policy, FrozenLaunch
    from slot_ppo import PPO
    from standing_policy import JumpPolicy
    from withdrawal_rollout import trial, compact
    torch.set_num_threads(1)
    torch.manual_seed(105060)
    torch.cuda.manual_seed_all(105060)
    env = SlotEnv(args.num_envs, [contract['profile']], proof=True, abort_dir=out/'aborts')
    standing = JumpPolicy(env.device).eval().requires_grad_(False)
    launch, policy = FrozenLaunch(env.device), Policy(env.device)
    load_source(policy, contract)
    exploration(policy, 0)
    ppo = PPO(policy)
    launch_state = {k: v.clone() for k, v in launch.state_dict().items()}
    level = contract['strength']

    def save(name, update):
        path = out/name
        temporary = path.with_suffix('.pt.tmp')
        torch.save(dict(task='small_step_assistance_withdrawal', model_state_dict=policy.state_dict(),
            actor_optimizer=ppo.actor_optimizer.state_dict(), critic_optimizer=ppo.critic_optimizer.state_dict(),
            update=update, num_envs=args.num_envs, voltage_v=24, assist_strength=level, action_dim=16,
            frozen_sha256=PARENT_PHYSICS_SHA, withdrawal_frozen_sha256=result['frozen_sha256'],
            training_frozen_sha256=result['training_frozen_sha256'], source_checkpoint_sha256=SOURCE_SHA,
            profile=contract['profile'], air_shift_in_actor_m=-.005,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), temporary)
        temporary.replace(path)
        return dict(path=str(path), sha256=sha(path), update=update)

    # Validation includes a same-size reference; 512 worlds repeat the 45 cases
    # unevenly, so do not confuse a different weighting with assistance benefit.
    if args.mode == 'validate':
        *_, reference = trial(env, standing, launch, policy, limit, strength=SOURCE_LEVEL)
        anchor = metrics(reference)
        write(out/'reference625.json', reference)
        assert admission(anchor, contract['anchor_metrics'])['passed'], '512 reference failed reproduction'
        result['batch_anchor_metrics'] = anchor
    initial = save('initial.pt', 0)
    *_, baseline = trial(env, standing, launch, policy, limit, strength=level)
    initial_metrics = metrics(baseline)
    write(out/'baseline.json', baseline)
    initial_admission = admission(initial_metrics, anchor)
    assert env.proof_samples > 0
    assert initial_admission['passed'], ('Lower-assist batch seed failed', initial_admission)
    result.update(completed_updates=0, actor_steps=0, executed_landing_plans=0,
        initial_checkpoint=initial, source_checkpoint=contract['source_checkpoint'],
        initial_metrics=initial_metrics, anchor_metrics=anchor, baseline=compact(baseline),
        selected=dict(checkpoint=initial, metrics=initial_metrics, evaluation=compact(baseline), update=0),
        evaluations=[], prefix_proof_samples=env.proof_samples, frozen_launch_unchanged=True)
    write(out/'progress.json', result)
    print(json.dumps(dict(event='admitted_seed', level=level, metrics=initial_metrics)), flush=True)
    if args.mode == 'validate':
        result['status'] = 'BATCH_VALIDATED'
        return
    budget = 2 if args.mode == 'smoke' else 128
    for update in range(1, budget+1):
        exploration(policy, update-1)
        started = time.monotonic()
        obs, actions, reward, eligible, summary = trial(env, standing, launch, policy, limit,
                                                       strength=level, stochastic=True)
        rollout_s = time.monotonic()-started
        assert bool(eligible.any()), 'No executed landing plans'
        ppo_started = time.monotonic()
        stats = ppo.update(obs, actions, reward, eligible)
        assert all(torch.equal(v, launch_state[k]) for k, v in launch.state_dict().items())
        result['completed_updates'] = update
        result['actor_steps'] += stats['actor_steps']
        result['executed_landing_plans'] += stats['eligible_samples']
        row = dict(update=update, num_envs=args.num_envs, **stats, rollout_s=rollout_s,
                   ppo_s=time.monotonic()-ppo_started, wall_s=time.monotonic()-started,
                   trial=compact(summary), metrics=metrics(summary), sample_kind='stochastic_training',
                   exploration_std=policy.std.cpu().tolist())
        write(out/f'trials_{update:04d}.json', summary)
        write(out/f'update_{update:04d}.json', row)
        result.update(latest_checkpoint=save('latest.pt', update), last_update=row)
        write(out/'progress.json', result)
        print(json.dumps(row), flush=True)
        if update >= 8:
            recent = [read(out/f'update_{i:04d}.json') for i in range(update-7, update+1)]
            assert sum(r['actor_steps'] for r in recent) > 0, 'Eight updates without an accepted actor step'
        if update % 8 == 0 or update == budget:
            checkpoint = save(f'model_{update:04d}.pt', update)
            *_, evaluation = trial(env, standing, launch, policy, limit, strength=level)
            measured = metrics(evaluation)
            admitted = admission(measured, anchor)
            entry = dict(update=update, checkpoint=checkpoint, evaluation=compact(evaluation),
                         metrics=measured, admission=admitted, sample_kind='deterministic_evaluation')
            write(out/f'evaluation_{update:04d}.json', dict(**entry, cases=evaluation['cases']))
            result['evaluations'].append(entry)
            result['latest_evaluation'] = entry
            if better(measured, result['selected']['metrics'], anchor):
                result['selected'] = entry
            print(json.dumps(dict(event='evaluation', **entry)), flush=True)
            assert measured['passed'] > 0, 'Deterministic evaluation has no passing worlds'
        result['final_training_frozen_sha256'] = verify_training()
        assert result['final_training_frozen_sha256'] == result['training_frozen_sha256']
        write(out/'progress.json', result)
    if args.mode == 'smoke':
        assert result['latest_evaluation']['admission']['passed'], 'Smoke lost qualification'
    assert result['actor_steps'] > 0
    result.update(final_checkpoint=save('final.pt', budget),
                  status='SMOKE_COMPLETED' if args.mode == 'smoke' else 'BUDGET_COMPLETED')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', required=True, choices=('smoke', 'validate', 'train'))
    p.add_argument('--run-id', required=True)
    p.add_argument('--num-envs', required=True, type=int, choices=(45, 512))
    p.add_argument('--probe-result', type=Path, required=True)
    p.add_argument('--smoke-result', type=Path)
    p.add_argument('--validation-result', type=Path)
    args = p.parse_args()
    assert re.fullmatch('[A-Za-z0-9_-]+', args.run_id)
    assert args.num_envs == (45 if args.mode == 'smoke' else 512)
    frozen, training_frozen = verify(), verify_training()
    contract = probe_contract(args.probe_result)
    anchor = contract['anchor_metrics']
    if args.mode == 'train':
        for path, status, stage in ((args.smoke_result, 'SMOKE_COMPLETED', 'smoke'),
                                    (args.validation_result, 'BATCH_VALIDATED', 'validate')):
            prior = checked_result(path, status, stage)
            assert prior['contract'] == contract
            assert prior['training_frozen_sha256'] == prior['final_training_frozen_sha256'] == training_frozen
        anchor = prior['batch_anchor_metrics']
    out = HERE/'runs'/args.run_id
    assert not out.exists(), 'Use a fresh run id'
    with exclusive(args.num_envs) as resources:
        out.mkdir(parents=True)
        started = time.monotonic()
        result = dict(status='RUNNING', mode=args.mode, resources=resources, num_envs=args.num_envs,
            utc_start=now(), frozen_sha256=frozen, training_frozen_sha256=training_frozen,
            contract=contract, voltage_v=24, assist_strength=contract['strength'], action_dim=16,
            observation_dim=17, physics_hz=400, fixed_takeoff_weights=True,
            zero_assistance_qualified=False, hardware_qualified=False,
            initialization='selected112 weights; unchanged slot controller; fresh Adam',
            no_further_assistance_reduction=True)
        write(out/'progress.json', result)
        try:
            execute(args, out, result, contract, anchor, limit_for(started, 43200 if args.mode == 'train' else 3600))
            result.update(final_frozen_sha256=verify(), final_training_frozen_sha256=verify_training())
            assert result['final_frozen_sha256'] == frozen
            assert result['final_training_frozen_sha256'] == training_frozen
        except BaseException as error:
            result.update(status='ERROR_STOPPED', error=repr(error), traceback=traceback.format_exc())
            print(result['traceback'], flush=True)
        finally:
            result.update(utc_end=now(), wall_s=time.monotonic()-started)
            write(out/'result.json', result)
            write(out/'progress.json', result)
    return int(result['status'] == 'ERROR_STOPPED')


if __name__ == '__main__':
    raise SystemExit(main())

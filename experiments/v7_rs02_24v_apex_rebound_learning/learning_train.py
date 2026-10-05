"""A single admitted assistance level, 512 worlds, 128 new landing PPO updates."""
import argparse
import json
from pathlib import Path
import re
import time
import traceback
from learning_runtime import (HERE, verify, source_contract, load_source, metrics,
                                read, write, sha, now, exclusive, limit_for)
from learning_runtime import (verify_training, learning_evidence, checked_result, exploration,
                              PARENT_PROBE_SHA)
from learning_contract import learning_admission, qualification, selected_better, diagnostic_better
from withdrawal_contract import SOURCE_SHA, PARENT_PHYSICS_SHA


def execute(args, out, result, contract, anchor, limit):
    import torch
    from apex_env import ApexEnv
    from slot_learning import Policy, FrozenLaunch
    from slot_ppo import PPO
    from standing_policy import JumpPolicy
    from apex_rollout import trial, compact
    torch.set_num_threads(1)
    torch.manual_seed(105060)
    torch.cuda.manual_seed_all(105060)
    env = ApexEnv(args.num_envs, [contract['profile']], proof=True, abort_dir=out/'aborts')
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
        torch.save(dict(task='apex_rebound_learning', model_state_dict=policy.state_dict(),
            actor_optimizer=ppo.actor_optimizer.state_dict(), critic_optimizer=ppo.critic_optimizer.state_dict(),
            update=update, num_envs=args.num_envs, voltage_v=24, takeoff_assist_strength=.625, after_apex_assist_strength=level,
            assist_schedule='observed_COM_apex_then_100ms_ramp', action_dim=16,
            frozen_sha256=PARENT_PHYSICS_SHA, apex_frozen_sha256=PARENT_PROBE_SHA,
            training_frozen_sha256=result['training_frozen_sha256'], source_checkpoint_sha256=SOURCE_SHA,
            profile=contract['profile'], air_shift_in_actor_m=-.005,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), temporary)
        temporary.replace(path)
        return dict(path=str(path), sha256=sha(path), update=update)

    initial = save('initial.pt', 0)
    *_, baseline = trial(env, standing, launch, policy, limit, strength=level)
    initial_metrics = metrics(baseline)
    write(out/'baseline.json', baseline)
    initial_admission = learning_admission(initial_metrics, anchor)
    assert env.proof_samples > 0
    result.update(completed_updates=0, actor_steps=0, executed_landing_plans=0,
        initial_checkpoint=initial, source_checkpoint=contract['source_checkpoint'],
        initial_metrics=initial_metrics, anchor_metrics=anchor, baseline=compact(baseline),
        selected=None,
        diagnostic_candidate=dict(checkpoint=initial, metrics=initial_metrics,
                                  evaluation=compact(baseline), update=0),
        evaluations=[], prefix_proof_samples=env.proof_samples, frozen_launch_unchanged=True)
    result['initial_learning_admission'] = initial_admission
    result['initial_final_qualification'] = qualification(initial_metrics, anchor)
    write(out/'progress.json', result)
    assert initial_admission['passed'], ('Learning entry failed', initial_admission)
    print(json.dumps(dict(event='learning_entry_passed', level=level, metrics=initial_metrics,
                          final_qualification=result['initial_final_qualification'])), flush=True)
    budget = 128
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
        if update == 2 or update % 8 == 0 or update == budget:
            checkpoint = save(f'model_{update:04d}.pt', update)
            *_, evaluation = trial(env, standing, launch, policy, limit, strength=level)
            measured = metrics(evaluation)
            admitted = qualification(measured, anchor)
            entry = dict(update=update, checkpoint=checkpoint, evaluation=compact(evaluation),
                         metrics=measured, admission=admitted,
                         learning_admission=learning_admission(measured, anchor),
                         sample_kind='deterministic_evaluation')
            write(out/f'evaluation_{update:04d}.json', dict(**entry, cases=evaluation['cases']))
            result['evaluations'].append(entry)
            result['latest_evaluation'] = entry
            incumbent = None if result['selected'] is None else result['selected']['metrics']
            if selected_better(measured, incumbent, anchor):
                result['selected'] = entry
            if diagnostic_better(measured, result['diagnostic_candidate']['metrics'], anchor):
                result['diagnostic_candidate'] = entry
            print(json.dumps(dict(event='evaluation', **entry)), flush=True)
            assert measured['passed'] > 0, 'Deterministic evaluation has no passing worlds'
        result['final_training_frozen_sha256'] = verify_training()
        assert result['final_training_frozen_sha256'] == result['training_frozen_sha256']
        write(out/'progress.json', result)
    assert result['actor_steps'] > 0
    result.update(final_checkpoint=save('final.pt', budget),
                  status='BUDGET_COMPLETED')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-id', required=True)
    p.add_argument('--num-envs', required=True, type=int, choices=(512,))
    p.add_argument('--probe-result', type=Path, required=True)
    p.add_argument('--preflight-result', type=Path, required=True)
    args = p.parse_args()
    assert re.fullmatch('[A-Za-z0-9_-]+', args.run_id)
    frozen, training_frozen = verify(), verify_training()
    contract, evidence = learning_evidence(args.probe_result)
    preflight = checked_result(args.preflight_result, 'LEARNING_ENTRY_VERIFIED', 'preflight')
    assert preflight['contract'] == contract and preflight['evidence'] == evidence
    anchor = evidence['historical_reference_metrics']
    prerequisites = dict(preflight=dict(path=str(args.preflight_result),
                                        sha256=sha(args.preflight_result)), reused_evidence=evidence)
    out = HERE/'runs'/args.run_id
    assert not out.exists(), 'Use a fresh run id'
    with exclusive(args.num_envs) as resources:
        out.mkdir(parents=True)
        started = time.monotonic()
        result = dict(status='RUNNING', mode='train', resources=resources, num_envs=args.num_envs,
            utc_start=now(), frozen_sha256=frozen, training_frozen_sha256=training_frozen,
            contract=contract, prerequisites=prerequisites, voltage_v=24,
            assist_strength=None, takeoff_assist_strength=.625,
            after_apex_assist_strength=contract['strength'], assist_schedule='observed_COM_apex_then_100ms_ramp',
            completed_updates=0, actor_steps=0, executed_landing_plans=0, action_dim=16,
            observation_dim=17, physics_hz=400, fixed_takeoff_weights=True,
            zero_assistance_qualified=False, hardware_qualified=False,
            initialization='selected112 weights; unchanged slot controller; fresh Adam',
            learning_rebound_limit_mps=.10, final_rebound_limit_mps=1e-6,
            unqualified_seed_can_learn=True, unqualified_seed_cannot_be_selected=True,
            no_further_assistance_reduction=True)
        write(out/'progress.json', result)
        try:
            execute(args, out, result, contract, anchor, limit_for(started, 43200))
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

"""Frozen launch, fixed lower assistance, fresh landing PPO with rollback."""
import time
from pathlib import Path
import transfer_runtime as rt
from transfer_contract import BEFORE, AFTER, VARIANT, admission, promote, final_qualified


def stack(contract, checkpoint, worlds, out, native=False, before=BEFORE, after=AFTER):
    import torch
    from fix_env import ConstraintEnv
    from slot_learning import Policy, FrozenLaunch
    from standing_policy import JumpPolicy
    torch.set_num_threads(1)
    torch.manual_seed(105068); torch.cuda.manual_seed_all(105068)
    assert rt.sha(checkpoint['path']) == checkpoint['sha256']
    state = torch.load(checkpoint['path'], map_location='cpu', weights_only=True)
    assert state['voltage_v'] == 24 and state['action_dim'] == 16 and state['profile'] == contract['profile']
    ancestors = {contract['source_checkpoint']['sha256'], contract['original_reference']['source_checkpoint']['sha256']}
    if checkpoint['sha256'] not in ancestors:
        assert state['training_frozen_sha256'] == rt.verify() and state['parent_fix_sha256'] == rt.PARENT_SHA
        assert state['constraint_variant'] == VARIANT
        assert (state['takeoff_assist_strength'],state['after_apex_assist_strength']) == (BEFORE,AFTER)
    env = ConstraintEnv(worlds, [contract['profile']], before=before, after=after, variant=VARIANT,
                        fast_backend=not native, proof=True, abort_dir=out/'aborts')
    standing = JumpPolicy(env.device).eval().requires_grad_(False)
    launch, policy = FrozenLaunch(env.device), Policy(env.device)
    policy.load_state_dict(state['model_state_dict'], strict=True)
    return env, standing, launch, policy


def measured_trial(parts, out, name, contract, limit, native=False, stochastic=False):
    from probe_rollout import trial
    from fix_audit import audit
    env, standing, launch, policy = parts
    trace = out/(name+'_traces.npz') if native else None
    obs, action, reward, eligible, summary = trial(env, standing, launch, policy, limit,
        before=env.before, after=env.after, stochastic=stochastic, trace_path=trace)
    path = out/(name+'.json')
    rt.write(path, summary)
    m = rt.metrics(summary)
    anchors = contract['anchors_native'] if native else contract['anchors_batch']
    pre = summary['max_pre_apex_mimic_v_rad_s']
    row = dict(metrics=m, summary=path.name, summary_sha256=rt.sha(path),
        strict_admission=admission(m, anchors, pre), learning_admission=admission(m, anchors, pre, learning=True),
        pre_apex_v_rad_s=pre, max_mimic_v_rad_s=summary['max_mimic_v_rad_s'],
        physics_receipt=env.physics_receipt, sample_kind=summary['sample_kind'],
        before=env.before, after=env.after, prefix_proof_samples=env.proof_samples)
    assert env.proof_samples > 0
    if native:
        row.update(trace=trace.name, trace_sha256=rt.sha(trace), mass_kg=env.mass,
                   audits=audit(trace, summary, env.mass, contract['profile'], env.before, env.after, VARIANT))
    return obs, action, reward, eligible, row


def preflight(out, result, limit):
    c = result['contract']
    for name, worlds, native in [('native',45,True), ('batch',512,False)]:
        result['stage'] = 'source6125_'+name; rt.write(out/'progress.json', result)
        parts = stack(c, c['source_checkpoint'], worlds, out, native=native)
        *_, row = measured_trial(parts, out, 'source6125_'+name, c, limit, native=native)
        result[name] = row; rt.write(out/'progress.json', result)
        assert row['learning_admission']['passed'], ('Lower target learning entry failed', row)
        if native: assert row['audits']['status'] == 'PASS'
        del parts
    result.update(status='PREFLIGHT_COMPLETED', final_qualification_claimed=False)


def learn(out, result, limit, budget, worlds):
    import torch
    from slot_ppo import PPO
    c = result['contract']
    parts = stack(c, c['source_checkpoint'], worlds, out)
    env, standing, launch, policy = parts
    launch_state = {k:v.clone() for k,v in launch.state_dict().items()}
    rt.exploration(policy, 0)
    ppo = PPO(policy)

    def save(name, update):
        path = out/name; tmp = path.with_suffix('.pt.tmp')
        torch.save(dict(task='rotor6125_learning', model_state_dict=policy.state_dict(),
            actor_optimizer=ppo.actor_optimizer.state_dict(), critic_optimizer=ppo.critic_optimizer.state_dict(),
            update=update, num_envs=worlds, voltage_v=24, profile=c['profile'], action_dim=16,
            takeoff_assist_strength=BEFORE, after_apex_assist_strength=AFTER, constraint_variant=VARIANT,
            training_frozen_sha256=result['frozen_sha256'], parent_fix_sha256=rt.PARENT_SHA,
            source_checkpoint_sha256=c['source_checkpoint']['sha256'],
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), tmp)
        tmp.replace(path)
        return dict(path=str(path), sha256=rt.sha(path), update=update)

    result.update(initial_checkpoint=save('initial.pt',0), completed_updates=0, actor_steps=0,
                  selected=None, evaluations=[], initialization='source80 weights with fresh Adam')
    *_, seed = measured_trial(parts, out, 'initial', c, limit)
    result['initial'] = seed
    assert seed['learning_admission']['passed'], ('Fresh initial learning entry failed', seed)
    rt.write(out/'progress.json', result)
    for update in range(1,budget+1):
        rt.exploration(policy, update-1)
        started = time.monotonic()
        obs, action, reward, eligible, sample = measured_trial(parts, out, f'trial_{update:04d}', c, limit, stochastic=True)
        assert bool(eligible.any()), 'No executed landing plans'
        stats = ppo.update(obs, action, reward, eligible)
        assert all(torch.equal(v,launch_state[k]) for k,v in launch.state_dict().items())
        result.update(completed_updates=update, actor_steps=result['actor_steps']+stats['actor_steps'],
                      latest_checkpoint=save('latest.pt',update))
        row = dict(update=update, sample=sample, stats=stats, wall_s=time.monotonic()-started,
                   exploration_std=policy.std.cpu().tolist(), sample_kind='stochastic_training')
        rt.write(out/f'update_{update:04d}.json',row)
        result['last_update'] = row
        if update >= 8:
            assert sum(rt.read(out/f'update_{i:04d}.json')['stats']['actor_steps'] for i in range(update-7,update+1)) > 0
        if update == 2 or update % 8 == 0 or update == budget:
            ck = save(f'model_{update:04d}.pt',update)
            *_, evaluation = measured_trial(parts,out,f'evaluation_{update:04d}',c,limit)
            entry = dict(update=update,checkpoint=ck,**evaluation)
            result['evaluations'].append(entry)
            result['latest_evaluation'] = entry
            if promote(entry): result['selected'] = entry
            assert evaluation['metrics']['passed'] > 0, 'No deterministic passing cases'
            if budget == 2: assert evaluation['learning_admission']['passed'], 'Smoke learning gate failed'
        result['final_frozen_sha256'] = rt.verify()
        assert result['final_frozen_sha256'] == result['frozen_sha256']
        rt.write(out/'progress.json',result)
        print(dict(update=update, actor_steps=result['actor_steps'],
                   selected_update=None if result['selected'] is None else result['selected']['update']),flush=True)
    assert result['actor_steps'] > 0
    result.update(final_checkpoint=save('final.pt',budget),
                  status='SMOKE_COMPLETED' if budget==2 else 'TRAIN_COMPLETED')


def evaluate(out,result,limit,train):
    c=result['contract']; result['models']={}
    selected=train['selected']
    entries=[('reference625',c['original_reference']['source_checkpoint'],.625,.625,1),
             ('seed',c['source_checkpoint'],BEFORE,AFTER,1),
             ('selected' if selected else 'candidate',(selected['checkpoint'] if selected else train['final_checkpoint']),BEFORE,AFTER,3),
             ('latest',train['final_checkpoint'],BEFORE,AFTER,3)]
    for name, ck, before, after, repeats in entries:
        row=dict(checkpoint=ck,before=before,after=after,batch=[])
        result['models'][name]=row; result['stage']=name+'_native'
        rt.write(out/'progress.json',result)
        parts=stack(c,ck,45,out,native=True,before=before,after=after)
        *_,row['native']=measured_trial(parts,out,name+'_native',c,limit,native=True)
        del parts
        parts=stack(c,ck,512,out,before=before,after=after)
        for i in range(1,repeats+1):
            result['stage']=f'{name}_batch_{i}'; rt.write(out/'progress.json',result)
            *_,sample=measured_trial(parts,out,f'{name}_batch_{i}',c,limit)
            sample['repeat']=i; row['batch'].append(sample)
            rt.write(out/'progress.json',result)
        del parts
        row['comparison_passed']=final_qualified(row,repeats)
        row['history_strict_passed']=(selected is not None if name=='selected' else
            train['latest_evaluation']['strict_admission']['passed'] if name=='latest' else False)
        row['qualified']=(name in ('selected','latest') and row['history_strict_passed'] and row['comparison_passed'])
    result.update(status='EVALUATION_COMPLETED',target_qualified=any(
        result['models'].get(n,{}).get('qualified',False) for n in ('selected','latest')))

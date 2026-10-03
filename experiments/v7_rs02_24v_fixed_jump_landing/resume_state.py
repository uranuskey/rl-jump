"""Explicit provenance and complete PPO state restoration for a smaller batch."""
import json
from pathlib import Path
import torch
import bootstrap
from bootstrap import HERE, ROOT
from runtime import sha, verify


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify_resume():
    frozen = verify()
    path = HERE/'RESUME_FROZEN.json'
    manifest = read(path)
    assert manifest['policy_frozen_sha256']==frozen
    for rel, digest in manifest['sha256'].items():
        if sha(ROOT/rel)!=digest:
            raise RuntimeError('Resume source changed: '+rel)
    return frozen, sha(path)


def source_state(folder, checkpoint, expected_update, frozen):
    source, receipt = read(folder/'result.json'), read(folder/'exit_receipt.json')
    assert receipt['process_exited'] and receipt['exit_code']==1 and receipt['stage']=='train'
    assert source['status']=='ERROR_STOPPED' and source['error']=="RuntimeError('STOP requested')"
    assert source['frozen_sha256']==receipt['frozen_sha256']==frozen
    assert source['completed_updates']==source['latest_checkpoint']['update']==expected_update
    assert 0<expected_update<128 and source['num_envs']==2048
    assert source['voltage_v']==24 and source['assist_strength']==.625
    assert source['prefix_proof']['status']=='PASS'
    assert sha(checkpoint)==source['latest_checkpoint']['sha256']
    updates = [read(folder/f'update_{i:04d}.json') for i in range(1, expected_update+1)]
    assert all(row['update']==i+1 and row['trial']['worlds']==2048 for i, row in enumerate(updates))
    assert sum(row['actor_steps'] for row in updates)==source['actor_steps']
    assert sum(row['eligible_samples'] for row in updates)==source['landing_transitions']
    for entry in [source['selected'], *source['evaluations']]:
        assert sha(entry['checkpoint']['path'])==entry['checkpoint']['sha256']
    return source


def restore(policy, ppo, checkpoint, expected_update, frozen, *, cuda=True):
    # Load on CPU so RNG ByteTensors do not get remapped onto CUDA.
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert state['update']==expected_update and state['frozen_sha256']==frozen
    assert state['voltage_v']==24 and state['assist_strength']==.625
    policy.load_state_dict(state['model_state_dict'], strict=True)
    ppo.actor_optimizer.load_state_dict(state['actor_optimizer'])
    ppo.critic_optimizer.load_state_dict(state['critic_optimizer'])
    steps = {}
    for name, optimizer in [('actor', ppo.actor_optimizer), ('critic', ppo.critic_optimizer)]:
        params = [p for group in optimizer.param_groups for p in group['params']]
        assert len(optimizer.state)==len(params)>0
        assert all(all(k in optimizer.state[p] for k in ('step', 'exp_avg', 'exp_avg_sq')) for p in params)
        steps[name] = sorted({int(optimizer.state[p]['step']) for p in params})
        assert min(steps[name])>0
        assert all(bool(torch.isfinite(optimizer.state[p][k]).all()) for p in params for k in ('exp_avg', 'exp_avg_sq'))
    assert all(bool(torch.isfinite(v).all()) for v in policy.state_dict().values())
    torch.set_rng_state(state['torch_rng'])
    if cuda:
        assert len(state['cuda_rng'])==torch.cuda.device_count()
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return dict(update=expected_update, optimizer_steps=steps, cpu_rng_restored=True,
                cuda_rng_restored=cuda, model_and_both_optimizers_restored=True)

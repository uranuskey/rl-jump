"""Actor-only interpolation, independently checked against immutable endpoints."""
from copy import deepcopy
import hashlib
from pathlib import Path
import filter_runtime as rt
from filter_contract import FRACTIONS,blend_parameter

def actor_hash(policy):
    digest=hashlib.sha256()
    for key,value in sorted(policy.actor.state_dict().items()):
        digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def sources(contract):
    import torch
    from slot_learning import Policy
    states=[];keys=None
    for suffix in ('a','b'):
        ck=contract['source_'+suffix];assert rt.sha(ck['path'])==ck['sha256']
        state=torch.load(ck['path'],map_location='cpu',weights_only=True)
        assert state['voltage_v']==24 and state['action_dim']==16 and state['profile']==contract['profile']
        assert state['tail_frozen_sha256']==rt.PARENT_SHA
        p=Policy('cpu');p.load_state_dict(state['model_state_dict'],strict=True)
        assert actor_hash(p)==contract['source_actor_'+suffix]
        actor_keys={'actor.'+k for k,_ in p.actor.named_parameters()}
        if keys is None:keys=actor_keys
        else:assert keys==actor_keys
        states.append(state)
    assert keys and set(states[0]['model_state_dict'])==set(states[1]['model_state_dict'])
    for key in keys:
        a,b=(s['model_state_dict'][key] for s in states)
        assert a.dtype==b.dtype and a.shape==b.shape and a.is_floating_point()
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
    return states,keys

def expected_model(states,keys,fraction):
    model=deepcopy(states[0]['model_state_dict'])
    for key in keys:model[key]=blend_parameter(model[key],states[1]['model_state_dict'][key],fraction)
    return model

def verify_candidate(candidate,contract):
    import torch
    from slot_learning import Policy
    index=candidate['index'];assert candidate['fraction']==FRACTIONS[index]
    ck=candidate['checkpoint'];assert ck['global_update']==0 and rt.sha(ck['path'])==ck['sha256']
    state=torch.load(ck['path'],map_location='cpu',weights_only=True)
    assert state['filter_frozen_sha256']==rt.verify() and state['voltage_v']==24 and state['action_dim']==16
    assert state['profile']==contract['profile'] and state['requires_fresh_optimizer'] is True
    assert state['source_sha256']==[contract['source_a']['sha256'],contract['source_b']['sha256']]
    assert state['fraction']==candidate['fraction'] and state['prior_shared_updates']==14
    assert not any(k in state for k in ('actor_optimizer','critic_optimizer','torch_rng','cuda_rng'))
    states,keys=sources(contract);model=expected_model(states,keys,candidate['fraction'])
    assert set(model)==set(state['model_state_dict'])
    for key,value in model.items():assert torch.equal(value,state['model_state_dict'][key]),key
    p=Policy('cpu');p.load_state_dict(state['model_state_dict'],strict=True)
    digest=actor_hash(p);assert digest==candidate['actor_hash']
    assert digest not in (contract['source_actor_a'],contract['source_actor_b'])
    return state

def prepare(run,contract):
    import torch
    from slot_learning import Policy
    torch.set_num_threads(1)
    folder=run/'candidates';assert not folder.exists();folder.mkdir()
    states,keys=sources(contract);out=[];seen=set()
    for index,fraction in enumerate(FRACTIONS):
        model=expected_model(states,keys,fraction)
        p=Policy('cpu');p.load_state_dict(model,strict=True);digest=actor_hash(p)
        assert digest not in seen and digest not in (contract['source_actor_a'],contract['source_actor_b']);seen.add(digest)
        path=folder/f'candidate_{index:02d}.pt'
        torch.save(dict(model_state_dict=model,voltage_v=24,action_dim=16,profile=contract['profile'],
            filter_frozen_sha256=rt.verify(),fraction=fraction,prior_shared_updates=14,global_update=0,
            requires_fresh_optimizer=True,source_sha256=[contract['source_a']['sha256'],contract['source_b']['sha256']]),path)
        record=dict(index=index,fraction=fraction,actor_hash=digest,
                    checkpoint=dict(path=str(path),sha256=rt.sha(path),global_update=0))
        verify_candidate(record,contract);out.append(record)
    rt.write(run/'candidates.json',out)
    return out

"""Verify immutable parent proofs and every finite actor candidate on CPU."""
import argparse
from pathlib import Path
import filter_runtime as rt
from filter_candidates import prepare,sources,verify_candidate
from filter_contract import FRACTIONS

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists()
    assert not (rt.ROOT/'experiments/v7_jump_in_place/PHYSICS_LOCK.json').exists()
    f,c=rt.verify(),rt.contract()
    folder=a.output.parent/'cpu_candidates';assert not folder.exists();folder.mkdir(parents=True)
    candidates=prepare(folder,c)
    states,keys=sources(c)
    import torch
    for record in candidates:
        state=verify_candidate(record,c);model=state['model_state_dict']
        assert any(not torch.equal(model[k],states[0]['model_state_dict'][k]) for k in keys)
        assert all(torch.equal(model[k],states[0]['model_state_dict'][k]) for k in model if k not in keys)
    out=dict(status='PREFLIGHT_VALIDATED',utc=rt.now(),frozen_sha256=f,contract=c,candidates=candidates,
             physical_trials=0,completed_updates=0,actor_steps=0,actor_parameter_keys=sorted(keys),
             unchanged_non_actor_keys=sorted(set(states[0]['model_state_dict'])-keys),
             fractions=list(FRACTIONS),parent_failures_preserved=True,
             qualification_required=True,requires_fresh_optimizer=True)
    rt.write(a.output,out)
    print({k:v for k,v in out.items() if k not in ('contract','candidates')},flush=True)

if __name__=='__main__':main()

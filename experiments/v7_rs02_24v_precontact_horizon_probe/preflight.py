"""CPU-only source, frozen chain and complete old qualification evidence."""
import argparse,json
from pathlib import Path
import horizon_runtime as rt
from horizon_contract import HORIZONS,profile,controller_id

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    f,c=rt.verify(),rt.contract();old=rt.source_evidence(c)
    import torch
    from slot_learning import Policy
    from recovery_worker import actor_hash,sample as original_sample
    from horizon_worker import sample as candidate_sample
    assert candidate_sample is original_sample
    torch.set_num_threads(1)
    state=torch.load(c['source_checkpoint']['path'],map_location='cpu',weights_only=True)
    assert state['recovery_frozen_sha256']==rt.PARENT_SHA and state['profile']==c['source_profile']
    assert state['global_update']==3 and state['voltage_v']==24 and state['action_dim']==16
    assert all(bool(torch.isfinite(v).all()) for v in state['model_state_dict'].values())
    policy=Policy('cpu');policy.load_state_dict(state['model_state_dict'],strict=True);assert actor_hash(policy)==rt.SOURCE_ACTOR
    assert old[0]['qualified'] and not old[1]['qualified']
    profiles=[profile(c['source_profile'],i) for i in range(len(HORIZONS))]
    assert all({k for k in c['source_profile'] if p[k]!=c['source_profile'][k]}=={'name','horizon_s'} for p in profiles)
    out=dict(status='PREFLIGHT_VALIDATED',utc=rt.now(),frozen_sha256=f,contract=c,physical_trials=0,new_ppo_updates=0,
        source_actor_hash=rt.SOURCE_ACTOR,source_profile=c['source_profile'],source_qualified_strength=0.25,source_failed_strength=0.2375,
        candidate_controllers=[dict(profile=p,controller_id=controller_id(rt.SOURCE_ACTOR,p),qualified=False) for p in profiles],
        no_inherited_candidate_qualification=True,final_physical_gates_unchanged=True)
    rt.write(a.output,out);print(json.dumps({k:v for k,v in out.items() if k not in ('contract','candidate_controllers')}))
if __name__=='__main__':main()

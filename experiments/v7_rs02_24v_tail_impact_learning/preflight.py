"""Read-only source/parent proof; GPU qualification is still required at35%."""
import argparse
from pathlib import Path
import tail_runtime as rt
from tail_contract import OBJECTIVE,penalty,BUDGET,PRIOR_SHARED_UPDATES
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    f,c=rt.verify(),rt.contract()
    import torch
    state=torch.load(c['source_checkpoint']['path'],map_location='cpu',weights_only=True)
    assert state['profile']==c['source_profile']==c['profile'] and state['voltage_v']==24 and state['action_dim']==16
    assert state[c['source_frozen_field']]==c['source_parent_frozen_sha256']
    from slot_learning import Policy
    from tail_worker import actor_hash
    policy=Policy('cpu');policy.load_state_dict(state['model_state_dict'],strict=True)
    assert actor_hash(policy)==c['source_actor_hash']
    assert BUDGET+PRIOR_SHARED_UPDATES==128 and penalty(340)==-20 and penalty(350)==-80
    out=dict(status='PREFLIGHT_VALIDATED',utc=rt.now(),frozen_sha256=f,contract=c,source_actor_hash=actor_hash(policy),
        physical_trials=0,source35_requalification_required=True,training_objective=OBJECTIVE,
        physical_entry_and_final_gates_unchanged=True,old_failure_preserved=True)
    rt.write(a.output,out)
    print({k:v for k,v in out.items() if k not in ('contract',)},flush=True)
if __name__=='__main__':main()

"""CPU source audit; no new physical trials and no inherited qualification."""
import argparse,json
from pathlib import Path
import recovery_runtime as rt
from recovery_contract import BUDGET,PRIOR_SHARED_UPDATES,qualification,learning_evidence,ACTOR_LR

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    f,c=rt.verify(),rt.contract()
    import torch
    from slot_learning import Policy
    from recovery_worker import actor_hash
    torch.set_num_threads(1)
    state=torch.load(c['source_checkpoint']['path'],map_location='cpu',weights_only=True)
    assert state[c['source_frozen_field']]==rt.PARENT_SHA and state['profile']==c['profile']
    assert state['global_update']==1 and state['before']==state['after']==0.25 and state['level_index']==0
    assert state['actor_optimizer']['state'] and state['critic_optimizer']['state'] and state['torch_rng'].numel()>0 and state['cuda_rng']
    assert all(0<g['lr']<=ACTOR_LR for g in state['actor_optimizer']['param_groups'])
    policy=Policy('cpu');policy.load_state_dict(state['model_state_dict'],strict=True)
    assert actor_hash(policy)==c['source_actor_hash'] and BUDGET+PRIOR_SHARED_UPDATES==128
    old=rt.source_evidence(c);assert not qualification(old['qualification'])
    evidence=learning_evidence(old['qualification'],c)
    assert evidence==old['learning_evidence'] and evidence['near_native_peak_entry'] and evidence['passed']
    from boundary_row_audit import check_row
    for native,row in [(True,old['qualification']['native'])]+[(False,r) for r in old['qualification']['batch']]:
        check_row(Path(row['evidence_dir']),row,c,native)
    rt.write(a.output,dict(status='PREFLIGHT_VALIDATED',utc=rt.now(),frozen_sha256=f,contract=c,physical_trials=0,
        source_actor_hash=actor_hash(policy),source_strength=0.25,source_qualification=False,source_learning_entry=True,
        learning_evidence=evidence,first_update_resume_source_optimizer_rng=True,
        no_same_failed_actor_retest=True,final_gates_unchanged=True,learning_thresholds_unchanged=True))
    print(json.dumps(dict(status='PREFLIGHT_VALIDATED',frozen=f,physical_trials=0,source_actor=actor_hash(policy),source_qualified=False)))
if __name__=='__main__':main()

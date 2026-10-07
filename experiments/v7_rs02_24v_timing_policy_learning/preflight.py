"""Independent CPU source and raw-gate audit before any new physical trials."""
import argparse,json
from pathlib import Path
import timing_runtime as rt
from timing_contract import BUDGET,PRIOR_SHARED_UPDATES,qualification,learning_evidence,ACTOR_LR,EXPLORATION_OFFSET

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    f,c=rt.verify(),rt.contract()
    import torch
    from slot_learning import Policy
    from timing_worker import actor_hash
    torch.set_num_threads(1)
    state=torch.load(c['source_checkpoint']['path'],map_location='cpu',weights_only=True)
    assert state[c['source_frozen_field']]==c['source_parent_frozen_sha256']==rt.SOURCE_PARENT_SHA
    assert state['profile']==c['source_profile'] and state['profile']!=c['profile']
    assert state['global_update']==3 and state['before']==state['after']==0.25 and state['level_index']==0
    assert state['voltage_v']==24 and state['action_dim']==16
    assert state['actor_optimizer']['state'] and state['critic_optimizer']['state'] and state['torch_rng'].numel()>0 and state['cuda_rng']
    assert all(bool(torch.isfinite(v).all()) for v in state['model_state_dict'].values())
    policy=Policy('cpu');policy.load_state_dict(state['model_state_dict'],strict=True)
    assert actor_hash(policy)==c['source_actor_hash'] and BUDGET+PRIOR_SHARED_UPDATES==128
    from boundary_row_audit import check_row
    evidence={}
    for key in ('source_qualified','source_failed'):
        old=rt.source_evidence(c,key);assert qualification(old['qualification'])==(key=='source_qualified')
        for native,row in [(True,old['qualification']['native'])]+[(False,r) for r in old['qualification']['batch']]:
            check_row(Path(row['evidence_dir']),row,c,native)
        evidence[key]=dict(original_qualified=old['qualified'],original_learning_entry=old['learning_entry'],
            recomputed_learning=learning_evidence(old['qualification'],c))
    assert evidence['source_qualified']['recomputed_learning']['passed']
    assert not evidence['source_failed']['recomputed_learning']['passed']
    assert c['first_update_resume_source_optimizer_rng'] is False
    rt.write(a.output,dict(status='PREFLIGHT_VALIDATED',utc=rt.now(),frozen_sha256=f,contract=c,physical_trials=0,
        source_actor_hash=actor_hash(policy),source_checkpoint_strength=0.25,source_qualification_strength=0.2375,
        failed_target_strength=0.225,source_evidence=evidence,first_update_resume_source_optimizer_rng=False,
        exploration_update_offset=EXPLORATION_OFFSET,no_same_failed_actor_retest=True,
        final_gates_unchanged=True,learning_thresholds_unchanged=True))
    print(json.dumps(dict(status='PREFLIGHT_VALIDATED',frozen=f,physical_trials=0,source_actor=actor_hash(policy))))
if __name__=='__main__':main()

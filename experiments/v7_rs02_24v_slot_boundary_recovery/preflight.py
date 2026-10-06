"""CPU checks on the frozen source, archived traces and preserved failures."""
import argparse,json
from pathlib import Path
import boundary_runtime as rt
from boundary_contract import BUDGET,PRIOR_SHARED_UPDATES,qualification,learning_evidence,admission

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    f,c=rt.verify(),rt.contract()
    import torch
    from slot_learning import Policy
    from boundary_worker import actor_hash
    torch.set_num_threads(1)
    state=torch.load(c['source_checkpoint']['path'],map_location='cpu',weights_only=True)
    assert state[c['source_frozen_field']]==rt.PARENT_SHA and state['profile']==c['profile']
    assert state['global_update']==3 and state['before']==state['after']==0.25 and state['level_index']==0
    assert state['actor_optimizer']['state'] and state['critic_optimizer']['state'] and state['torch_rng'].numel()>0 and state['cuda_rng']
    policy=Policy('cpu');policy.load_state_dict(state['model_state_dict'],strict=True)
    assert actor_hash(policy)==c['source_actor_hash'] and BUDGET+PRIOR_SHARED_UPDATES==128
    old=rt.source_evidence(c);assert qualification(old['qualification'])
    from boundary_row_audit import check_row
    for native,row in [(True,old['qualification']['native'])]+[(False,r) for r in old['qualification']['batch']]:
        check_row(Path(row['evidence_dir']),row,c,native)
    from boundary_trace_audit import audit
    failed=Path(c['preserved_partial_failure']['folder']);s=rt.read(failed/'native.json')
    proof=audit(failed/'native_traces.npz',s,old['qualification']['native']['mass_kg'],c['profile'],0.225,0.225,'rotor5ms')
    assert proof['status']=='PASS'
    strict=admission(rt.metrics(s),c['anchors_native'],s['max_pre_apex_mimic_v_rad_s'])
    assert strict==c['preserved_partial_failure']['native_strict'] and not strict['passed']
    rt.write(a.output,dict(status='PREFLIGHT_VALIDATED',utc=rt.now(),frozen_sha256=f,contract=c,physical_trials=0,
        source_actor_hash=actor_hash(policy),source_qualified_strength=0.25,source_qualification=True,
        archived_partial_native_audit=proof,archived_partial_native_strict=strict,archived_partial_not_qualified=True,
        first_update_resume_source_optimizer_rng=True,no_same_failed_actor_retest=True,final_gates_unchanged=True))
    print(json.dumps(dict(status='PREFLIGHT_VALIDATED',frozen=f,physical_trials=0,source_actor=actor_hash(policy),partial_native_qualified=False)))
if __name__=='__main__':main()


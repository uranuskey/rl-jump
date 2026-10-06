"""CPU-only verification of the failed source and new bounded learning eligibility."""
import argparse
from pathlib import Path
import target_runtime as rt
from target_contract import qualification,learning_evidence,BUDGET,PRIOR_SHARED_UPDATES
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    f,c=rt.verify(),rt.contract()
    import torch
    from slot_learning import Policy
    from target_worker import actor_hash
    s=torch.load(c['source_checkpoint']['path'],map_location='cpu',weights_only=True)
    assert s[c['source_frozen_field']]==c['source_parent_frozen_sha256'] and s['profile']==c['profile']
    assert s['global_update']==16 and s['before']==s['after']==.275
    policy=Policy('cpu');policy.load_state_dict(s['model_state_dict'],strict=True)
    assert actor_hash(policy)==c['source_actor_hash'] and BUDGET+PRIOR_SHARED_UPDATES==128
    q=rt.source_evidence(c);rows=q['qualification'];evidence=learning_evidence(rows,c)
    assert not qualification(rows) and not q['qualified'] and not q['learning_entry']
    assert evidence['passed'] and evidence['near_mean_target_entry'] and not evidence['original_learning_entry']
    from balanced_row_audit import check_row
    for native,row in [(True,rows['native'])]+[(False,x) for x in rows['batch']]:
        check_row(Path(row['evidence_dir']),row,c,native)
    rt.write(a.output,dict(status='PREFLIGHT_VALIDATED',utc=rt.now(),frozen_sha256=f,contract=c,
        source_actor_hash=actor_hash(policy),physical_trials=0,source_original_qualified=False,
        source_original_learning_entry=False,source_new_learning_evidence=evidence,
        first_training_fresh_optimizer=True,final_qualification_unchanged=True))
    print(dict(status='PREFLIGHT_VALIDATED',frozen=f,source_actor=actor_hash(policy),learning_evidence=evidence),flush=True)
if __name__=='__main__':main()

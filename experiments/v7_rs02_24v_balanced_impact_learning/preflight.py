"""Read-only proof, source actor and objective validation before any new rollout."""
import argparse
from pathlib import Path
import balanced_runtime as rt
from balanced_contract import OBJECTIVE,objective,bulk_penalty,tail_penalty,BUDGET,PRIOR_SHARED_UPDATES,qualification
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists()
    f,c=rt.verify(),rt.contract()
    import torch
    from slot_learning import Policy
    from balanced_worker import actor_hash
    s=torch.load(c['source_checkpoint']['path'],map_location='cpu',weights_only=True)
    assert s[c['source_frozen_field']]==c['source_parent_frozen_sha256'] and s['profile']==c['profile']
    policy=Policy('cpu');policy.load_state_dict(s['model_state_dict'],strict=True)
    assert actor_hash(policy)==c['source_actor_hash']
    assert BUDGET+PRIOR_SHARED_UPDATES==128
    q=rt.source_evidence(c,'source_qualified');bad=rt.source_evidence(c,'source_failed')
    assert qualification(q['qualification']) and not qualification(bad['qualification'])
    from balanced_row_audit import check_row
    for old in (q,bad):
        rows=old['qualification']
        for native,row in [(True,rows['native'])]+[(False,x) for x in rows['batch']]:
            check_row(Path(row['evidence_dir']),row,c,native)
    checks={str(v):dict(bulk=bulk_penalty(v),tail=tail_penalty(v)) for v in [0,280,300,310,330,340,350]}
    rt.write(a.output,dict(status='PREFLIGHT_VALIDATED',utc=rt.now(),frozen_sha256=f,contract=c,
        source_actor_hash=actor_hash(policy),physical_trials=0,source_qualification_reused=True,
        failed_source_qualification_preserved=True,training_objective=OBJECTIVE,penalty_examples=checks))
    print(dict(status='PREFLIGHT_VALIDATED',frozen=f,source_actor=actor_hash(policy),checks=checks),flush=True)
if __name__=='__main__':main()

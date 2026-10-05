"""CPU-only proof of an eligible35% source and unchanged failed32.5% evidence."""
import argparse
from pathlib import Path
import margin_runtime as rt
from margin_contract import impact_only_failure,warmup_request,validate_request

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    frozen=rt.verify();c=rt.contract()
    validate_request(warmup_request(c,frozen))
    q=rt.parent.checked(c['retained_previous_qualification'])
    failed=rt.parent.checked(c['failed_lower_level']['folder'])
    assert impact_only_failure(failed['qualification'])
    import torch
    state=torch.load(c['source_checkpoint']['path'],map_location='cpu',weights_only=True)
    assert state['profile']==c['source_profile']==c['profile']
    assert state['voltage_v']==24 and state['action_dim']==16
    assert state[c['source_frozen_field']]==c['source_parent_frozen_sha256']
    rows=[]
    for r in failed['qualification']['batch']:
        anchors=c['anchors_batch'];cap=1.03*anchors['corrected_reference']['max_force_n']
        rows.append(dict(repeat=r['repeat'],metrics=r['metrics'],learning=r['learning_admission'],
            strict=r['strict_admission'],corrected_reference_peak_cap_n=cap,
            excess_n=r['metrics']['max_force_n']-cap))
    rt.write(a.output,dict(status='PREFLIGHT_VALIDATED',utc=rt.now(),frozen_sha256=frozen,contract=c,
        frontier_qualified=q['qualified'],failed_lower_qualified=failed['qualified'],
        source_actor_hash=q['actor_hash'],old_failure_preserved=True,new_physical_trials=0,
        frontier_rows=[q['qualification']['native']['metrics']]+[r['metrics'] for r in q['qualification']['batch']],
        lower_rows=rows,unchanged_final_and_entry_gates=True))
    print(dict(status='PREFLIGHT_VALIDATED',frozen_sha256=frozen,frontier35_qualified=True,
        lower325_qualified=False,new_physical_trials=0),flush=True)

if __name__=='__main__':main()

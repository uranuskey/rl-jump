"""CPU-only frozen-chain and failed-evidence validation, before any worker starts."""
import argparse
from pathlib import Path
import rare_runtime as rt
from rare_contract import learning_admission,metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    c=rt.contract();f=rt.verify();proof=c['reused_initial_qualification']
    old=rt.parent.checked(proof['folder']);rows=[]
    for native,row in [(True,old['qualification']['native'])]+[(False,x) for x in old['qualification']['batch']]:
        s=rt.read(Path(row['evidence_dir'])/row['summary'])
        assert metrics(s)==rt.parent.metrics(s)==row['metrics']
        anchors=c['anchors_native'] if native else c['anchors_batch']
        entry=learning_admission(s,anchors,row['pre_apex_v_rad_s'])
        assert entry['passed']
        rows.append(dict(summary=row['summary'],metrics=row['metrics'],strict=row['strict_admission'],learning=entry))
    import torch
    ck=torch.load(c['source_checkpoint']['path'],map_location='cpu',weights_only=True)
    assert ck['profile']==c['source_profile'] and ck['voltage_v']==24 and ck['action_dim']==16
    rt.write(a.output,dict(status='PREFLIGHT_VALIDATED',utc=rt.now(),frozen_sha256=f,contract=c,
        old_qualified=old['qualified'],old_learning_entry=old['learning_entry'],new_learning_entry=True,
        unchanged_old_result_sha256=rt.sha(Path(proof['folder'])/'result.json'),
        retained_previous_qualification=c['retained_previous_qualification'],rows=rows,new_physical_trials=0))
    print(dict(status='PREFLIGHT_VALIDATED',frozen_sha256=f,old_failure_preserved=True,new_physical_trials=0),flush=True)

if __name__=='__main__':main()

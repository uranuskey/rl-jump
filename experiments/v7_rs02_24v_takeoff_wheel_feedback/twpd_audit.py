"""A partial native rejection is audited as a diagnostic, never qualification."""
import argparse
from pathlib import Path
import twpd_runtime as rt
from twpd_contract import qualification,learning_entry,learning_evidence,validate_request,native_ready

def check_qualification(folder,request,result):
    from twpd_row_audit import check_row
    assert result['request']==request and result['mode']=='qualify'
    validate_request(request);rows=result['qualification'];c=request['contract']
    assert rows['native']['metrics']['worlds']==45
    ready=native_ready(rows['native'])
    if ready:
        assert len(rows['batch'])==3 and [r['repeat'] for r in rows['batch']]==[1,2,3]
        assert all(r['metrics']['worlds']==512 for r in rows['batch'])
        assert result['new_physical_trials']==1581 and result['evaluation_complete'] is True
    else:
        assert rows['batch']==[] and result['new_physical_trials']==45
        assert result['evaluation_complete'] is False and result['stop_reason']=='NATIVE_REJECTED'
        assert result['qualified'] is False
    for native,row in [(True,rows['native'])]+[(False,x) for x in rows['batch']]:
        check_row(Path(row['evidence_dir']),row,c,native)
        assert row['before']==row['after']==0 and row['sample_kind']=='deterministic_evaluation'
    assert result['qualified']==qualification(rows)
    assert result['learning_entry']==learning_entry(rows,c)
    assert result['learning_evidence']==learning_evidence(rows,c)
    assert result['controller_id']==c['controller_id']
    out=dict(status='AUDITED' if ready else 'DIAGNOSTIC_AUDITED',utc=rt.now(),
        result_sha256=rt.sha(folder/'result.json'),qualified=result['qualified'],
        full_qualification_audited=ready,actual_worlds=result['new_physical_trials'],
        actor_hash=result['actor_hash'],controller_id=result['controller_id'])
    path=folder/'qualification_audit.json'
    if path.exists():
        old=rt.read(path);assert {k:v for k,v in old.items() if k!='utc'}=={k:v for k,v in out.items() if k!='utc'}
    else:rt.write(path,out)
    return out

def audit(run):
    d=rt.checked(run);assert d['contract']==rt.contract()
    assert len(d['events'])==1 and d['completed_updates']==d['actor_steps']==0
    event=d['events'][0];folder=Path(event['folder']);r=rt.checked(folder)
    assert event['result_sha256']==rt.sha(folder/'result.json')
    req=rt.read(folder/'request.json');check_qualification(folder,req,r)
    assert r['actor_hash']==d['contract']['source_actor_hash']
    assert d['zero_assistance_qualified']==r['qualified']
    assert d['stop_reason']==('ZERO_ASSISTANCE_REACHED' if r['qualified'] else 'ZERO_PROBE_FAILED')
    for row in [r['qualification']['native']]+r['qualification']['batch']:
        e=row['external_wrench'];assert e['min_sampled_ticks']>0
        assert e['max_abs_linear_force_n']==e['max_abs_body_torque_nm']==e['max_abs_root_generalized_force']==0
    out=dict(status='AUDITED' if r['evaluation_complete'] else 'DIAGNOSTIC_AUDITED',utc=rt.now(),
        result_sha256=rt.sha(run/'result.json'),frozen_sha256=rt.verify(),completed_updates=0,actor_steps=0,
        updates_in_128_budget=54,shared_remaining=74,completed_evaluation_worlds=r['new_physical_trials'],
        zero_assistance_qualified=r['qualified'],full_qualification_audited=r['evaluation_complete'],
        stop_reason=d['stop_reason'],actual_external_wrench_zero=True,physical_and_final_gates_unchanged=True,hardware_qualified=False)
    rt.write(run/'audit_result.json',out);return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();audit(a.run)
if __name__=='__main__':main()

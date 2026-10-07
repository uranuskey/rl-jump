"""Independent strict zero-assistance qualification replay; zero PPO."""
import argparse
from pathlib import Path
import zsnap_runtime as rt
from zsnap_contract import qualification,learning_entry,learning_evidence,validate_request,reuse_key
def check_qualification(folder,request,result):
    from zsnap_row_audit import check_row
    assert result['request']==request and result['mode']=='qualify'
    validate_request(request);rows=result['qualification'];c=request['contract']
    assert rows['native']['metrics']['worlds']==45
    assert len(rows['batch'])==3 and [r['repeat'] for r in rows['batch']]==[1,2,3]
    if request['reuse_source_evidence']:
        key=reuse_key(request);assert key is not None
        old=rt.source_evidence(c,key)
        assert result['reused_qualification']==c[key] and result['new_physical_trials']==0
        assert rows==old['qualification'] and result['actor_hash']==old['actor_hash']
        assert result['original_source_learning_entry']==old['learning_entry']
        assert result['original_source_qualified']==old['qualified']
        assert (old['request']['before'],old['request']['after'])==(request['before'],request['after'])
    else:
        assert 'reused_qualification' not in result and result['new_physical_trials']==45+3*512
    for native,row in [(True,rows['native'])]+[(False,x) for x in rows['batch']]:
        check_row(Path(row['evidence_dir']),row,c,native)
        assert (row['before'],row['after'])==(request['before'],request['after'])
        assert row['sample_kind']=='deterministic_evaluation'
    assert qualification(rows)==result['qualified'] and learning_entry(rows,c)==result['learning_entry']
    assert learning_evidence(rows,c)==result['learning_evidence']
    out=dict(status='AUDITED',utc=rt.now(),result_sha256=rt.sha(folder/'result.json'),qualified=result['qualified'],
             learning_entry=result['learning_entry'],learning_evidence=result['learning_evidence'],actor_hash=result['actor_hash'])
    path=folder/'qualification_audit.json'
    if path.exists():
        old=rt.read(path);assert {k:v for k,v in old.items() if k!='utc'}=={k:v for k,v in out.items() if k!='utc'}
    else:rt.write(path,out)
    return out
def audit(run):
 d=rt.checked(run);assert d['contract']==rt.contract()
 assert len(d['events'])==1 and d['completed_updates']==0 and d['actor_steps']==0
 event=d['events'][0];folder=Path(event['folder']);r=rt.checked(folder)
 assert event['result_sha256']==rt.sha(folder/'result.json')
 req=rt.read(folder/'request.json');validate_request(req);check_qualification(folder,req,r)
 assert req['checkpoint']==d['contract']['source_checkpoint'] and r['actor_hash']==d['contract']['source_actor_hash']
 assert req['before']==req['after']==0 and r['new_physical_trials']==1581
 assert d['zero_assistance_qualified']==r['qualified']
 assert d['stop_reason']==('ZERO_ASSISTANCE_REACHED' if r['qualified'] else 'ZERO_PROBE_FAILED')
 for row in [r['qualification']['native']]+r['qualification']['batch']:
  e=row['external_wrench'];assert e['min_sampled_ticks']>0
  assert e['max_abs_linear_force_n']==e['max_abs_body_torque_nm']==e['max_abs_root_generalized_force']==0
 out=dict(status='AUDITED',utc=rt.now(),result_sha256=rt.sha(run/'result.json'),frozen_sha256=rt.verify(),completed_updates=0,actor_steps=0,
  updates_in_128_budget=54,shared_remaining=74,completed_evaluation_worlds=1581,zero_assistance_qualified=r['qualified'],stop_reason=d['stop_reason'],
  actual_external_wrench_zero=True,physical_and_final_gates_unchanged=True,hardware_qualified=False)
 rt.write(run/'audit_result.json',out);return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();audit(a.run)
if __name__=='__main__':main()

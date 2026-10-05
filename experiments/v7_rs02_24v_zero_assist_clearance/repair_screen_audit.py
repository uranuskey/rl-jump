"""Recompute candidate selection from all recorded cases and array evidence."""
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
import repair_bootstrap as rt
from repair_profiles import candidates,choose,validate_profile
from repair_contract import admission,wrench_valid

def inspect(folder,c,ck):
    d=rt.read(folder/'result.json');receipt=rt.read(folder/'exit_receipt.json')
    assert receipt['process_exited'] and receipt['exit_code']==0
    assert d['status']=='SCREENED' and d['source_checkpoint']==ck
    assert d['frozen_sha256']==d['final_frozen_sha256']==rt.verify()
    assert d['training_updates']==0 and d['qualification'] is False
    assert d['before']==d['after']==.425 and d['prefix_proof_samples']>0
    assert d['screen_sha256']==rt.sha(folder/'screen.json')
    assert d['poses_sha256']==rt.sha(folder/'poses.npz')
    from audit_fix import check_physics
    check_physics(d['physics_receipt'],'rotor5ms')
    s=rt.read(folder/'screen.json');assert s['sample_kind']=='deterministic_evaluation'
    expected=candidates(c['profile']);assert [r['profile'] for r in d['candidates']]==expected
    assert len(s['cases'])==45*len(expected)
    with np.load(folder/'poses.npz',allow_pickle=False) as z:
        for key in ('q','v','ticks','minimum_q','min_height','slot_error','external_peak','external_ticks'):
            assert len(z[key])==len(s['cases']) and np.isfinite(z[key]).all(),key
        for i,r in enumerate(d['candidates']):
            validate_profile(c['profile'],r['profile']);sl=slice(45*i,45*(i+1));rows=s['cases'][sl]
            m=rt.metrics(dict(cases=rows,reasons=dict(Counter(x['reason'] for x in rows))))
            assert m==r['metrics']
            pre=max(x['constraint_residual_maxima']['pre_apex_v_rad_s'] for x in rows)
            assert pre==r['pre_apex_v_rad_s']
            for key,learning in [('strict_admission',False),('learning_admission',True)]:
                assert admission(m,c['anchors_native'],pre,learning)==r[key]
            assert 1000*float(z['slot_error'][sl].max())==r['max_slot_error_mm']
            assert 1000*float(z['min_height'][sl].min())==r['min_actual_height_mm']
            peak=z['external_peak'][sl].max(0).tolist()
            e=dict(max_abs_linear_force_n=peak[0],max_abs_body_torque_nm=peak[1],
                max_abs_root_generalized_force=peak[2],min_sampled_ticks=int(z['external_ticks'][sl].min()))
            assert e==r['external_wrench']
            assert wrench_valid(dict(before=.425,after=.425,external_wrench=e))==r['external_valid']
    assert d['selected']==choose(d['candidates'])
    return d

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    c,ck=rt.source();d=inspect(a.run,c,ck)
    rt.write(a.run/'screen_audit.json',dict(status='SCREEN_AUDITED',utc=rt.now(),
        result_sha256=rt.sha(a.run/'result.json'),selected=d['selected'],qualification=False))

if __name__=='__main__':main()

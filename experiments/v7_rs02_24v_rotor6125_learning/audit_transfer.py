"""Independently reconstruct admissions, selected history, traces and exits."""
import argparse
import transfer_runtime as rt
from transfer_contract import BEFORE,AFTER,VARIANT,admission,promote,final_qualified


def check_row(folder,row,contract,native=False):
    from audit_fix import check_physics
    from fix_audit import audit
    path=folder/row['summary']
    assert rt.sha(path)==row['summary_sha256']
    s=rt.read(path); m=rt.metrics(s)
    assert m==row['metrics'] and s['max_pre_apex_mimic_v_rad_s']==row['pre_apex_v_rad_s']
    assert s['max_mimic_v_rad_s']==row['max_mimic_v_rad_s']
    assert s['sample_kind']==row['sample_kind']
    check_physics(row['physics_receipt'],VARIANT)
    anchors=contract['anchors_native'] if native else contract['anchors_batch']
    for key,learning in [('strict_admission',False),('learning_admission',True)]:
        assert admission(m,anchors,row['pre_apex_v_rad_s'],learning)==row[key]
    if native:
        path=folder/row['trace']; assert rt.sha(path)==row['trace_sha256']
        assert audit(path,s,row['mass_kg'],contract['profile'],row['before'],row['after'],VARIANT)==row['audits']


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run-id',required=True); args=p.parse_args()
    root=rt.HERE/'runs'; c=rt.contract(); f=rt.verify()
    pre=root/(args.run_id+'_preflight'); smoke=root/(args.run_id+'_smoke')
    train=root/args.run_id; evaluation=root/(args.run_id+'_eval')
    p=rt.checked(pre,'PREFLIGHT_COMPLETED')
    for key in ('native','batch'):
        check_row(pre,p[key],c,key=='native'); assert p[key]['learning_admission']['passed']
        assert p[key]['metrics']['worlds']==(45 if key=='native' else 512)
    results={}
    for folder,status,budget in [(smoke,'SMOKE_COMPLETED',2),(train,'TRAIN_COMPLETED',128)]:
        d=rt.checked(folder,status); results[status]=d
        assert d['completed_updates']==budget and d['update_budget']==budget
        assert d['num_envs']==(45 if budget==2 else 512)
        assert (d['before'],d['after'],d['constraint_variant'])==(BEFORE,AFTER,VARIANT)
        check_row(folder,d['initial'],c); assert d['initial']['learning_admission']['passed']
        steps=0
        for i in range(1,budget+1):
            row=rt.read(folder/f'update_{i:04d}.json')
            assert row['update']==i and row['sample_kind']=='stochastic_training'
            check_row(folder,row['sample'],c)
            assert row['sample']['metrics']['worlds']==d['num_envs']
            steps+=row['stats']['actor_steps']
        assert steps==d['actor_steps'] and steps>0
        selected=None
        for row in d['evaluations']:
            check_row(folder,row,c)
            assert rt.sha(row['checkpoint']['path'])==row['checkpoint']['sha256']
            if promote(row): selected=row
        assert selected==d['selected']
        for key in ('initial_checkpoint','final_checkpoint','latest_checkpoint'):
            assert rt.sha(d[key]['path'])==d[key]['sha256']
    e=rt.checked(evaluation,'EVALUATION_COMPLETED'); trained=results['TRAIN_COMPLETED']
    expected=['reference625','seed','selected' if trained['selected'] else 'candidate','latest']
    assert list(e['models'])==expected
    checkpoints=[c['original_reference']['source_checkpoint'],c['source_checkpoint'],
                 trained['selected']['checkpoint'] if trained['selected'] else trained['final_checkpoint'],trained['final_checkpoint']]
    for name,ck in zip(expected,checkpoints):
        row=e['models'][name]; assert row['checkpoint']==ck and rt.sha(ck['path'])==ck['sha256']
        check_row(evaluation,row['native'],c,True)
        for repeat in row['batch']: check_row(evaluation,repeat,c)
        assert row['comparison_passed']==final_qualified(row,1 if name in ('reference625','seed') else 3)
        history=(trained['selected'] is not None if name=='selected' else trained['latest_evaluation']['strict_admission']['passed'] if name=='latest' else False)
        assert row['history_strict_passed']==history
        assert row['qualified']==(name in ('selected','latest') and history and row['comparison_passed'])
    qualified=any(e['models'].get(n,{}).get('qualified',False) for n in ('selected','latest'))
    assert qualified==e['target_qualified']
    output=dict(status='AUDITED',utc=rt.now(),frozen_sha256=f,
        completed_updates=128,actor_steps=trained['actor_steps'],
        source_checkpoint=c['source_checkpoint'],before=BEFORE,after=AFTER,constraint_variant=VARIANT,
        selected_update=None if trained['selected'] is None else trained['selected']['update'],
        models=e['models'],target_qualified=qualified,
        old_failure_retained=True,training_rebound_limit_mps=.10,final_rebound_limit_mps=1e-6,
        voltage_v=24,estimated_motor_curve=True,zero_assistance_qualified=False,hardware_qualified=False,
        stage_result_sha256={folder.name:rt.sha(folder/'result.json') for folder in (pre,smoke,train,evaluation)})
    rt.write(evaluation/'audit_result.json',output)
    print({k:v for k,v in output.items() if k!='models'},flush=True)


if __name__=='__main__': main()

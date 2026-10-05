import copy, unittest
from rare_contract import (metrics, learning_admission, admission, eligible_contact, learning_entry,
    qualification, decision, chunk_size, validate_reuse, LEVELS)

def case(world=0):
    return dict(world=world,passed=True,reason='success',wheel_cm=9.35,com_cm=11.92,
        success_s=3.,touchdown_s=1.38,end_s=3.,gate_tick=480,
        landing_metrics=dict(peak_force_n=310.,first_stop_com_drop_m=.053,
            peak_rebound_vz_mps=0.,best_continuous_stable_s=1.),
        terminal_diagnostics=dict(leg_height_m=[.094,.095],motor_torque_nm=[13.,-6.,13.,-6.],
            motor_envelope_nm=[16.,17.,16.,17.],reward_apex=True,phase_before=3,self_contact=True,
            nonwheel_force_n=0.,saturated=False,leg_overspeed=[False]*4,wheel_overspeed=[False]*2,
            mimic_q_error_rad=.00001,mimic_v_error_rad_s=.0002))

def fail(row):
    row.update(passed=False,reason='illegal_contact',end_s=1.445,success_s=0.)
    row['landing_metrics'].update(peak_force_n=344.,best_continuous_stable_s=0.)

def summary(n=512,bad=1):
    s={'cases':[case(i) for i in range(n)]}
    for r in s['cases'][:bad]:fail(r)
    return s

ANCHORS={'original':metrics(summary(512,0)),'corrected':metrics(summary(512,0))}

def row(n=512,bad=0,repeat=None):
    s=summary(n,bad);m=metrics(s)
    r=dict(metrics=m,strict_admission=admission(m,ANCHORS,.006),
        learning_admission=learning_admission(s,ANCHORS,.006),before=.425,after=.425,
        external_wrench=dict(min_sampled_ticks=100,max_abs_linear_force_n=0.,
            max_abs_root_generalized_force=0.,max_abs_body_torque_nm=8.5),audits={'status':'PASS'})
    if repeat is not None:r['repeat']=repeat
    return r

def rows(bad=1):return dict(native=row(45,0),batch=[row(512,bad if i==2 else 0,i) for i in range(1,4)])

class RareContact(unittest.TestCase):
    def test_one_contact_is_learnable_but_still_failed(self):
        s=summary();a=learning_admission(s,ANCHORS,.006)
        self.assertTrue(a['passed']);self.assertFalse(a['original']['passed'])
        self.assertFalse(s['cases'][0]['passed']);self.assertEqual(a['failed_world_ids'],[0])
        self.assertEqual(decision(rows(),0),'LEARN');self.assertFalse(qualification(rows()))
    def test_two_contact_limit(self):
        self.assertTrue(learning_admission(summary(bad=2),ANCHORS,.006)['passed'])
        self.assertFalse(learning_admission(summary(bad=3),ANCHORS,.006)['passed'])
    def test_native_failure_is_never_admitted(self):
        self.assertFalse(learning_admission(summary(45,1),ANCHORS,.006)['passed'])
    def test_350N_cap_is_learning_only(self):
        s=summary();s['cases'][0]['landing_metrics']['peak_force_n']=350.
        self.assertTrue(learning_admission(s,ANCHORS,.006)['passed'])
        s['cases'][0]['landing_metrics']['peak_force_n']=350.001
        self.assertFalse(learning_admission(s,ANCHORS,.006)['passed'])
    def test_no_body_ground_contact(self):
        s=summary();s['cases'][0]['terminal_diagnostics']['nonwheel_force_n']=.01
        self.assertFalse(learning_admission(s,ANCHORS,.006)['passed'])
    def test_preapex_failure_and_constraint_rejected(self):
        s=summary();s['cases'][0]['gate_tick']=-1
        self.assertFalse(learning_admission(s,ANCHORS,.006)['passed'])
        self.assertFalse(learning_admission(summary(),ANCHORS,.008001)['passed'])
    def test_only_early_posttouch_compression(self):
        for field,value in [('phase_before',2),('self_contact',False),('reward_apex',False)]:
            s=summary();s['cases'][0]['terminal_diagnostics'][field]=value
            self.assertFalse(learning_admission(s,ANCHORS,.006)['passed'])
        s=summary();s['cases'][0]['end_s']=1.531
        self.assertFalse(learning_admission(s,ANCHORS,.006)['passed'])
    def test_motor_envelope_and_overspeed_unchanged(self):
        for field,value in [('motor_torque_nm',[17.1]*4),('saturated',True),('leg_overspeed',[True]*4),
                ('wheel_overspeed',[True]*2),('mimic_q_error_rad',.002),('mimic_v_error_rad_s',.011)]:
            s=summary();s['cases'][0]['terminal_diagnostics'][field]=value
            self.assertFalse(learning_admission(s,ANCHORS,.006)['passed'])
    def test_nonfinite_or_incomplete_terminal_rejected(self):
        r=summary()['cases'][0];r['terminal_diagnostics']['leg_height_m'][0]=float('nan')
        self.assertFalse(eligible_contact(r));del r['terminal_diagnostics']
        self.assertFalse(eligible_contact(r))
    def test_surviving_unstable_case_is_not_hidden(self):
        s=summary();s['cases'][1]['landing_metrics']['best_continuous_stable_s']=.99
        self.assertFalse(learning_admission(s,ANCHORS,.006)['passed'])
    def test_global_height_stroke_rebound_mean_force_kept(self):
        for key,value,in_landing in [('wheel_cm',9.,False),('com_cm',11.,False),
                ('first_stop_com_drop_m',.049,True),('peak_force_n',320.,True),('peak_rebound_vz_mps',.101,True)]:
            s=summary()
            for r in s['cases']:(r['landing_metrics'] if in_landing else r)[key]=value
            self.assertFalse(learning_admission(s,ANCHORS,.006)['passed'])
    def test_duplicate_world_ids_rejected(self):
        s=summary();s['cases'][1]['world']=0
        self.assertFalse(learning_admission(s,ANCHORS,.006)['passed'])
    def test_strict_qualification_and_all_three_batches_required(self):
        r=rows(0);self.assertTrue(qualification(r));self.assertEqual(decision(r,128),'ADVANCE')
        r['batch'].pop();self.assertFalse(qualification(r));self.assertFalse(learning_entry(r))
        r=rows(0);r['batch'][1]['repeat']=3;self.assertFalse(qualification(r))
    def test_any_external_linear_force_rejects_entry(self):
        r=rows();r['batch'][0]['external_wrench']['max_abs_linear_force_n']=.01
        self.assertFalse(learning_entry(r))
    def test_zero_requires_actual_zero_body_torque(self):
        r=rows(0)
        for x in [r['native']]+r['batch']:x.update(before=0.,after=0.);x['external_wrench']['max_abs_body_torque_nm']=0.
        self.assertTrue(qualification(r))
        r['native']['external_wrench']['max_abs_body_torque_nm']=1e-8
        self.assertFalse(qualification(r))
    def test_no_advance_on_small_rebound(self):
        s=summary(512,0);s['cases'][0]['landing_metrics']['peak_rebound_vz_mps']=.05
        self.assertTrue(learning_admission(s,ANCHORS,.006)['passed'])
        self.assertFalse(admission(metrics(s),ANCHORS,.006)['passed'])
    def test_budget_and_chunks(self):
        self.assertEqual(chunk_size(0,0),2);self.assertEqual(chunk_size(2,2),8)
        self.assertEqual(chunk_size(126,10),2);self.assertEqual(decision(rows(),128),'BUDGET_EXHAUSTED')
    def test_reuse_cannot_repeat_same_actor_or_other_level(self):
        source=dict(path='retained.pt',sha256='source')
        q=dict(mode='qualify',reuse_preflight=True,level_index=0,global_updates=0,before=.425,after=.425,
            checkpoint=source,contract=dict(source_checkpoint=source,reused_initial_qualification=dict(
                folder='old_failed',result_sha256='old',old_qualified=False,old_learning_entry=False)))
        validate_reuse(q)
        for key,value in [('mode','train'),('level_index',1),('global_updates',2),('before',.40),
                ('checkpoint',dict(path='changed.pt',sha256='changed')),('reuse_preflight',False)]:
            bad=copy.deepcopy(q);bad[key]=value
            with self.assertRaises(AssertionError):validate_reuse(bad)
    def test_declared_course_reaches_zero(self):
        self.assertEqual(LEVELS[0],('uniform425',.425,.425))
        self.assertEqual(LEVELS[-1],('uniform000',0.,0.));self.assertEqual(len(LEVELS),18)

if __name__=='__main__':unittest.main()

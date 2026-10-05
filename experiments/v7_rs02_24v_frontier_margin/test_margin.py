import copy,unittest
import margin_contract as m
import rare_contract as parent
from test_rare import rows,ANCHORS,summary

class FrontierMargin(unittest.TestCase):
    def test_both_admission_and_qualification_are_the_parent_functions(self):
        for name in ('admission','qualification','learning_admission','learning_entry'):
            self.assertIs(getattr(m,name),getattr(parent,name))
    def test_over_limit_successful_batch_still_cannot_learn_at_that_level(self):
        s=summary(512,0)
        s['cases'][0]['landing_metrics']['peak_force_n']=342.
        self.assertFalse(m.learning_admission(s,ANCHORS,.006)['passed'])
        self.assertFalse(m.admission(m.metrics(s),ANCHORS,.006)['passed'])
    def test_only_impact_failure_triggers_this_repair_hypothesis(self):
        r=rows(0)
        checks=r['batch'][0]['strict_admission']
        checks['passed']=False
        for c in checks['comparisons'].values():
            c['passed']=False;c['checks']['worst_impact']=False;c['reasons']=['worst_impact']
        self.assertTrue(m.impact_only_failure(r))
        checks['comparisons']['original']['checks']['wheel_height']=False
        self.assertFalse(m.impact_only_failure(r))
    def test_contact_or_native_audit_failure_does_not_match_impact_only(self):
        self.assertFalse(m.impact_only_failure(rows(1)))
        r=rows(0);r['native']['audits']['status']='FAIL'
        self.assertFalse(m.impact_only_failure(r))
    def test_incomplete_batch_evidence_rejected(self):
        r=rows(0);r['batch'].pop();self.assertFalse(m.impact_only_failure(r))
    def test_warmup_is_exactly_two_updates_on_qualified35(self):
        c={'source_checkpoint':{'path':'qualified35.pt','sha256':'source'}}
        r=m.warmup_request(c,'frozen');self.assertTrue(m.validate_request(r))
        self.assertEqual((r['before'],r['after'],r['updates'],r['resume_optimizer']),(.35,.35,2,False))
        for key,value in [('before',.325),('after',.325),('updates',8),('resume_optimizer',True),
                ('global_updates',1),('level_index',0),('mode','qualify')]:
            bad=copy.deepcopy(r);bad[key]=value
            with self.assertRaises(AssertionError):m.validate_request(bad)
    def test_old_failed_actor_cannot_be_evaluated_before_warmup(self):
        r=dict(mode='qualify',level_index=0,before=.325,after=.325,global_updates=0,reuse_preflight=False)
        with self.assertRaises(AssertionError):m.validate_request(r)
        r['global_updates']=2;self.assertTrue(m.validate_request(r))
        r['reuse_preflight']=True
        with self.assertRaises(AssertionError):m.validate_request(r)
    def test_no_undeclared_assistance_level(self):
        r=dict(mode='qualify',level_index=0,before=.35,after=.35,global_updates=2,reuse_preflight=False)
        with self.assertRaises(AssertionError):m.validate_request(r)
    def test_remaining_budget_includes_warmup(self):
        self.assertEqual(m.BUDGET+m.PRIOR_COURSE_UPDATES,128)
        self.assertEqual(m.chunk_size(2,0),2)
        self.assertEqual(m.chunk_size(4,2),8)
        self.assertEqual(m.chunk_size(124,8),2)
    def test_qualification_still_advances_immediately(self):
        self.assertEqual(m.decision(rows(0),2),'ADVANCE')
        self.assertEqual(m.decision(rows(0),m.BUDGET),'ADVANCE')
    def test_course_ends_only_at_exact_zero(self):
        self.assertEqual(m.LEVELS[0],('uniform325',.325,.325))
        self.assertEqual(m.LEVELS[-1],('uniform000',0.,0.))
        self.assertEqual(len(m.LEVELS),14)

if __name__=='__main__':unittest.main()

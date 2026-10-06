import ast,math,unittest
from pathlib import Path
import filter_contract as c
import tail_contract as parent

def event(level,candidate,qualified):
    return dict(level_index=level,candidate_index=candidate,qualified=qualified)

class ContractTests(unittest.TestCase):
    def test_fixed_finite_fractions(self):
        self.assertEqual(c.FRACTIONS,(.75,.5,.25));self.assertEqual(len(set(c.FRACTIONS)),3)
    def test_gate_function_identity(self):
        for name in ('admission','learning_admission','qualification','learning_entry','metrics','wrench_valid','zero_qualified'):
            self.assertIs(getattr(c,name),getattr(parent,name))
    def test_zero_new_ppo(self):
        self.assertEqual(c.NEW_PPO_UPDATES,0);self.assertEqual(c.PRIOR_SHARED_UPDATES,14)
    def test_synchronous_levels(self):
        self.assertEqual([x[1] for x in c.LEVELS],[i/1000 for i in range(250,-1,-25)])
        self.assertTrue(all(b==a for _,b,a in c.LEVELS))
    def test_interior_linear_step(self):
        self.assertEqual(c.blend_parameter(4.,12.,.75),10.)
        self.assertEqual(c.blend_parameter(12.,4.,.25),10.)
    def test_no_endpoint_or_invalid_fraction(self):
        for value in (0.,1.,-1.,2.,math.nan,math.inf):
            with self.assertRaises(AssertionError):c.blend_parameter(1.,2.,value)
    def test_initial_request(self):
        x=c.replay([]);self.assertEqual((x['level_index'],x['candidate_index']),(0,0))
    def test_failure_advances_distinct_candidate(self):
        x=c.replay([event(0,0,False)]);self.assertEqual((x['level_index'],x['candidate_index']),(0,1))
    def test_success_immediately_descends(self):
        x=c.replay([event(0,0,False),event(0,1,True)])
        self.assertEqual((x['level_index'],x['candidate_index'],x['selected']),(1,1,1))
    def test_three_failures_terminal(self):
        x=c.replay([event(0,i,False) for i in range(3)])
        self.assertEqual(x['stop_reason'],'NO_FEASIBLE_STEP');self.assertEqual(x['promotions'],[])
    def test_continuation_failure_preserves_qualified(self):
        x=c.replay([event(0,0,True),event(1,0,False)])
        self.assertEqual(x['stop_reason'],'TARGET_FAILED');self.assertEqual(x['promotions'],[(0,0)])
    def test_no_duplicate_failed_actor_test(self):
        with self.assertRaises(AssertionError):c.replay([event(0,0,False),event(0,0,True)])
    def test_no_skipped_or_changed_level(self):
        for events in ([event(0,1,True)],[event(0,0,True),event(2,0,True)],[event(0,0,True),event(1,1,True)]):
            with self.assertRaises(AssertionError):c.replay(events)
    def test_zero_requires_every_level(self):
        events=[event(i,0,True) for i in range(len(c.LEVELS))]
        self.assertEqual(c.replay(events)['stop_reason'],'ZERO_ASSISTANCE_REACHED')
        self.assertIsNone(c.replay(events[:-1])['stop_reason'])
    def test_no_events_after_terminal(self):
        with self.assertRaises(AssertionError):c.replay([event(0,i,False) for i in range(3)]+[event(0,0,True)])
    def test_request_validation(self):
        ck=dict(path='candidate.pt',sha256='digest',global_update=0)
        q=dict(mode='qualify',role='filter',level_index=0,before=.25,after=.25,global_updates=0,reuse_preflight=False,
               candidate=dict(index=0,fraction=.75,checkpoint=ck,actor_hash='new'),checkpoint=ck)
        self.assertTrue(c.validate_request(q))
        for key,value in (('mode','train'),('before',.275),('global_updates',1),('reuse_preflight',True),('role','withdraw')):
            with self.assertRaises(AssertionError):c.validate_request(dict(q,**{key:value}))
    def test_inherited_physical_sampling_unchanged(self):
        here=Path(__file__).resolve().parent
        def function(path,name):
            return ast.dump(next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
        old=here.parent/'v7_rs02_24v_tail_impact_learning/tail_worker.py'
        for name in ('sample','qualify','actor_hash'):
            self.assertEqual(function(here/'filter_worker.py',name),function(old,name))
    def test_candidate_cpu_checks_do_not_shift_rollout_rng(self):
        text=(Path(__file__).resolve().parent/'filter_worker.py').read_text(encoding='utf-8')
        self.assertLess(text.index("state=verify_candidate(request['candidate'],c)"),text.index('torch.manual_seed(105070)'))

if __name__=='__main__':unittest.main()

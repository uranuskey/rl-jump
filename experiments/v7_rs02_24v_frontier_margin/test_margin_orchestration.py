"""Exercise warmup-to-qualification sequencing without launching any simulator."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import importlib.util,tempfile,unittest,contextlib,io
import margin_contract as contract
from test_rare import rows

class Sequence(unittest.TestCase):
    def exercise(self,unchanged=False,first_target_learns=False):
        with tempfile.TemporaryDirectory() as temp:
            runtime=SimpleNamespace(HERE=Path(temp),write=lambda *args:None,sha=lambda p:str(p))
            audit=SimpleNamespace(check_qualification=lambda *args:None)
            spec=importlib.util.spec_from_file_location('margin_sequence_test',Path(__file__).with_name('margin_course.py'))
            module=importlib.util.module_from_spec(spec)
            with patch.dict('sys.modules',{'margin_runtime':runtime,'margin_audit':audit}):spec.loader.exec_module(module)
            calls=[];state={'actor':'source','failed_once':False}
            def worker(run,req,result):
                contract.validate_request(req);calls.append(req.copy())
                folder=Path(temp)/str(len(calls))
                if req['mode']=='train':
                    old=state['actor'];state['actor']=old if unchanged else 'actor'+str(len(calls))
                    return folder,dict(actor_hash_before=old,actor_hash_after=state['actor'],actor_steps=req['updates']*8,
                        completed_updates=req['updates'],checkpoint={'path':'model'+str(len(calls)),
                            'sha256':'new','global_update':req['global_updates']+req['updates']})
                fail=first_target_learns and not state['failed_once'];state['failed_once']=True
                q=rows(1 if fail else 0)
                for row in [q['native']]+q['batch']:
                    row.update(before=req['before'],after=req['after'])
                    row['external_wrench']['max_abs_body_torque_nm']=20*req['before']
                return folder,dict(actor_hash=state['actor'],qualification=q)
            module.worker=worker
            result=dict(contract={'source_checkpoint':{'path':'old','sha256':'old'},'source_actor_hash':'source'},frozen_sha256='f')
            with contextlib.redirect_stdout(io.StringIO()):module.execute(Path(temp),result)
            return calls,result
    def test_warmup_precedes_every_lower_qualification(self):
        calls,result=self.exercise()
        self.assertEqual(calls[0]['before'],.35);self.assertTrue(calls[0]['warmup'])
        self.assertEqual(calls[1]['before'],.325);self.assertEqual(calls[1]['global_updates'],2)
        self.assertEqual(len(calls),15)
        self.assertEqual(result['completed_updates'],2);self.assertEqual(result['actor_steps'],16)
        self.assertEqual(result['deepest_qualified'],'uniform000');self.assertTrue(result['zero_assistance_qualified'])
    def test_unchanged_actor_cannot_retest_failed_level(self):
        with self.assertRaises(AssertionError):self.exercise(unchanged=True)
    def test_target_learning_starts_fresh_optimizer_after_warmup(self):
        calls,result=self.exercise(first_target_learns=True)
        train=calls[2]
        self.assertEqual(train['mode'],'train');self.assertEqual(train['before'],.325)
        self.assertFalse(train['resume_optimizer']);self.assertEqual(train['updates'],2)
        self.assertEqual(result['completed_updates'],4);self.assertEqual(result['actor_steps'],32)

if __name__=='__main__':unittest.main()

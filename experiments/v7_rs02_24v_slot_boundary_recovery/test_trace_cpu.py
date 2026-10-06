"""CPU replay and deliberate-corruption rejection on archived trace data."""
import importlib.util,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
HAS_TORCH=importlib.util.find_spec('torch') is not None
@unittest.skipUnless(HAS_TORCH,'Remote dependency runtime required; no simulation is run')
class PreservedTraceTests(unittest.TestCase):
    def test_archived_trace_and_tamper_rejection(self):
        import numpy as np
        from unittest.mock import patch
        import boundary_runtime as rt
        import boundary_trace_audit as fixed
        path=rt.PARENT/'runs/course_01/level_01_u0003_target/native_traces.npz'
        d=rt.read(rt.PARENT/'runs/course_01/result.json');profile=d['contract']['profile']
        self.assertEqual(rt.sha(path),rt.FAILED_TRACE_SHA)
        self.assertEqual(fixed.slot_audit(path,profile)['status'],'PASS')
        original_load=np.load
        for name,delta in [('requested_payload',.01),('sensor_slot_motor_increment_nm',.01),('sensor_slot_allocated_force_n',.1)]:
            with original_load(path) as z:
                wrong=z[name].copy()
                if name=='requested_payload':wrong[633,16,3]+=delta
                elif name=='sensor_slot_motor_increment_nm':wrong[633,16,1,1]+=delta
                else:wrong[633,16,1]+=delta
                class Proxy:
                    def __enter__(self):return self
                    def __exit__(self,*a):return False
                    def __getitem__(self,key):return wrong if key==name else z[key]
                with patch.object(fixed.np,'load',lambda *a,**kw:Proxy()):
                    with self.assertRaises(AssertionError,msg=name):fixed.slot_audit(path,profile)
if __name__=='__main__':unittest.main()


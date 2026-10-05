"""Exercise the independent audit against an existing real 400 Hz trajectory."""
import tempfile,unittest
from pathlib import Path
import allocation_paths
import numpy as np
from allocation_trace_audit import slot_audit
class Audit(unittest.TestCase):
    def test_real_trace_and_tamper_rejection(self):
        old=allocation_paths.PARENT/'runs/course_01/level_00_u0000_qualify'
        if not (old/'native_traces.npz').exists():self.skipTest('Remote evidence required')
        p=allocation_paths.read(old/'request.json')['contract']['profile'].copy()
        p['slot_action_bound']=1.
        with np.load(old/'native_traces.npz') as z:a={k:z[k] for k in z.files}
        a['sensor_slot_action_bound']=np.ones_like(a['sensor_slot_kp'])
        base=allocation_paths.HERE/'runs/audit_unit'
        base.mkdir(parents=True,exist_ok=True)
        fixture=base/'legacy_fixture.npz'
        np.savez_compressed(fixture,**a)
        result=slot_audit(fixture,p);self.assertEqual(result['status'],'PASS')
        live=a['active'] & a['landing_control_enabled']
        t,w=np.argwhere(live)[20]
        a['requested_payload'][t,w,0]+=.05
        bad=base/'tampered_fixture.npz';np.savez_compressed(bad,**a)
        with self.assertRaises(AssertionError):slot_audit(bad,p)
if __name__=='__main__':unittest.main()

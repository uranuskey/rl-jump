"""Pure CPU controller boundaries and unchanged qualification gate regressions."""
import ast
import copy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'v7_rs02_24v_zero_assist_snapshot'
MID=HERE.parent/'v7_rs02_24v_midpoint_curriculum'
# Contract modules are CPU-only; do not import the physics/runtime bootstrap.
sys.path.insert(0,str(HERE))
sys.path.extend(str(p) for p in HERE.parent.iterdir() if p.is_dir() and str(p) not in sys.path)
from twpd_control import wheel_feedback,PROFILE
import twpd_gates as gates
from twpd_contract import native_ready


def node(path,name):
    return next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                if isinstance(n,ast.FunctionDef) and n.name==name)


def source_rows():
    for path in (MID/'runs/course_01/level_00_u0001_frontier_recovery_check/result.json',
                 MID/'runs/parent_review_01/u1_1416_bundle/level_00_u0001_frontier_recovery_check/result.json'):
        if path.is_file():
            return json.loads(path.read_text(encoding='utf-8-sig'))['qualification']
    raise FileNotFoundError('Preserved midpoint u1 result required')


class WheelFeedback(unittest.TestCase):
    def call(self,pitch,gyro,ticks,apex=None,terminal=None):
        pitch=np.asarray(pitch,dtype=np.float64);gyro=np.asarray(gyro,dtype=np.float64)
        ticks=np.asarray(ticks,dtype=np.float64)
        apex=np.zeros(pitch.shape,dtype=bool) if apex is None else np.asarray(apex,dtype=bool)
        terminal=np.zeros(pitch.shape,dtype=bool) if terminal is None else np.asarray(terminal,dtype=bool)
        return wheel_feedback(pitch,gyro,ticks,apex,terminal,np)

    def test_handoff_and_ramp_boundaries(self):
        action,enabled,ramp=self.call([1]*5,[0]*5,[239,240,250,260,1000])
        np.testing.assert_array_equal(enabled,[False,True,True,True,True])
        np.testing.assert_allclose(ramp,[0,0,.5,1,1],atol=1e-14,rtol=0)
        np.testing.assert_allclose(action,[0,0,.5,1,1],atol=1e-14,rtol=0)

    def test_pitch_gyro_sign_and_action_saturation(self):
        action,_,_=self.call([.1,-.1,0,0,100,-100],[0,0,1,-1,0,0],[260]*6)
        np.testing.assert_allclose(action,[.7,-.7,.5,-.5,1,-1],atol=1e-14,rtol=0)

    def test_apex_and_terminal_gate_each_world(self):
        action,enabled,_=self.call([.1]*4,[0]*4,[260]*4,
            apex=[False,True,False,True],terminal=[False,False,True,True])
        np.testing.assert_array_equal(enabled,[True,False,False,False])
        np.testing.assert_allclose(action,[.7,0,0,0],atol=1e-14,rtol=0)

    def test_nonfinite_input_rejected_even_when_disabled(self):
        for field in range(3):
            for invalid in (np.nan,np.inf,-np.inf):
                args=[[.1],[0.],[0.]];args[field]=[invalid]
                with self.subTest(field=field,invalid=invalid),self.assertRaises(AssertionError):
                    self.call(*args,apex=[True],terminal=[True])

    def test_outputs_finite_and_bounded(self):
        rng=np.random.default_rng(149)
        action,enabled,ramp=self.call(rng.uniform(-2,2,4096),rng.uniform(-20,20,4096),rng.integers(0,2001,4096))
        self.assertTrue(np.isfinite(action).all() and np.isfinite(ramp).all())
        self.assertTrue(np.all(abs(action)<=PROFILE['action_bound']))
        self.assertTrue(np.all((ramp>=0)&(ramp<=1)))
        self.assertTrue(np.all(action[~enabled]==0))

    def test_torch_cpu_matches_numpy_when_available(self):
        try: torch=importlib.import_module('torch')
        except ImportError: self.skipTest('Torch unavailable; NumPy CPU controller tests still required')
        pitch=np.array([.1,-.1,0,.1,.1,.1],dtype=np.float64)
        gyro=np.array([0,0,-1,0,0,0],dtype=np.float64)
        ticks=np.array([239,240,250,260,260,260],dtype=np.float64)
        apex=np.array([False,False,False,False,True,False])
        terminal=np.array([False,False,False,False,False,True])
        expected=wheel_feedback(pitch,gyro,ticks,apex,terminal,np)
        actual=wheel_feedback(*(torch.from_numpy(x) for x in (pitch,gyro,ticks,apex,terminal)),torch)
        for a,b in zip(expected,actual):
            np.testing.assert_allclose(a,b.numpy(),atol=1e-14,rtol=0)


class NativeAndQualificationGates(unittest.TestCase):
    def test_entire_old_gate_module_unchanged(self):
        self.assertEqual(ast.dump(ast.parse((OLD/'zsnap_gates.py').read_text(encoding='utf-8'))),
                         ast.dump(ast.parse((HERE/'twpd_gates.py').read_text(encoding='utf-8'))))

    def ready_row(self):
        # Synthetic counterfactual for gate logic only, not measured qualification.
        row=copy.deepcopy(source_rows()['native'])
        row.update(before=0.,after=0.)
        row['metrics'].update(worlds=45,passed=45)
        row['strict_admission']['passed']=True;row['audits']['status']='PASS'
        row['external_wrench'].update(max_abs_linear_force_n=0.,max_abs_body_torque_nm=0.,
            max_abs_root_generalized_force=0.,min_sampled_ticks=1)
        return row

    def test_native_ready_requires_every_strict_condition(self):
        row=self.ready_row();self.assertTrue(native_ready(row))
        for group,key,value in (('metrics','worlds',44),('metrics','passed',44),
            ('strict_admission','passed',False),('audits','status','UNQUALIFIED'),
            ('external_wrench','max_abs_linear_force_n',1e-12),
            ('external_wrench','max_abs_body_torque_nm',1e-12),
            ('external_wrench','max_abs_root_generalized_force',1e-12),
            ('external_wrench','min_sampled_ticks',0)):
            bad=copy.deepcopy(row);bad[group][key]=value
            with self.subTest(group=group,key=key):self.assertFalse(native_ready(bad))

    def test_missing_batches_cannot_qualify_under_real_gates(self):
        rows=copy.deepcopy(source_rows());rows['native']=self.ready_row()
        for row in rows['batch']:
            row.update(before=0.,after=0.);row['strict_admission']['passed']=True
            row['external_wrench'].update(max_abs_linear_force_n=0.,max_abs_body_torque_nm=0.,
                max_abs_root_generalized_force=0.,min_sampled_ticks=1)
        self.assertTrue(gates.qualification(rows))
        for count in (0,1,2):
            truncated=copy.deepcopy(rows);truncated['batch']=truncated['batch'][:count]
            with self.subTest(batch_count=count):self.assertFalse(gates.qualification(truncated))

    def test_native_failure_never_builds_or_samples_batch(self):
        failed=self.ready_row();failed['strict_admission']['passed']=False
        builds=[];samples=[]
        def build(request,out,n,native=False):
            builds.append((n,native));return ([None]*4,None)
        def sample(parts,request,out,name,limit,native=False):
            samples.append((name,native));return (None,failed)
        namespace=dict(rt=SimpleNamespace(write=lambda *a:None),build=build,sample=sample,
            actor_hash=lambda actor:'cpu_test_hash',native_ready=native_ready,
            qualification=gates.qualification,learning_entry=gates.learning_entry,
            learning_evidence=gates.learning_evidence)
        function=node(HERE/'twpd_worker.py','qualify')
        exec(compile(ast.Module(body=[function],type_ignores=[]),'qualify','exec'),namespace)
        result={}
        namespace['qualify'](dict(reuse_source_evidence=False,reuse_preflight=False,
            contract={'native_first_stop_if_not_strict':True,'controller_id':'cpu_test_controller',
                      'anchors_batch':{},'anchors_native':{}}),HERE/'runs/cpu_gate_mock',result,lambda:None)
        self.assertEqual(builds,[(45,True)])
        self.assertEqual(samples,[('native',True)])
        self.assertFalse(result['qualified'])
        self.assertEqual(result['qualification']['batch'],[])
        self.assertEqual(result['new_physical_trials'],45)
        self.assertFalse(result['evaluation_complete'])


if __name__=='__main__': unittest.main(verbosity=2)

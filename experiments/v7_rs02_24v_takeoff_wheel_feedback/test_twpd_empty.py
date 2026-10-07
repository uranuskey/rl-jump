"""CPU-only regression on real archived traces; no Torch/runtime imports."""
import ast
import copy
import json
import os
from pathlib import Path
import tempfile
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'v7_rs02_24v_zero_assist_snapshot'
REMOTE_TRACE = OLD/'runs/course_01/level_00_u0000_zero_probe/native_traces.npz'
LOCAL_TRACE = OLD/'runs/parent_review_01/zero_native_1445_full_bundle/level_00_u0000_zero_probe/native_traces.npz'
DEFAULT_TRACE = REMOTE_TRACE if REMOTE_TRACE.is_file() else LOCAL_TRACE


def extract(path, name, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[name]


class EmptySlotAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trace = Path(os.environ.get('TWPD_EMPTY_TRACE', str(DEFAULT_TRACE)))
        if not cls.trace.is_file():
            raise FileNotFoundError('Real trace fixture missing; set TWPD_EMPTY_TRACE: '+str(cls.trace))
        cls.namespace = {'np':np, 'json':json, 'Path':Path, '__file__':str(HERE/'twpd_trace_audit.py')}
        cls.audit = staticmethod(extract(HERE/'twpd_trace_audit.py', 'slot_audit', cls.namespace))
        with np.load(cls.trace, allow_pickle=False) as archive:
            cls.arrays = {key:archive[key].copy() for key in (
                'active', 'landing_control_enabled', 'sensor_slot_enabled',
                'sensor_slot_motor_increment_nm', 'sensor_slot_allocated_force_n')}
        assert np.any(cls.arrays['active']), 'Fixture must contain actual active physics samples'
        assert not np.any(cls.arrays['landing_control_enabled']), 'Fixture must demonstrate landing not reached'

    def corrupted_audit(self, key, value, index=None):
        arrays = {name:array.copy() for name,array in self.arrays.items()}
        if index is None:
            arrays[key].flat[0] = value
        else:
            arrays[key][index] = value
        scratch = HERE/'runs/empty_cpu_tests'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='twpd_empty_', dir=scratch) as folder:
            path = Path(folder)/'corrupted_trace.npz'
            np.savez_compressed(path, **arrays)
            return self.audit(path, {})

    def test_real_complete_trace_is_not_reached_and_not_qualified(self):
        result = self.audit(self.trace, {})
        self.assertEqual(result['status'], 'NOT_REACHED')
        self.assertFalse(result['qualified'])
        self.assertEqual(result['physical_samples'], 0)
        self.assertFalse(result['numpy_feedback_recomputed'])
        self.assertFalse(result['encoder_causality_verified'])

    def test_nonzero_motor_increment_rejected(self):
        with self.assertRaisesRegex(AssertionError, 'Nonzero slot motor increment'):
            self.corrupted_audit('sensor_slot_motor_increment_nm', np.float32(1e-8))

    def test_nonzero_allocated_force_rejected(self):
        with self.assertRaisesRegex(AssertionError, 'Nonzero allocated slot force'):
            self.corrupted_audit('sensor_slot_allocated_force_n', np.float32(1e-8))

    def test_slot_enabled_without_landing_rejected(self):
        with self.assertRaisesRegex(AssertionError, 'Slot enabled without live'):
            self.corrupted_audit('sensor_slot_enabled', True)

    def test_forged_landing_on_inactive_row_rejected(self):
        # active & landing stays empty; a plain empty-mask return would hide this.
        inactive = tuple(np.argwhere(~self.arrays['active'])[0])
        with self.assertRaisesRegex(AssertionError, 'Landing enabled outside live'):
            self.corrupted_audit('landing_control_enabled', True, inactive)

    def test_nonempty_formula_and_tolerances_are_unchanged(self):
        old = ast.parse((OLD/'zsnap_trace_audit.py').read_text(encoding='utf-8'))
        new = ast.parse((HERE/'twpd_trace_audit.py').read_text(encoding='utf-8'))
        old_function = next(node for node in old.body if isinstance(node, ast.FunctionDef) and node.name == 'slot_audit')
        new_function = copy.deepcopy(next(node for node in new.body if isinstance(node, ast.FunctionDef) and node.name == 'slot_audit'))
        self.assertIsInstance(new_function.body[0].body[1], ast.If)
        del new_function.body[0].body[1]
        self.assertEqual(ast.dump(old_function, include_attributes=False), ast.dump(new_function, include_attributes=False))

    def test_forged_all_pass_summary_cannot_qualify_empty_landing(self):
        namespace = dict(self.namespace)
        namespace['slot_audit'] = self.audit
        namespace['audit_arrays'] = lambda *args: {}
        with np.load(self.trace, allow_pickle=False) as archive:
            variant = {'rotor_timeconst':archive['sensor_rotor_eq_timeconst_s'].flat[0],
                       'tolerance':archive['sensor_solver_tolerance'].flat[0]}
        namespace['checked_variant'] = lambda name: variant
        audit = extract(HERE/'twpd_trace_audit.py', 'inherited_audit', namespace)
        def forbidden(*args, **kwargs):
            raise AssertionError('Physical qualification must not run for NOT_REACHED')
        physical = ModuleType('audit_training'); physical.audit_trace = forbidden
        controller = ModuleType('slot_probe')
        controller.load_controller_audit = controller.schedule_audit = forbidden
        with patch.dict(sys.modules, {'audit_training':physical, 'slot_probe':controller}):
            result = audit(self.trace, {'passed':45, 'worlds':45}, 7.228699, {}, 0., 0., 'test')
        self.assertEqual(result['status'], 'UNQUALIFIED')
        self.assertEqual(result['slot']['status'], 'NOT_REACHED')


if __name__ == '__main__':
    unittest.main(verbosity=2)

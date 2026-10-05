"""A lower takeoff target, bounded learning entry, unchanged final gates."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'v7_rs02_24v_apex_rebound_learning'))
from learning_contract import learning_admission, qualification

BEFORE, AFTER, VARIANT = .6125, .60, 'rotor5ms'


def admission(measured, anchors, pre_apex_v, learning=False):
    function = learning_admission if learning else qualification
    checks = {name:function(measured, anchor) for name, anchor in anchors.items()}
    margin = 0 <= pre_apex_v <= .008
    return dict(passed=margin and all(c['passed'] for c in checks.values()),
                pre_apex_constraint_margin=margin, comparisons=checks,
                learning_rebound_limit_mps=.10 if learning else 1e-6)


def promote(entry):
    # Latest strict-passing policy at this lower assistance, not lowest impact.
    return entry['update'] > 0 and entry['strict_admission']['passed']


def final_qualified(row, repeats=3):
    return (row['native']['metrics']['worlds'] == 45
            and row['native']['audits']['status'] == 'PASS'
            and row['native']['strict_admission']['passed']
            and len(row['batch']) == repeats
            and [r['repeat'] for r in row['batch']] == list(range(1, repeats+1))
            and all(r['metrics']['worlds'] == 512 and r['strict_admission']['passed'] for r in row['batch']))

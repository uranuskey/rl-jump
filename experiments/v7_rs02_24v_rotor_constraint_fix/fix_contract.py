"""Separate solve convergence from the compliance of the ideal rotor coupling."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'v7_rs02_24v_takeoff_withdrawal_probe'))
from probe_contract import admission, qualified as original_qualified

VARIANTS = {
    'original': dict(iterations=100, tolerance=1e-8, ls_iterations=50, ls_tolerance=.01, rotor_timeconst=.020),
    'solve_strict': dict(iterations=200, tolerance=1e-10, ls_iterations=100, ls_tolerance=.0001, rotor_timeconst=.020),
    'rotor10ms': dict(iterations=100, tolerance=1e-8, ls_iterations=50, ls_tolerance=.01, rotor_timeconst=.010),
    'rotor5ms': dict(iterations=100, tolerance=1e-8, ls_iterations=50, ls_tolerance=.01, rotor_timeconst=.005),
}
LEVELS = [('current625_600', .625, .60), ('takeoff6125_600', .6125, .60),
          ('uniform600', .60, .60), ('uniform575', .575, .575),
          ('uniform550', .55, .55), ('uniform525', .525, .525), ('uniform500', .50, .50)]


def checked_variant(name):
    v = VARIANTS[name]
    assert v['rotor_timeconst'] >= 2*.0025
    assert v['iterations'] >= 100 and v['tolerance'] <= 1e-8
    return v.copy()


def checked_levels(before, after):
    assert (before, after) in [(a, b) for _, a, b in LEVELS] + [(.625, .625)]
    return before, after


def prefix_qualified(rows):
    return (len(rows) == 3 and [r['repeat'] for r in rows] == [1, 2, 3]
        and all(r['worlds'] == 512 and r['failed'] == 0 and r['max_mimic_v_rad_s'] <= .008
                and r['max_mimic_q_rad'] <= .001 and r['prefix_s'] >= 1.20 for r in rows))


def qualified(row, repeats=3):
    old = row['native'].get('old_physics_comparison')
    return (original_qualified(row, repeats)
        and row['native']['old_reference_admission']['passed']
        and (old is None or old['passed'])
        and all(r['old_reference_admission']['passed'] for r in row['batch_replays']))


def deepest_qualified(models):
    chosen = None
    for name, _, _ in LEVELS:
        if name not in models or not qualified(models[name]):
            break
        chosen = name
    return chosen

"""Small, qualified assistance steps; unchanged height and physical gates."""
import math

SOURCE_LEVEL = .625
PROBE_LEVELS = (.60, .6125)
SOURCE_SHA = '55e497c46a32b7116f96b9f563afff5e7f9cc0268ce8937ecdef275c0f82c6bf'
SOURCE_AUDIT_SHA = '2429fc0eb74ed8eed0aa0d5d6292042b2dd7d49bcae6bc2ebbb05189edd00bb3'
PARENT_TRAINING_SHA = '07cc43cdc6acfbf55d26acfd579f4a8100e62cb3569182b58edc511c5def3069'
PARENT_PHYSICS_SHA = '4ef61ca9d5689005391c487fe7bca14bf95b31b11f95db079040d65d5b9b1b8f'


def admission(candidate, anchor):
    """All cases pass; no more than 1% height loss or small force drift."""
    reasons = []
    numeric = ('mean_force_n', 'max_force_n', 'mean_com_stroke_m',
               'mean_wheel_cm', 'mean_com_cm', 'mean_recovery_s',
               'min_stable_s', 'max_rebound_mps')
    if any(not math.isfinite(candidate[k]) for k in numeric):
        return dict(passed=False, reasons=['nonfinite_metrics'])
    checks = {
        'all_cases': candidate['worlds'] > 0 and candidate['passed'] == candidate['worlds'],
        'wheel_height': candidate['mean_wheel_cm'] >= max(9.065, .99*anchor['mean_wheel_cm']),
        'com_height': candidate['mean_com_cm'] >= max(11.74, .99*anchor['mean_com_cm']),
        'actual_stroke': candidate['mean_com_stroke_m'] >= max(.050, .98*anchor['mean_com_stroke_m']),
        'mean_impact': candidate['mean_force_n'] <= 1.02*anchor['mean_force_n'],
        'worst_impact': candidate['max_force_n'] <= 1.03*anchor['max_force_n'],
        'recovery': candidate['mean_recovery_s'] <= anchor['mean_recovery_s']+.10,
        'stable_interval': candidate['min_stable_s'] >= 1.-1e-6,
        'no_rebound': candidate['max_rebound_mps'] <= 1e-6,
    }
    reasons += [name for name, passed in checks.items() if not passed]
    return dict(passed=not reasons, reasons=reasons, checks=checks)


def better(candidate, incumbent, anchor):
    return (admission(candidate, anchor)['passed']
            and candidate['mean_force_n'] < incumbent['mean_force_n']-.25)

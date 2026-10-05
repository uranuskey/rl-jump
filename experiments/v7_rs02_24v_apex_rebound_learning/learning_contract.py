"""Learning eligibility is separate from unchanged final qualification."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'v7_rs02_24v_assist_withdrawal'))
from withdrawal_contract import admission as qualification

TRAINING_REBOUND_LIMIT_MPS = .10  # Research admission bound, not a hardware limit.


def learning_admission(candidate, anchor):
    final = qualification(candidate, anchor)
    if 'checks' not in final:
        return dict(passed=False, reasons=final['reasons'], final_qualification=final)
    checks = {k: v for k, v in final['checks'].items() if k != 'no_rebound'}
    checks['bounded_training_rebound'] = 0 <= candidate['max_rebound_mps'] <= TRAINING_REBOUND_LIMIT_MPS
    reasons = [k for k, v in checks.items() if not v]
    return dict(passed=not reasons, reasons=reasons, checks=checks,
                final_qualification=final, training_rebound_limit_mps=TRAINING_REBOUND_LIMIT_MPS)


def selected_better(candidate, incumbent, anchor):
    if not qualification(candidate, anchor)['passed']:
        return False
    return (incumbent is None or not qualification(incumbent, anchor)['passed']
            or candidate['mean_force_n'] < incumbent['mean_force_n']-.25)


def diagnostic_better(candidate, incumbent, anchor):
    if not learning_admission(candidate, anchor)['passed']:
        return False
    return (candidate['max_rebound_mps'], candidate['mean_force_n']) < (
        incumbent['max_rebound_mps'], incumbent['mean_force_n'])


def evaluation_entries(train):
    level = train['contract']['strength']
    selected = train['selected']
    return [('reference625', train['source_checkpoint'], .625),
            ('seed', train['initial_checkpoint'], level),
            ('selected' if selected is not None else 'candidate',
             (selected or train['diagnostic_candidate'])['checkpoint'], level),
            ('latest', train['final_checkpoint'], level)]


def replay_qualified(rows, count=3):
    return (len(rows) == count and [r['repeat'] for r in rows] == list(range(1, count+1))
            and all(r['metrics']['worlds'] == 512 and r['admission']['passed'] for r in rows))

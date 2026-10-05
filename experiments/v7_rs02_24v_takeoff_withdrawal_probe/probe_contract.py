"""Reduce assistance only after qualification; impact improvement is not a gate."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'v7_rs02_24v_assist_withdrawal'))
from withdrawal_contract import admission

LEVELS = (
    ('current625_600', .625, .60),
    ('takeoff6125_600', .6125, .60),
    ('uniform600', .60, .60),
    ('uniform575', .575, .575),
)


def checked_levels(before, after):
    assert 0 <= after <= before <= .625
    assert (before, after) in [(a, b) for _, a, b in LEVELS] + [(.625, .625)]
    return before, after


def qualified(row, repeats=3):
    native = row['native']
    batch = row['batch_replays']
    return (native['metrics']['worlds'] == 45 and native['admission']['passed']
            and native['audits']['status'] == 'PASS' and len(batch) == repeats
            and [r['repeat'] for r in batch] == list(range(1, repeats+1))
            and all(r['metrics']['worlds'] == 512 and r['admission']['passed'] for r in batch))


def deepest_qualified(models):
    chosen = None
    for name, _, _ in LEVELS:
        row = models.get(name)
        if row is None or not qualified(row):
            break
        chosen = name
    return chosen

"""Relearn at61.25/60 after the explicit rotor correction; old failures persist."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = HERE.parent/'v7_rs02_24v_rotor_constraint_fix'
sys.path.insert(0, str(PARENT))
from fix_runtime import verify as verify_parent, contract as parent_contract, read, write, sha, now, metrics, exclusive
from compliant_runtime import resource_limit
from apex_training_runtime import exploration
from transfer_contract import BEFORE, AFTER, VARIANT

PARENT_SHA = 'c5ea8dd2111aac3a5e8634e8f7316d9e264d744408c72a8c952bd002e923e42c'


def verify():
    assert verify_parent() == PARENT_SHA
    f = read(HERE/'FROZEN.json')
    assert f['parent_fix_sha256'] == PARENT_SHA
    for rel, digest in f['sha256'].items():
        assert sha(ROOT/rel) == digest, 'Withdrawal adaptation source changed: '+rel
    return sha(HERE/'FROZEN.json')


def receipt(folder, filename='exit_receipt.json'):
    r = read(Path(folder)/filename)
    assert r['process_exited'] and r['exit_code'] == 0
    return r


def contract():
    h = parent_contract()
    folder = PARENT/'runs/fix_01'
    receipt(folder); receipt(folder, 'audit_exit_receipt.json')
    receipt(folder.with_name('fix_01_launcher'), 'wrapper_exit_receipt.json')
    a = read(folder/'audit_result.json')
    assert a['status'] == 'AUDITED' and a['frozen_sha256'] == PARENT_SHA
    assert a['constraint_variant'] == VARIANT and not a['further_withdrawal_qualified']
    assert a['models']['reference625']['qualified']
    assert not a['models']['current625_600']['qualified']
    assert a['models']['current625_600']['batch_replays'][0]['admission']['reasons'] == ['no_rebound']
    assert a['source_checkpoint'] == h['source_checkpoint']
    h.update(fix_audit=dict(path=str(folder/'audit_result.json'), sha256=sha(folder/'audit_result.json')),
        fix_reference=a['models']['reference625'], before=BEFORE, after=AFTER, constraint_variant=VARIANT,
        old_current_failed_qualification_retained=True,
        target_kind='independent lower-assistance learning target, not a promoted source policy')
    h['anchors_native'] = dict(corrected_reference=h['fix_reference']['native']['metrics'],
        original_reference=h['old_models']['reference625']['native']['metrics'])
    h['anchors_batch'] = dict(corrected_reference=h['fix_reference']['batch_replays'][0]['metrics'],
        original_reference=h['old_models']['reference625']['batch_replays'][0]['metrics'])
    return h


def limit_for(started, seconds=43200):
    parent = resource_limit(started, seconds)
    def limit():
        assert not (HERE/'STOP').exists(), 'Rotor6125 adaptation STOP requested'
        parent()
    return limit


def checked(folder, status):
    folder = Path(folder)
    receipt(folder)
    d = read(folder/'result.json')
    assert d['status'] == status and d['contract'] == contract()
    assert d['frozen_sha256'] == d['final_frozen_sha256'] == verify()
    return d

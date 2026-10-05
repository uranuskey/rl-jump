"""New explicit physics revision, preserving every old frozen experiment."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = HERE.parent/'v7_rs02_24v_takeoff_withdrawal_probe'
sys.path.insert(0, str(PARENT))
from probe_runtime import (verify as verify_parent, handoff_contract, read, write, sha,
                           now, metrics, exclusive)
from compliant_runtime import resource_limit

PARENT_SHA = 'c1a0cf35025fe9328b72b816233220e99a1d481103c7d47ff79288ea332f6f2e'


def verify():
    assert verify_parent() == PARENT_SHA
    manifest = read(HERE/'FROZEN.json')
    assert manifest['parent_probe_sha256'] == PARENT_SHA
    for rel, digest in manifest['sha256'].items():
        assert sha(ROOT/rel) == digest, 'Constraint study source changed: '+rel
    return sha(HERE/'FROZEN.json')


def contract():
    h = handoff_contract(PARENT/'runs/source_handoff_01.json')
    p = PARENT/'runs/probe_01'
    a = read(p/'audit_result.json')
    for f in ('exit_receipt.json', 'audit_exit_receipt.json'):
        r = read(p/f)
        assert r['process_exited'] and r['exit_code'] == 0
    assert a['status'] == 'AUDITED' and a['deepest_qualified'] == 'current625_600'
    assert a['models']['current625_600']['qualified']
    assert not a['models']['takeoff6125_600']['qualified']
    h['old_audit'] = dict(path=str(p/'audit_result.json'), sha256=sha(p/'audit_result.json'))
    h['old_models'] = a['models']
    return h


def limit_for(started, seconds):
    old = resource_limit(started, seconds)
    def limit():
        assert not (HERE/'STOP').exists(), 'Rotor constraint study STOP requested'
        old()
    return limit

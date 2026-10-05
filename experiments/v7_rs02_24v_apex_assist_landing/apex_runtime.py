"""Phase-specific assistance over unchanged, verified physics and controllers."""
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = HERE.parent/'v7_rs02_24v_assist_withdrawal'
sys.path.insert(0, str(PARENT))
from withdrawal_runtime import (verify as verify_parent, source_contract, load_source,
    metrics, read, write, sha, now, exclusive, limit_for as parent_limit)
from withdrawal_contract import (SOURCE_SHA, SOURCE_LEVEL, PARENT_PHYSICS_SHA,
                                 admission, better)

PARENT_SHA = 'cf4f32c7d77e2706f3ed3b8b42c0095ce451f76c39e935f6fd6dd590219d0919'
RAMP_TICKS = 40
TARGETS = (.60, .6125)


def verify():
    assert verify_parent() == PARENT_SHA
    manifest = read(HERE/'FROZEN.json')
    assert manifest['parent_sha256'] == PARENT_SHA
    for rel, digest in manifest['sha256'].items():
        assert sha(ROOT/rel) == digest, 'Apex assistance source changed: '+rel
    return sha(HERE/'FROZEN.json')


def limit_for(started, seconds):
    old = parent_limit(started, seconds)
    def limit():
        assert not (HERE/'STOP').exists(), 'Apex assistance STOP requested'
        old()
    return limit

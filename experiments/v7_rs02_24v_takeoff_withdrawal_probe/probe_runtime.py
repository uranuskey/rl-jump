"""Verify immutable ancestors and preserve the deliberately stopped PPO evidence."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = HERE.parent/'v7_rs02_24v_apex_rebound_learning'
sys.path.insert(0, str(PARENT))
from learning_runtime import verify as verify_parent, read, write, sha, now, metrics, exclusive
from withdrawal_runtime import source_contract
from compliant_runtime import resource_limit
from probe_contract import admission

PARENT_SHA = '46770936761e52cf6389e12c4ae3633623ee152d49c9fa3b3aab4de224ef7ac6'


def verify():
    assert verify_parent() == PARENT_SHA
    manifest = read(HERE/'FROZEN.json')
    assert manifest['parent_training_sha256'] == PARENT_SHA
    for rel, digest in manifest['sha256'].items():
        assert sha(ROOT/rel) == digest, 'Withdrawal probe source changed: '+rel
    return sha(HERE/'FROZEN.json')


def handoff_contract(path):
    path = Path(path)
    h = read(path)
    assert h['status'] == 'STOP_REQUESTED_FOR_OBJECTIVE_CHANGE'
    assert h['source_training_frozen_sha256'] == PARENT_SHA
    training = Path(h['source_training'])
    assert training.resolve() == (PARENT/'runs/train_01').resolve()
    result = read(training/'result.json')
    assert result['status'] == 'ERROR_STOPPED'
    assert result['error'] == "AssertionError('Rebound learning STOP requested')"
    evidence = {}
    for p in (training/'exit_receipt.json', training.with_name(training.name+'_launcher')/'wrapper_exit_receipt.json'):
        receipt = read(p)
        assert receipt['process_exited'] and receipt['exit_code'] == 1
        evidence[str(p)] = sha(p)
    assert result['completed_updates'] >= h['source_checkpoint']['update'] > 0
    assert result['actor_steps'] > 0
    assert sha(h['source_checkpoint']['path']) == h['source_checkpoint']['sha256']
    e = h['source_evaluation']
    assert sha(e['path']) == e['sha256']
    entry = read(e['path'])
    assert entry['checkpoint'] == h['source_checkpoint']
    measured = metrics(dict(cases=entry['cases'], reasons=entry['metrics']['reasons']))
    assert measured == entry['metrics']
    assert entry['admission'] == admission(measured, h['anchor_metrics'])
    assert entry['admission']['passed'] and measured['worlds'] == 512
    original = source_contract()
    assert h['profile'] == original['profile']
    return dict(handoff=dict(path=str(path), sha256=sha(path)),
        source_checkpoint=h['source_checkpoint'], source_evaluation=e,
        parent_exit_evidence=evidence, parent_result_sha256=sha(training/'result.json'),
        parent_completed_updates=result['completed_updates'], parent_actor_steps=result['actor_steps'],
        parent_was_deliberately_stopped=True, original_reference=original,
        profile=h['profile'], historical_anchor=h['anchor_metrics'])


def limit_for(started, seconds):
    # The old learning STOP remains in place. It must not disable this new,
    # independently authorized probe or be removed to restart the old study.
    parent = resource_limit(started, seconds)
    def limit():
        assert not (HERE/'STOP').exists(), 'Takeoff withdrawal probe STOP requested'
        parent()
    return limit

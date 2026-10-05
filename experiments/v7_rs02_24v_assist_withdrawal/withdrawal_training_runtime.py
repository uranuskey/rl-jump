"""Conditional PPO admission; no automatic second assistance reduction."""
from pathlib import Path
from withdrawal_runtime import HERE, ROOT, verify, read, write, sha, source_contract
from withdrawal_contract import admission


def verify_training():
    probe_frozen = verify()
    manifest = read(HERE/'TRAINING_FROZEN.json')
    assert manifest['probe_frozen_sha256'] == probe_frozen
    for rel, digest in manifest['sha256'].items():
        assert sha(ROOT/rel) == digest, 'Training source changed: '+rel
    return sha(HERE/'TRAINING_FROZEN.json')


def checked_result(path, status, stage):
    path = Path(path)
    receipt = read(path.parent/'exit_receipt.json')
    result = read(path)
    assert receipt['process_exited'] and receipt['exit_code'] == 0 and receipt['stage'] == stage
    assert result['status'] == status
    assert result['frozen_sha256'] == result['final_frozen_sha256'] == verify()
    return result


def probe_contract(path):
    path = Path(path)
    probe = checked_result(path, 'PROBE_COMPLETED', 'probe')
    assert probe['lower_assistance_admitted'] and probe['selected'] is not None
    assert probe['contract'] == source_contract()
    selected = probe['models'][probe['selected']['name']]
    reference = probe['models']['reference625']
    assert selected['audits']['status'] == reference['audits']['status'] == 'PASS'
    assert admission(selected['metrics'], reference['metrics'])['passed']
    assert selected['strength'] in (.60, .6125)
    for name in (probe['selected']['name'], 'reference625'):
        row = probe['models'][name]
        assert sha(path.parent/(name+'.json')) == row['summary_sha256']
        assert sha(path.parent/(name+'_traces.npz')) == row['trace_sha256']
    return dict(**probe['contract'], strength=selected['strength'],
                anchor_metrics=reference['metrics'], probe_metrics=selected['metrics'],
                probe_result=dict(path=str(path), sha256=sha(path)))


def exploration(policy, update):
    t = max(0., min(1., update/128))
    policy.std.copy_(policy.std.new_tensor((.035-.010*t,)*8+(.012-.004*t,)*8))

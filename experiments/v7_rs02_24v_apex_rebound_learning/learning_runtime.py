"""Isolated learning entry over the frozen apex assistance experiment."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = HERE.parent/'v7_rs02_24v_apex_assist_landing'
sys.path.insert(0, str(PARENT))
from apex_runtime import (verify as verify_apex, source_contract, load_source,
    metrics as parent_metrics, read, write, sha, now, exclusive, limit_for as parent_limit)
from apex_training_runtime import (verify_training as verify_parent_training,
                                   probe_contract, exploration)
from learning_contract import learning_admission, qualification

PARENT_PROBE_SHA = 'b4eae640eee4fc25be7c4552681dfd935a3954b9e21ab75b193fa5a065aae26a'
PARENT_TRAINING_SHA = '832a3e63b3c9031153c5eb9c15b4144fcf8c2a640fe09c31d874aa239b370040'


def verify():
    assert verify_apex() == PARENT_PROBE_SHA
    assert verify_parent_training() == PARENT_TRAINING_SHA
    manifest = read(HERE/'TRAINING_FROZEN.json')
    assert manifest['parent_training_sha256'] == PARENT_TRAINING_SHA
    for rel, digest in manifest['sha256'].items():
        assert sha(ROOT/rel) == digest, 'Learning source changed: '+rel
    return sha(HERE/'TRAINING_FROZEN.json')


verify_training = verify


def metrics(summary):
    result = parent_metrics(summary)
    values = [c['landing_metrics']['peak_rebound_vz_mps'] for c in summary['cases']]
    result.update(rebounding_worlds=sum(v > 1e-6 for v in values),
                  mean_rebound_mps=sum(values)/len(values))
    return result


def checked_result(path, status, stage):
    path = Path(path)
    receipt, result = read(path.parent/'exit_receipt.json'), read(path)
    assert receipt['process_exited'] and receipt['exit_code'] == 0 and receipt['stage'] == stage
    assert result['status'] == status
    assert result['frozen_sha256'] == result['final_frozen_sha256'] == verify()
    return result


def learning_evidence(probe_path):
    """Reuse measured physics evidence; never relabel the failed old validation."""
    contract = probe_contract(probe_path)
    assert contract['strength'] == .60
    evidence = {}
    for name in ('train_01_smoke', 'train_01_validate'):
        folder = PARENT/'runs'/name
        result, receipt = read(folder/'result.json'), read(folder/'exit_receipt.json')
        assert result['contract'] == contract
        assert result['frozen_sha256'] == PARENT_PROBE_SHA
        assert result['training_frozen_sha256'] == PARENT_TRAINING_SHA
        assert receipt['process_exited']
        files = ['result.json', 'exit_receipt.json']
        if name.endswith('smoke'):
            assert receipt['stage'] == 'smoke' and receipt['exit_code'] == 0
            assert result['status'] == 'SMOKE_COMPLETED' and result['completed_updates'] == 2
            assert result['actor_steps'] > 0
            assert result['final_training_frozen_sha256'] == PARENT_TRAINING_SHA
            assert result['latest_evaluation']['admission']['passed']
        else:
            assert receipt['stage'] == 'validate' and receipt['exit_code'] == 1
            assert result['status'] == 'ERROR_STOPPED'
            assert result['completed_updates'] == result['actor_steps'] == 0
            assert result['initial_admission']['reasons'] == ['no_rebound']
            reference = metrics(read(folder/'reference625.json'))
            seed = metrics(read(folder/'baseline.json'))
            assert seed['worlds'] == reference['worlds'] == 512
            assert qualification(reference, contract['anchor_metrics'])['passed']
            assert learning_admission(seed, reference)['passed']
            assert not qualification(seed, reference)['passed']
            evidence['historical_reference_metrics'] = reference
            evidence['historical_seed_metrics'] = seed
            evidence['new_learning_admission'] = learning_admission(seed, reference)
            files += ['reference625.json', 'baseline.json']
        evidence[name] = dict(status=result['status'], actual_exit_code=receipt['exit_code'],
            folder=str(folder), sha256={f: sha(folder/f) for f in files})
    return contract, evidence


def limit_for(started, seconds):
    parent = parent_limit(started, seconds)
    def limit():
        assert not (HERE/'STOP').exists(), 'Rebound learning STOP requested'
        parent()
    return limit

"""Isolated assistance experiment; verify every frozen ancestor before use."""
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = HERE.parent/'v7_rs02_24v_slot_landing_probe'
sys.path.insert(0, str(PARENT))
from slot_training_runtime import verify_training, sha, read, write, now, exclusive, metrics
from slot_probe import resource_limit
from withdrawal_contract import (SOURCE_SHA, SOURCE_AUDIT_SHA, SOURCE_LEVEL,
                                 PARENT_TRAINING_SHA, PARENT_PHYSICS_SHA)


def verify():
    assert verify_training() == PARENT_TRAINING_SHA
    manifest = read(HERE/'FROZEN.json')
    assert manifest['parent_training_sha256'] == PARENT_TRAINING_SHA
    for rel, digest in manifest['sha256'].items():
        assert sha(ROOT/rel) == digest, 'Withdrawal source changed: '+rel
    return sha(HERE/'FROZEN.json')


def source_contract():
    training = PARENT/'runs/train_02'
    evaluation = PARENT/'runs/train_02_eval'
    audit_path = evaluation/'audit_result.json'
    assert sha(audit_path) == SOURCE_AUDIT_SHA
    audit = read(audit_path)
    assert audit['status'] == 'AUDITED' and audit['completed_updates'] == 128
    for path in (training/'exit_receipt.json', evaluation/'exit_receipt.json',
                 evaluation/'audit_exit_receipt.json'):
        receipt = read(path)
        assert receipt['process_exited'] and receipt['exit_code'] == 0
    selected = audit['models']['selected']
    checkpoint = training/'model_0112.pt'
    assert sha(checkpoint) == selected['checkpoint']['sha256'] == SOURCE_SHA
    assert selected['checkpoint']['update'] == 112
    assert selected['profile']['name'] == 'late3_slot_high'
    return dict(source_checkpoint=dict(path=str(checkpoint), sha256=SOURCE_SHA, update=112),
                profile=selected['profile'], previous_metrics=selected['metrics'],
                previous_assist=SOURCE_LEVEL, source_audit_sha256=SOURCE_AUDIT_SHA)


def limit_for(started, seconds):
    parent = resource_limit(started, seconds)
    def limit():
        assert not (HERE/'STOP').exists(), 'Withdrawal STOP requested'
        assert not (PARENT/'STOP').exists(), 'Parent STOP requested'
        parent()
    return limit


def load_source(policy, contract):
    import torch
    path = contract['source_checkpoint']['path']
    assert sha(path) == SOURCE_SHA
    state = torch.load(path, map_location='cpu', weights_only=True)
    assert state['update'] == 112 and state['training_frozen_sha256'] == PARENT_TRAINING_SHA
    assert state['frozen_sha256'] == PARENT_PHYSICS_SHA
    assert state['voltage_v'] == 24 and state['assist_strength'] == SOURCE_LEVEL
    assert state['action_dim'] == 16 and state['profile'] == contract['profile']
    policy.load_state_dict(state['model_state_dict'], strict=True)
    return state

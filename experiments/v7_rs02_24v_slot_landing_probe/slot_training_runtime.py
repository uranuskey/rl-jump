"""Immutable training contract and admission of a reproduced physical seed."""
from pathlib import Path
import time
from slot_probe import (HERE, ROOT, SOURCE_SHA, SOURCE_FROZEN, code_hashes,
                        verify, sha, read, write, now, exclusive, resource_limit,
                        seed_qualified, metrics, eligible)


def verify_training():
    assert verify() == SOURCE_FROZEN
    manifest = read(HERE/'TRAINING_FROZEN.json')
    assert manifest['parent_frozen_sha256'] == SOURCE_FROZEN
    for rel, digest in manifest['sha256'].items():
        assert sha(ROOT/rel) == digest, 'Training source changed: '+rel
    return sha(HERE/'TRAINING_FROZEN.json')


def checked_result(path, status, stage=None):
    path = Path(path)
    result, receipt = read(path), read(path.parent/'exit_receipt.json')
    assert result['status'] == status, (path, result['status'])
    assert receipt['process_exited'] and receipt['exit_code'] == 0, receipt
    if stage is not None:
        assert receipt['stage'] == stage
    return result


def seed_contract(probe):
    probe = Path(probe)
    pipeline = read(probe/'pipeline_receipt.json')
    assert pipeline['status'] == 'SEED_VALIDATED'
    assert pipeline['stages'] and all(r['process_exited'] and r['exit_code'] == 0
                                     for r in pipeline['stages'])
    baseline_path = probe.with_name(probe.name+'_baseline')/'result.json'
    repeat_path = probe.with_name(probe.name+'_repeat')/'result.json'
    baseline = checked_result(baseline_path, 'EVALUATED', 'baseline')
    repeat = checked_result(repeat_path, 'EVALUATED', 'repeat')
    selected = checked_result(read(probe/'selection.json')['result'], 'EVALUATED')
    for result in (baseline, selected, repeat):
        assert result['frozen_sha256'] == result['final_frozen_sha256'] == SOURCE_FROZEN
        assert result['source_sha256'] == result['final_source_sha256'] == code_hashes()
        for key in ('physical_audit', 'controller_audit', 'schedule_audit', 'slot_audit'):
            assert result[key]['status'] == 'PASS'
    assert repeat['profiles'] == selected['profiles'] and len(repeat['profiles']) == 1
    assert seed_qualified(repeat['all_metrics'], baseline['all_metrics'])
    assert seed_qualified(selected['all_metrics'], baseline['all_metrics'])
    checkpoint = repeat['source_checkpoint']
    assert sha(checkpoint['path']) == checkpoint['sha256'] == SOURCE_SHA
    return dict(profile=repeat['profiles'][0], baseline=baseline['all_metrics'],
                repeated_seed=repeat['all_metrics'], source_checkpoint=checkpoint,
                baseline_result=dict(path=str(baseline_path),sha256=sha(baseline_path)),
                repeat_result=dict(path=str(repeat_path),sha256=sha(repeat_path)),
                probe=str(probe), authorization='User authorized self approval and PPO',
                improved_before_ppo=eligible(repeat['all_metrics'], baseline['all_metrics']))


def limit_for(started, seconds):
    parent = resource_limit(started, seconds)
    def limit():
        assert not (HERE/'STOP').exists(), 'Slot study STOP requested'
        parent()
    return limit


def receipt_identity(result, frozen):
    assert result['training_frozen_sha256'] == result['final_training_frozen_sha256'] == frozen
    assert result['frozen_sha256'] == result['final_frozen_sha256'] == SOURCE_FROZEN


def selected_better(candidate, incumbent, baseline):
    return (seed_qualified(candidate, baseline)
            and candidate['mean_force_n'] <= .97*baseline['mean_force_n']
            and candidate['max_force_n'] <= baseline['max_force_n']
            and candidate['mean_force_n'] < incumbent['mean_force_n']-.25)

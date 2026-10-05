"""Verify reused evidence and gate semantics without another physical replay."""
import argparse
import re
from pathlib import Path
from learning_runtime import HERE, verify, learning_evidence, write, now


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-id', required=True)
    p.add_argument('--probe-result', type=Path, required=True)
    args = p.parse_args()
    assert re.fullmatch('[A-Za-z0-9_-]+', args.run_id)
    frozen = verify()
    contract, evidence = learning_evidence(args.probe_result)
    out = HERE/'runs'/args.run_id
    assert not out.exists(), 'Use a fresh preflight id'
    out.mkdir(parents=True)
    result = dict(status='LEARNING_ENTRY_VERIFIED', utc_end=now(), contract=contract,
        evidence=evidence, physics_replays=0, ppo_updates=0, final_qualified=False,
        frozen_sha256=frozen, final_frozen_sha256=verify())
    write(out/'result.json', result)
    print(dict(status=result['status'], historical_validate_exit=1,
               new_learning_admission=evidence['new_learning_admission']), flush=True)


if __name__ == '__main__':
    main()

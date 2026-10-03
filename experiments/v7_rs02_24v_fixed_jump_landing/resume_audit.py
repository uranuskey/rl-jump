"""Additional source/state/mixed-batch audit after the original 400 Hz trace audit."""
import argparse
from pathlib import Path
import bootstrap
from runtime import sha, write
from resume_state import read, verify_resume


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--training', type=Path, required=True)
    p.add_argument('--evaluation', type=Path, required=True)
    args = p.parse_args()
    frozen, resume_frozen = verify_resume()
    result, audit = read(args.training/'result.json'), read(args.evaluation/'audit_result.json')
    receipt = read(args.training/'exit_receipt.json')
    assert result['resume_frozen_sha256']==result['final_resume_frozen_sha256']==receipt['resume_frozen_sha256']==resume_frozen
    assert result['frozen_sha256']==audit['frozen_sha256']==frozen
    assert audit['status']=='AUDITED' and result['status']=='BUDGET_COMPLETED'
    source = result['resumed_from']
    original = Path(source['run'])
    assert sha(original/'result.json')==source['result_sha256']
    assert sha(source['checkpoint']['path'])==source['checkpoint']['sha256']
    boundary = source['checkpoint']['update']
    assert result['restore']['update']==boundary and result['restore']['model_and_both_optimizers_restored']
    assert result['same_state_sensor_equivalence'] and not result['bitwise_trajectory_continuation']
    for i in range(1, 129):
        path = args.training/f'update_{i:04d}.json'
        row = read(path)
        assert row['trial']['worlds']==(2048 if i<=boundary else 512)
        if i<=boundary:
            assert sha(path)==sha(original/path.name)
    assert sha(args.training/'initial.pt')==sha(original/'initial.pt')
    report = dict(status='AUDITED', policy_frozen_sha256=frozen, resume_frozen_sha256=resume_frozen,
        environment_schedule=result['environment_schedule'], inherited_updates=boundary,
        new_updates=128-boundary, total_updates=128, resumed_from=source,
        original_audit_sha256=sha(args.evaluation/'audit_result.json'),
        zero_assistance_qualified=False, hardware_qualified=False)
    write(args.evaluation/'resume_audit_result.json', report)
    print(report)


if __name__=='__main__':
    main()

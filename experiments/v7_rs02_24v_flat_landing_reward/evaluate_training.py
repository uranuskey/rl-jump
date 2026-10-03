"""Serial, fixed-assistance baseline/selected-policy evaluation with full traces."""
import argparse
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import bootstrap
from bootstrap import HERE
from runtime import exclusive, sha, verify, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-result', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    frozen = verify()
    trained = json.loads(args.train_result.read_text(encoding='utf-8'))
    assert trained['status'] == 'BUDGET_COMPLETED'
    assert trained['completed_updates'] == 128
    assert trained['frozen_sha256'] == trained['final_frozen_sha256'] == frozen
    assert trained['final_training_assist'] == .625
    folder = args.train_result.parent
    initial = dict(path=str(folder / 'initial.pt'), sha256=sha(folder / 'initial.pt'), update=0)
    selected = trained['selected']['checkpoint']
    assert sha(Path(selected['path'])) == selected['sha256']
    out = HERE / 'runs' / args.run_id
    assert not out.exists()
    with exclusive(45) as resource:
        out.mkdir(parents=True)
        start = time.monotonic()
        result = dict(status='STARTING', resource=resource, frozen_sha256=frozen,
                      evaluator_sha256=sha(Path(__file__)), training_result_sha256=sha(args.train_result),
                      utc_start=datetime.now(timezone.utc).isoformat(), cases=45,
                      assist_strength=.625, voltage_v=24, terrain='flat',
                      assistance_withdrawal_enabled=False, hardware_used=False,
                      case_scope='Original nine delays times five initial conditions; not held-out conditions',
                      selected_is_initial=selected['update'] == 0)

        def limit():
            import torch
            if (HERE / 'STOP').exists() or (HERE.parent / 'v7_jump_in_place/STOP').exists():
                raise RuntimeError('STOP requested')
            if time.monotonic() - start > 1200:
                raise RuntimeError('Evaluation wall time bound')
            if torch.cuda.mem_get_info()[0] < 512 * 1024 ** 2:
                raise RuntimeError('GPU reserve')

        try:
            import torch
            from landing import LandingEnv
            from standing_policy import JumpPolicy
            from learning import Policy
            from train import compact, qualifies, trial
            torch.set_num_threads(1)
            torch.manual_seed(827)
            torch.cuda.manual_seed_all(827)
            env = LandingEnv(45, abort_dir=out / 'aborts')
            result['mass_kg'] = float(env.m.body_subtreemass[env.base])
            standing = JumpPolicy(env.device).eval()
            policy = Policy(env.device).eval()
            candidates = [('baseline', initial)]
            if not result['selected_is_initial']:
                candidates.append(('selected', selected))
            for name, checkpoint in candidates:
                path = Path(checkpoint['path'])
                assert sha(path) == checkpoint['sha256']
                payload = torch.load(path, map_location=env.device, weights_only=False)
                assert payload['frozen_sha256'] == frozen and payload['voltage_v'] == 24
                policy.load_state_dict(payload['model_state_dict'])
                *_, summary = trial(env, standing, policy, limit, .625, False, out / (name + '_traces.npz'))
                write(out / (name + '.json'), summary)
                result[name] = dict(checkpoint=checkpoint, evaluation=compact(summary), qualifies=qualifies(summary))
                print(json.dumps(dict(event=name, **result[name])), flush=True)
            if result['selected_is_initial']:
                result['selected'] = result['baseline']
            result.update(status='EVALUATED', final_frozen_sha256=verify(),
                          selected_qualified=result['selected']['qualifies'],
                          trained_policy_promoted=not result['selected_is_initial'] and result['selected']['qualifies'],
                          zero_assistance_qualified=False, hardware_qualified=False)
        except BaseException as error:
            result.update(status='ERROR_STOPPED', error=repr(error), traceback=traceback.format_exc())
            print(result['traceback'], flush=True)
        finally:
            result.update(wall_s=time.monotonic() - start, utc_end=datetime.now(timezone.utc).isoformat())
            write(out / 'result.json', result)
            print(json.dumps({k: v for k, v in result.items() if k not in ('baseline', 'selected')}), flush=True)
    return int(result['status'] == 'ERROR_STOPPED')


if __name__ == '__main__':
    raise SystemExit(main())

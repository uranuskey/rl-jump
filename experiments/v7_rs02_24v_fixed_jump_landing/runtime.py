"""128-update24V learned reference/force/velocity-feedforward experiment."""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import bootstrap
from bootstrap import HERE, ROOT, CHECKPOINT, SOURCE_SHA,TRAINED_SHA
from curve_contract import ACTOR_DIM,CRITIC_DIM,ACTION_DIM


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    for attempt in range(20):
        try:
            tmp.replace(path)
            break
        except PermissionError:
            if attempt==19:raise
            time.sleep(.05)


def verify():
    d = json.loads((HERE/'FROZEN.json').read_text(encoding='utf-8'))
    for rel, digest in d['sha256'].items():
        if sha(ROOT/rel) != digest:
            raise RuntimeError('Frozen source changed: '+rel)
    assert sha(CHECKPOINT) == SOURCE_SHA
    import importlib.metadata
    import mujoco_warp
    expected = json.loads((ROOT/'patches/manifest.json').read_text())
    if importlib.metadata.version('mujoco-warp') != expected['version']:
        raise RuntimeError('mujoco-warp version mismatch')
    native = Path(mujoco_warp.__file__).parent/'_src/forward.py'
    if sha(native) != expected['patched_forward_sha256']:
        raise RuntimeError('mujoco-warp forward.py differs from frozen physics dependency')
    return sha(HERE/'FROZEN.json')


@contextmanager
def exclusive(n):
    from runtime_gate import runtime_conflicts
    others = runtime_conflicts()
    if others and n > 256:
        raise RuntimeError('Other executor active; use authorized smaller run')
    free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()[0])
    threshold = 6144 if n > 256 else 1536
    if free < threshold:
        raise RuntimeError(f'Insufficient startup GPU reserve: {free} < {threshold} MiB')
    lock = HERE.parent/'v7_jump_in_place/PHYSICS_LOCK.json'
    resource = dict(pid=os.getpid(), num_envs=n, gpu_free_mib=free, concurrent_executors=others,
                    small_run_fallback_authorized=True, utc=datetime.now(timezone.utc).isoformat())
    with lock.open('x', encoding='utf-8') as f:
        json.dump(resource, f)
    try:
        yield resource
    finally:
        if lock.exists() and json.loads(lock.read_text(encoding='utf-8'))['pid'] == os.getpid():
            lock.unlink()


def evaluate(env, policy, limit, trace_dir=None):
    import numpy as np
    import torch
    from environment import reset_cases, SUCCESS, FAILED, REASONS
    reset_cases(env)
    env.record = trace_dir is not None
    records = []
    n = min(45, env.n)
    peak = torch.zeros(n, device=env.device)
    full_apex = torch.zeros(n, dtype=torch.bool, device=env.device)
    max_tilt = peak.clone()
    for step in range(500):
        limit()
        with torch.no_grad():
            env.step(policy.actor(env.obs), auto_reset=False)
        x = env.last['sample']
        free = x['wheel_force_n'][:n].max(1).values <= .5
        # Task metric is sampled at every physical tick, not only the actor clock.
        peak = torch.maximum(peak, env.task.peak_clearance_m[:n])
        full_apex |= (env.task.takeoff_time[:n] > 0) & free & (x['com_vz_mps'][:n] <= 0)
        max_tilt = torch.maximum(max_tilt, x['tilt_rad'][:n])
        if env.record:
            rows = env.last['traces']
            block = {k: torch.stack([r[k][:n] for r in rows]).cpu().numpy()
                for k in ('active','ticks','native_time','q','v','phase','reason','motor_target',
                          'arrived_reference_height','arrived_corrections','motor_position_pre','arrived_motor_velocity',
                          'requested_height','curve_state','curve_action','requested_thrust_force','thrust_requested_force',
                          'thrust_motor_unlimited','thrust_motor_applied','thrust_allocation_fraction','thrust_pair_contact',
                          'motor_request_before_thrust','base_rotation_pre',
                          'motor_base_request','motor_velocity_feedforward','requested_motor_velocity',
                          'assist_wrench','assist_strength','assist_effective_strength','assist_world_omega_pre','assist_world_omega_post',
                          'assist_power_w','assist_work_j','height_only_reward','reward_flight_height',
                          'reward_release_z','reward_release_time')}
            block.update({'sensor_'+k: torch.stack([r['sample'][k][:n] for r in rows]).cpu().numpy() for k in rows[0]['sample']})
            records.append(block)
        if bool(((env.task.phase == FAILED) | (env.ticks >= 4000)).all()):
            break
    passed = (env.task.phase == SUCCESS) & (env.ticks >= 4000)
    rows = []
    for i in range(n):
        rows.append(dict(case=i, delay_ticks=i//5, condition=i%5, passed=bool(passed[i]),
            reason=REASONS[int(env.task.reason[i])], ticks=int(env.ticks[i]),
            flight_height_m=float(env.task.height_score.peak[i]),
            height_only_return=float(env.task.height_score.total_reward[i]),
            contact_loss_time_s=float(env.task.height_score.release_time[i]),
            contact_loss_com_vz_mps=float(env.task.height_score.release_vz[i]),
            flight_height_apex_observed=bool(env.task.height_score.apex[i]),
            peak_clearance_m=float(peak[i]), post_takeoff_com_rise_m=float(env.task.com_rise_m[i]),
            takeoff_time_s=float(env.task.takeoff_time[i]), touchdown_time_s=float(env.task.touchdown_time[i]),
            observed_com_apex=bool(full_apex[i]), tilt_deg_at_policy_samples=float(max_tilt[i]*180/np.pi),
            target_15cm_band_hit=bool(.135 <= peak[i] <= .15),
            assist_strength=float(env.assist_strength[i]),assist_signed_work_j=float(env.assist_work[i,0]),
            assist_positive_work_j=float(env.assist_work[i,1]),assist_peak_torque_nm=float(env.assist_peak[i])))
    if trace_dir:
        trace_dir.mkdir(parents=True, exist_ok=False)
        arrays = {k: np.concatenate([b[k] for b in records]) for k in records[0]}
        records.clear()
        for i in range(n):
            mask = arrays['active'][:, i]
            a = {k: v[:, i][mask] for k, v in arrays.items()}
            assert len(a['q']) == rows[i]['ticks']
            np.savez_compressed(trace_dir/f'case_{i:03d}.npz', **a)
        del arrays
    result = dict(mean_flight_height_m=sum(r['flight_height_m'] for r in rows)/n,
        max_flight_height_m=max(r['flight_height_m'] for r in rows),
        mean_height_only_return=sum(r['height_only_return'] for r in rows)/n,
        reward_apex_cases=sum(r['flight_height_apex_observed'] for r in rows),
        reward_formula='120 * positive increment of COM peak above contact-loss COM; confirmed flight only',
        cases=rows, distinct_cases=n, worlds=env.n, passed_cases=sum(r['passed'] for r in rows),
        passed_worlds=int(passed.sum()), reasons=dict(Counter(r['reason'] for r in rows)),
        valid_takeoffs=sum(r['takeoff_time_s'] > 0 for r in rows),
        cases_at_least_1cm=sum(r['peak_clearance_m'] >= .01 for r in rows),
        peak_clearance_m=max(r['peak_clearance_m'] for r in rows),
        mean_clearance_m=sum(r['peak_clearance_m'] for r in rows)/n,
        mean_post_takeoff_com_rise_m=sum(r['post_takeoff_com_rise_m'] for r in rows)/n,
        cases_observed_com_apex=sum(r['observed_com_apex'] for r in rows),
        truncated_heights_are_observed_lower_bounds=True, hardware_ready=False,
        assist_strength=float(env.assist_strength[0]),
        mean_assist_signed_work_j=float(env.assist_work[:n,0].mean()),
        mean_assist_positive_work_j=float(env.assist_work[:n,1].mean()),
        max_assist_torque_nm=float(env.assist_peak[:n].max()),
        external_linear_force_n=[0.,0.,0.],external_yaw_torque_nm=0.,
        upper_bound_proven=False)
    env.record = False
    return result

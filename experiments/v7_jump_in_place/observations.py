"""CPU reference for ONE projection shared by rollout/PPO/evaluation/export.

Input raw35 follows the inspected V7 schema. Jump action history has the NEW
foot-target meaning. Contact, true clearance and event phase never reach actor.
The native Torch adapter and deployment implementation still require parity tests.
"""
import hashlib
import json
import numpy as np

VERSION = "V7_JUMP_FRESH_ACTOR136_CRITIC168_FOOT_TARGET_V1"
KEEP = tuple(i for i in range(35) if i not in (14, 16, 31, 32))
COMMAND_FIELDS = ("target_height_div_0p05", "jump_requested", "request_age_div_6")
PRIVILEGED_FIELDS = (
    "left_support_force_n", "right_support_force_n",
    "left_mesh_bottom_clearance_m", "right_mesh_bottom_clearance_m",
    "com_rise_since_takeoff_m", "com_vz_world_mps", "left_leg_height_m", "right_leg_height_m",
    "phase_settle", "phase_compress", "phase_flight", "phase_recover", "phase_success", "phase_failed",
    "phase_age_s", "stable_landing_hold_s",
)
SCHEMA = dict(version=VERSION, history_frames=4, history_order="oldest_to_newest",
              base_raw_per_frame=35, actor_raw_indices=KEEP, command_fields=COMMAND_FIELDS,
              privileged_fields=PRIVILEGED_FIELDS, actor_dim=136, critic_dim=168,
              normalization="disabled", action_dim=6,
              checkpoint_loading="No locomotion model or optimizer; fresh initialization only")
SCHEMA_SHA256 = hashlib.sha256(json.dumps(SCHEMA, sort_keys=True).encode()).hexdigest()


def compose(raw35_history, command_history, privileged):
    raw = np.asarray(raw35_history, dtype=np.float32)
    cmd = np.asarray(command_history, dtype=np.float32)
    private = np.asarray(privileged, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[1:] != (4, 35):
        raise ValueError("raw history must be [N,4,35]")
    n = raw.shape[0]
    if cmd.shape != (n, 4, 3) or private.shape != (n, 16):
        raise ValueError("command [N,4,3] and privileged [N,16] required")
    if not all(np.isfinite(v).all() for v in (raw, cmd, private)):
        raise ValueError("Nonfinite observations")
    if (cmd < 0).any() or (cmd > 1).any():
        raise ValueError("Normalized command features outside [0,1]")
    actor = np.concatenate((raw[:, :, KEEP], cmd), axis=-1).reshape(n, 136).copy()
    critic = np.concatenate((np.concatenate((raw, cmd), axis=-1).reshape(n, 152), private), axis=1)
    return actor, critic

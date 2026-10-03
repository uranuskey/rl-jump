"""Isolated continuous landing task; the imported original experiment is frozen."""
from pathlib import Path
import sys
import os
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FLAT_PARENT=HERE.parent/'v7_rs02_24v_flat_landing_reward'
sys.path.append(str(FLAT_PARENT))
# NVRTC needs ASCII temporary paths on Windows; change only this process.
if os.name == 'nt':
    scratch=ROOT/'.tmp'
    cache=ROOT/'.warp_cache'
    if not str(scratch).isascii():
        raise RuntimeError('Place the portable checkout in an ASCII path for CUDA NVRTC')
    scratch.mkdir(exist_ok=True)
    cache.mkdir(exist_ok=True)
    os.environ['TEMP']=os.environ['TMP']=str(scratch)
    os.environ['WARP_CACHE_PATH']=str(cache)
PARENT=HERE.parent/'v7_rs02_learned_curve'
CHECKPOINT=ROOT/'models/standing_crouch_verified.pt'
SOURCE_SHA='989d35d76c3f81a152db602a200f3d2ea3be62b4452db7a0a47a0c186e237f5b'
TRAINED_CHECKPOINT=ROOT/'models/voltage_training_0032.pt'
TRAINED_SHA='abdbd7452cdb5d75f84003255ee079f80b00da34768733004223649a9696d744'
CURVE_CHECKPOINT=ROOT/'models/learned_curve_0032.pt'
CURVE_SHA='1f32599c4c42d8cfe0da9f641b892946ec9a9b9690c3575e4ca1d12e0e7b4e7e'
for p in (PARENT,HERE.parent/'v7_rs02_voltage_training',HERE.parent/'v7_rs02_push_stroke_2cm',
          HERE.parent/'v7_rs02_24v_jump_probe',HERE.parent/'v7_height_conditioned_transfer',
          HERE.parent/'v7_jump_curriculum',HERE.parent/'v7_jump_in_place',
          HERE.parent/'v7_mujoco_training_gate/vendor',HERE.parent/'v7_huber_continuous02_256'):
    if str(p) not in sys.path:
        sys.path.append(str(p))

# No balance-assist checkpoint is loaded by this task.

ASSIST_PARENT=HERE.parent/'v7_rs02_balance_assist_ramp'

COMPARE_PARENT=HERE.parent/'v7_rs02_height_only'
COMPARE_CHECKPOINT=ROOT/'models/height_only_0016.pt'
COMPARE_SHA='533d334ccfbea3d4f3c0c17b19e599cb50bb725c760dc3ce8c8a17290da0779e'

VOLTAGE_PARENT=HERE.parent/'v7_rs02_height_voltage_compare'

ARCHIVE=HERE.parent/'v7_rs02_height_48v_training'

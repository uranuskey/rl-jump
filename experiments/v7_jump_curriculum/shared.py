"""Bind reused, frozen V7 jump components without modifying the prior experiment."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFERENCE = HERE.parent/'v7_jump_in_place'
if str(REFERENCE) not in sys.path:
    sys.path.append(str(REFERENCE))


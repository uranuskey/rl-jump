"""Explicit read-only dependencies of the isolated height-conditioned transfer."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CURRICULUM = HERE.parent / 'v7_jump_curriculum'
JUMP = HERE.parent / 'v7_jump_in_place'
for folder in (CURRICULUM, JUMP):
    if str(folder) not in sys.path:
        sys.path.append(str(folder))

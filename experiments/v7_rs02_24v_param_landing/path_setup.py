"""Use the existing frozen physics without shadowing its module names."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = HERE.parent/'v7_rs02_24v_fixed_jump_landing'
sys.path.insert(0, str(PARENT))
import bootstrap

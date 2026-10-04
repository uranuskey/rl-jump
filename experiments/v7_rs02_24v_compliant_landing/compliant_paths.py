"""Isolated controller revision over the frozen guided-landing physics."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = HERE.parent/'v7_rs02_24v_guided_landing'
sys.path.insert(0, str(PARENT))
import guided_paths

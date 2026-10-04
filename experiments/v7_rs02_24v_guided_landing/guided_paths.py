"""Reuse all three frozen parents without shadowing their module names."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = HERE.parent/'v7_rs02_24v_param_landing'
sys.path.insert(0, str(PARENT))
import path_setup

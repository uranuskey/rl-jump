"""Import immutable ancestors without changing their source or manifests."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PARENT=HERE.parent/'v7_rs02_24v_clearance_withdrawal'
sys.path.insert(0,str(PARENT))
import clearance_runtime as previous
from clearance_runtime import read,write,sha,now,metrics,exclusive
assert previous.HERE==PARENT

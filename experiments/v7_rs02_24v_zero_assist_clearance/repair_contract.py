"""Requalify repaired 45%, then immediately descend on unchanged strict gates."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'v7_rs02_24v_zero_assist_curriculum'))
from zero_contract import admission,qualification,learning_entry,decision,chunk_size,zero_qualified,wrench_valid
LEVELS=[(f'uniform{i:03d}',i/1000,i/1000) for i in range(450,-1,-25)]
BUDGET=128
PRIOR_UPDATES=2

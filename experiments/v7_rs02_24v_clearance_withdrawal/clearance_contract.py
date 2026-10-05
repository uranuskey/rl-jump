"""Immediate withdrawal from the retained55% baseline, preserving all prior gates."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'v7_rs02_24v_adaptive_withdrawal'))
from curriculum_contract import admission,qualification,learning_entry
LEVELS=[('uniform550',.55,.55),('uniform525',.525,.525),('uniform500',.50,.50)]
PRIOR_UPDATES=2
BUDGET=128-PRIOR_UPDATES

def chunk_size(completed,level_updates):
    assert 0<=completed<=BUDGET and level_updates>=0
    return min(2 if level_updates==0 else 8,BUDGET-completed)

def decision(rows,completed):
    if qualification(rows):return 'ADVANCE'
    if not learning_entry(rows):return 'ENTRY_FAILED'
    if completed>=BUDGET:return 'BUDGET_EXHAUSTED'
    return 'LEARN'

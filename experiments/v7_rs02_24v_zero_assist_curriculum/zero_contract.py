"""Uniform assistance curriculum down to exact zero; physical gates inherited."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'v7_rs02_24v_slot_allocator_withdrawal'))
from allocation_contract import admission,qualification as inherited_qualification,learning_entry as inherited_entry
LEVELS=[(f'uniform{i:03d}',i/1000,i/1000) for i in range(500,-1,-25)]
BUDGET=128
PRIOR_UPDATES=2

def checked_levels(before,after):
    assert (before,after) in [(a,b) for _,a,b in LEVELS], 'Undeclared assistance level'
    return before,after

def wrench_valid(row):
    e=row.get('external_wrench',{})
    before,after=row.get('before'),row.get('after')
    try: checked_levels(before,after)
    except (AssertionError,TypeError): return False
    return (e.get('min_sampled_ticks',0)>0
        and e.get('max_abs_linear_force_n')==0
        and e.get('max_abs_root_generalized_force')==0
        and 0<=e.get('max_abs_body_torque_nm',float('inf'))<=20*before+5e-5
        and (before!=0 or e.get('max_abs_body_torque_nm')==0))

def qualification(rows):
    return inherited_qualification(rows) and all(wrench_valid(r) for r in [rows['native']]+rows['batch'])

def learning_entry(rows):
    return inherited_entry(rows) and all(wrench_valid(r) for r in [rows['native']]+rows['batch'])

def decision(rows,completed):
    if qualification(rows):return 'ADVANCE'
    if not learning_entry(rows):return 'ENTRY_FAILED'
    if completed>=BUDGET:return 'BUDGET_EXHAUSTED'
    return 'LEARN'

def chunk_size(completed,level_updates):
    assert 0<=completed<=BUDGET and level_updates>=0
    return min(2 if level_updates==0 else 8,BUDGET-completed)

def zero_qualified(promotions):
    return bool(promotions and promotions[-1]['before']==promotions[-1]['after']==0)


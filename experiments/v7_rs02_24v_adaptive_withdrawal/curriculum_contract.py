"""Shared update budget and immediate, fully tested assistance reductions."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'v7_rs02_24v_rotor6125_learning'))
from transfer_contract import admission,final_qualified

LEVELS=[('takeoff6125_600',.6125,.60),('uniform600',.60,.60),
        ('uniform575',.575,.575),('uniform550',.55,.55),
        ('uniform525',.525,.525),('uniform500',.50,.50)]
BUDGET=128


def chunk_size(completed,level_updates):
    assert 0<=completed<=BUDGET and level_updates>=0
    return min(2 if level_updates==0 else 8,BUDGET-completed)


def qualification(rows):
    return final_qualified(rows,3)


def learning_entry(rows):
    return (rows['native']['audits']['status']=='PASS'
            and rows['native']['learning_admission']['passed']
            and len(rows['batch'])==3
            and all(r['learning_admission']['passed'] for r in rows['batch']))


def decision(rows,completed):
    if qualification(rows): return 'ADVANCE'
    if not learning_entry(rows): return 'ENTRY_FAILED'
    if completed>=BUDGET: return 'BUDGET_EXHAUSTED'
    return 'LEARN'

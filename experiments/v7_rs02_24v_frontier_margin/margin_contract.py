"""Train at the qualified frontier before attempting the failed lower level."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'v7_rs02_24v_rare_contact_learning'))
from rare_contract import admission,qualification,learning_admission,learning_entry,metrics,wrench_valid,zero_qualified
LEVELS=[(f'uniform{i:03d}',i/1000,i/1000) for i in range(325,-1,-25)]
PRIOR_COURSE_UPDATES=2
PRIOR_UPDATES=4
BUDGET=128-PRIOR_COURSE_UPDATES
WARMUP_STRENGTH=.35
WARMUP_UPDATES=2

def decision(rows,completed):
    if qualification(rows):return 'ADVANCE'
    if not learning_entry(rows):return 'ENTRY_FAILED'
    if completed>=BUDGET:return 'BUDGET_EXHAUSTED'
    return 'LEARN'

def chunk_size(completed,level_updates):
    assert 0<=completed<=BUDGET and level_updates>=0
    return min(2 if level_updates==0 else 8,BUDGET-completed)

def impact_only_failure(rows):
    if qualification(rows) or rows['native']['audits']['status']!='PASS':return False
    entries=[rows['native']]+rows['batch']
    if rows['native']['metrics']['worlds']!=45 or len(rows['batch'])!=3:return False
    if [r['repeat'] for r in rows['batch']]!=[1,2,3]:return False
    if any(r['metrics']['worlds']!=512 for r in rows['batch']):return False
    for r in entries:
        a=r['strict_admission']
        if not wrench_valid(r) or not a['pre_apex_constraint_margin']:return False
        for c in a['comparisons'].values():
            if not c.get('checks') or any(not v for k,v in c['checks'].items() if k!='worst_impact'):return False
    return True

def warmup_request(c,frozen):
    return dict(mode='train',warmup=True,contract=c,frozen_sha256=frozen,level_index=-1,
        before=WARMUP_STRENGTH,after=WARMUP_STRENGTH,checkpoint=c['source_checkpoint'],
        global_updates=0,reuse_preflight=False,updates=WARMUP_UPDATES,resume_optimizer=False)

def validate_request(r):
    assert r['mode'] in ('qualify','train')
    assert not r.get('reuse_preflight'),'No repeated failed evidence or unchanged actor replay'
    if r.get('warmup'):
        assert r==warmup_request(r['contract'],r['frozen_sha256'])
    else:
        index=r['level_index'];assert 0<=index<len(LEVELS)
        assert (r['before'],r['after'])==tuple(LEVELS[index][1:])
        assert WARMUP_UPDATES<=r['global_updates']<=BUDGET
    return True

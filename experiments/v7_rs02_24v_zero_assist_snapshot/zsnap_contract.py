"""User-requested zero-assistance diagnostic; no training or admission relaxation."""
from zsnap_gates import qualification,admission,learning_admission,learning_entry,learning_evidence,objective,metrics,zero_qualified,OBJECTIVE,ACTOR_LR,MAX_ACCEPTED_KL
LEVELS=[('uniform000',0.0,0.0)]
PRIOR_SHARED_UPDATES=54
BUDGET=0
EXPLORATION_OFFSET=40
def reuse_key(r):return None
def validate_request(r):
 assert r['mode']=='qualify' and r['role']=='zero_probe'
 assert r['level_index']==0 and r['before']==r['after']==0.0 and r['global_updates']==0
 assert r['reuse_preflight'] is False and r['reuse_source_evidence'] is False
 assert r['checkpoint']==r['contract']['source_checkpoint']
 assert r['contract']['no_training'] is True
 return True

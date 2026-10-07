"""CPU-only identity/freeze/old evidence check before the explicit zero probe."""
import argparse,json
from pathlib import Path
import zsnap_runtime as rt
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
 f,c=rt.verify(),rt.contract()
 import torch
 from slot_learning import Policy
 from zsnap_worker import actor_hash
 from midpoint_audit import check_qualification
 torch.set_num_threads(1)
 state=torch.load(c['source_checkpoint']['path'],map_location='cpu',weights_only=True)
 assert state['midpoint_frozen_sha256']==rt.PARENT_SHA and state['profile']==c['profile']==c['source_profile']
 assert state['global_update']==1 and state['before']==state['after']==.23125 and state['level_index']==0
 assert state['voltage_v']==24 and state['action_dim']==16 and all(bool(torch.isfinite(v).all()) for v in state['model_state_dict'].values())
 policy=Policy('cpu');policy.load_state_dict(state['model_state_dict'],strict=True);assert actor_hash(policy)==c['source_actor_hash']
 folder=Path(c['source_failed']['folder']);r=rt.parent.checked(folder);check_qualification(folder,r['request'],r)
 assert not r['qualified'] and r['actor_hash']==c['source_actor_hash']
 rt.write(a.output,dict(status='PREFLIGHT_VALIDATED',utc=rt.now(),frozen_sha256=f,contract=c,physical_trials=0,
  source_actor_hash=actor_hash(policy),source_checkpoint_strength=.23125,requested_strength=0.0,no_training=True,
  no_same_failed_actor_retest=True,final_gates_unchanged=True,learning_thresholds_unchanged=True))
 print(json.dumps(dict(status='PREFLIGHT_VALIDATED',frozen=f,physical_trials=0,source_actor=actor_hash(policy))))
if __name__=='__main__':main()

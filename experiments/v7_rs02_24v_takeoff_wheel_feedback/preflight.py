"""CPU source Torch, original failure and unchanged-chain checks; no new physics."""
import argparse,json
from pathlib import Path
import twpd_runtime as rt

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    f,c=rt.verify(),rt.contract()
    import torch,numpy as np
    from slot_learning import Policy
    from twpd_worker import actor_hash
    from twpd_trace_audit import slot_audit
    from twpd_wrench_audit import external_arrays
    torch.set_num_threads(1)
    state=torch.load(c['source_checkpoint']['path'],map_location='cpu',weights_only=True)
    assert state[c['source_frozen_field']]==c['source_parent_frozen_sha256']
    assert state['profile']==c['profile']==c['source_profile']
    assert state['global_update']==1 and state['before']==state['after']==.23125
    assert state['voltage_v']==24 and state['action_dim']==16
    assert all(bool(torch.isfinite(v).all()) for v in state['model_state_dict'].values())
    policy=Policy('cpu');policy.load_state_dict(state['model_state_dict'],strict=True)
    assert actor_hash(policy)==c['source_actor_hash']
    folder=Path(c['zero_failure']['folder']);trace=folder/'native_traces.npz'
    slots=slot_audit(trace,c['profile']);assert slots['status']=='NOT_REACHED' and slots['qualified'] is False
    with np.load(trace) as z:
        active=z['active'];assert active.shape[1]==45 and int(active.sum())==18871
        external=external_arrays(z,0.,0.)
        assert all(external[k]==0 for k in ['max_abs_linear_force_n','max_abs_body_torque_nm','max_abs_root_generalized_force'])
        assert not np.any(z['landing_control_enabled'])
    out=dict(status='PREFLIGHT_VALIDATED',utc=rt.now(),frozen_sha256=f,contract=c,physical_trials=0,
        source_actor_hash=actor_hash(policy),source_checkpoint_strength=.23125,requested_strength=0.,no_training=True,
        no_same_failed_actor_retest=True,new_controller_profile_verified=True,old_failure_preserved=True,
        final_gates_unchanged=True,learning_thresholds_unchanged=True,old_empty_slot=slots,old_external_wrench=external)
    rt.write(a.output,out);print(json.dumps(dict(status=out['status'],frozen=f,physical_trials=0,controller_id=c['controller_id'])))
if __name__=='__main__':main()

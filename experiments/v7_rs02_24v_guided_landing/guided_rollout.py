import guided_paths
from param_rollout import trial as parent_trial, compact, qualifies


def trial(*args,**kwargs):
    observation,action,reward,eligible,summary=parent_trial(*args,**kwargs)
    summary['reward_mode']='undiscounted guided cushion score: ten retained terms plus three guidance terms'
    summary['force_target_n']=300.
    return observation,action,reward,eligible,summary


def force(summary):
    return summary['mean_landing_metrics']['peak_force_n']


def better(candidate,current):
    return qualifies(candidate) and force(candidate)<force(current)-.25

"""Only the measured reward changes; controller, latency and physics are frozen."""
import guided_paths
from param_env import ParameterEnv
from guided_reward import GuidedReward


class GuidedEnv(ParameterEnv):
    def __init__(self,n,**kwargs):
        super().__init__(n,**kwargs)
        self.task.landing_reward=GuidedReward(n,self.device,self.cfg.physics_dt_s,
                                            float(self.m.body_subtreemass[self.base]))

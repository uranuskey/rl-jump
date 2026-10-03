"""Continue the frozen 24V launch through touchdown and stable recovery."""
import bootstrap
from dataclasses import replace
import torch
from environment import JumpEnv
from controlled_env import HeightEnv
from plan_contract import reference, HANDOFF
from jump_task import SUCCESS, FAILED, RECOVER

class LandingEnv(JumpEnv):
    def __init__(self,n,**kw):
        super().__init__(n,voltage=24,**kw)
        self.cfg=replace(self.cfg,episode_s=5.0)
        self.task.cfg=self.cfg
        self.landing_start=torch.zeros(n,device=self.device)
        self.landing_parameters=torch.tensor([.18,.22,.16,.14,.45],device=self.device).expand(n,-1).clone()
        self.feedback=torch.zeros(n,8,device=self.device)
    def reset(self,mask,**kw):
        if hasattr(self,'landing_start'): self.landing_start[mask]=0
        return super().reset(mask,**kw)
    def terminal_mask(self):
        return self.task.phase==FAILED
    def step(self,standing_actions,*,auto_reset=False):
        t=self.ticks*.0025
        state=reference(self.plan,t)
        apex=self.task.height_score.apex
        new=apex & (self.landing_start==0)
        self.landing_start[new]=t[new]
        h_air,duration,h_bottom,absorb,recover=self.landing_parameters.unbind(-1)
        u=((t-self.landing_start)/duration).clamp(0,1)
        smooth=u*u*(3-2*u)
        h=self.plan[:,3]+(h_air-self.plan[:,3])*smooth
        vh=(h_air-self.plan[:,3])*6*u*(1-u)/duration
        touch=self.task.touchdown_time>0
        age=(t-self.task.touchdown_time).clamp_min(0)
        a=(age/absorb).clamp(0,1)
        b=((age-absorb)/recover).clamp(0,1)
        ht=h_air+(h_bottom-h_air)*a*a*(3-2*a)+(.18-h_bottom)*b*b*(3-2*b)
        vt=(h_bottom-h_air)*6*a*(1-a)/absorb+(.18-h_bottom)*6*b*(1-b)/recover
        h=torch.where(touch,ht,h);vh=torch.where(touch,vt,vh)
        landing=torch.stack((h-.18,vh,torch.zeros_like(h),torch.ones_like(h)),1)
        state=torch.where(apex[:,None],landing,state)
        self.curve_state=torch.where((t>=HANDOFF)[:,None],state,self.curve_state)
        # Learned causal feedback uses current IMU/encoder/body-velocity estimates.
        q,v,gyro,linear,g,height,tilt=self.state()
        gains=4*torch.tanh(self.feedback)
        wheel=(gains[:,0]*g[:,0]*10+gains[:,1]*gyro[:,1]*.2+
               gains[:,2]*linear[:,0]*.5+gains[:,3]*v[:,4:].mean(1)*.05)
        pitch=gains[:,4]*g[:,0]*10+gains[:,5]*gyro[:,1]*.2
        roll=gains[:,6]*g[:,1]*10+gains[:,7]*gyro[:,0]*.2
        correction=torch.stack((pitch+roll,pitch-roll,pitch-roll,pitch+roll,wheel,wheel),1)
        actions=torch.where((t<HANDOFF)[:,None],standing_actions[:,:6],correction)
        return HeightEnv.step(self,actions,auto_reset=auto_reset)

"""Reward actual first-deceleration stroke, never a later cosmetic crouch."""
import guided_paths
import torch
from landing_reward import FlatLandingReward

SPEC = dict(version='guided_cushion_v1', force_target_n=300., early_window_s=.02,
            impact_multiplier=6., failure_multiplier=3., useful_stroke_reward=20.,
            stroke_margin=1.25, stroke_min_m=.04, stroke_max_m=.06,
            early_excess_penalty=20., target_bonus=20.)


def extra_terms(metrics, passed, mass):
    energy = metrics['incident_vertical_energy_j']
    desired = (SPEC['stroke_margin']*energy/(SPEC['force_target_n']-mass*9.81)).clamp(.04,.06)
    stroke = (metrics['first_stop_com_drop_m']/desired).clamp(0,1)
    return dict(useful_deceleration_stroke=20*passed*metrics['first_stop_observed']*stroke,
        first_contact_excess=-20*metrics['early_excess_integral_s']/.02,
        force_target_progress=20*passed*torch.exp(-(metrics['peak_force_n']-300).clamp_min(0)/50))


class GuidedReward(FlatLandingReward):
    def __init__(self,n,device,dt,mass_kg):
        super().__init__(n,device,dt,mass_kg)
        self.extra_names=('previous_com_z','touch_com_z','first_stop_com_drop_m',
                          'first_stop_elapsed_s','early_peak_force_n','early_excess_integral_s')
        for name in self.extra_names:
            setattr(self,name,torch.zeros(n,device=device))
        self.first_stop_seen=torch.zeros(n,dtype=torch.bool,device=device)

    def reset(self,mask):
        super().reset(mask)
        for name in self.extra_names:
            getattr(self,name)[mask]=0
        self.first_stop_seen[mask]=False

    def update(self,x,before,after,active,time,apex,recovery_ticks):
        total=x['wheel_force_n'].clamp_min(0).sum(1)
        first=active & ~self.touched & apex & (before==2) & (x['wheel_force_n'].max(1).values>=1)
        self.touch_com_z=torch.where(first,self.previous_com_z,self.touch_com_z)
        super().update(x,before,after,active,time,apex,recovery_ticks)
        age=time-self.touch_time
        decelerating=active & self.touched & ~self.first_stop_seen & (age<=.25)
        drop=(self.touch_com_z-x['com_z_m']).clamp_min(0)
        self.first_stop_com_drop_m=torch.where(decelerating,
            torch.maximum(self.first_stop_com_drop_m,drop),self.first_stop_com_drop_m)
        stopped=decelerating & (x['com_vz_mps']>=-.05)
        self.first_stop_elapsed_s=torch.where(stopped,age,self.first_stop_elapsed_s)
        self.first_stop_seen |= stopped
        early=active & self.touched & (age<=SPEC['early_window_s'])
        self.early_peak_force_n=torch.where(early,torch.maximum(self.early_peak_force_n,total),self.early_peak_force_n)
        self.early_excess_integral_s+=early*((total-300).clamp_min(0)/300).square().clamp(max=9)*self.dt
        self.previous_com_z=torch.where(active,x['com_z_m'],self.previous_com_z)

    def metrics(self):
        result=super().metrics()
        result.update({name:getattr(self,name) for name in self.extra_names[2:]})
        result['first_stop_observed']=self.first_stop_seen.float()
        return result

    def score(self,task,ticks):
        _,passed,retained=super().score(task,ticks)
        terms=dict(self.last_terms)
        terms['impact']*=SPEC['impact_multiplier']
        terms['failure']*=SPEC['failure_multiplier']
        terms['insufficient_height']*=SPEC['failure_multiplier']
        terms.update(extra_terms(self.metrics(),passed,self.mass_kg))
        self.last_terms=terms
        return sum(terms.values()),passed,retained

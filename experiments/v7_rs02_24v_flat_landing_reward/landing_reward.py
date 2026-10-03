"""Flat-ground cushion-and-settle reward measured at every physical tick.

Normalization constants are research reward scales, not hardware safety limits.
Height is a prerequisite, not a benefit to trade against a softer low jump.
"""
import math
import torch

SPEC=dict(version='flat_cushion_v1',assist_strength=.625,terrain='flat',
    min_wheel_m=.09,min_com_rise_m=.115,impact_window_s=.25,motion_window_s=1.,
    finish_reward=100.,stable_progress_reward=30.,settle_time_reward=10.,absorption_reward=5.,
    impact_penalty=15.,rebound_penalty=25.,motion_penalty=10.,asymmetry_penalty=5.,
    invalid_height_penalty=100.,failure_penalty=100.,force_soft_scale_bw=2.,force_range_bw=4.,
    rebound_scale_mps=.25,minimum_compression_for_absorption_m=.002)

class FlatLandingReward:
    def __init__(self,n,device,dt,mass_kg):
        self.dt=dt;self.mass_kg=mass_kg;self.weight_n=mass_kg*9.81
        self.names=('touch_time','pre_touch_vz','touch_leg_height','min_leg_height','peak_force_n',
            'negative_leg_work_j','impulse_sum_ns','impulse_difference_ns','peak_rebound_vz',
            'motion_integral','motion_duration','best_stable_s','previous_com_vz')
        for name in self.names:setattr(self,name,torch.zeros(n,device=device))
        self.touched=torch.zeros(n,dtype=torch.bool,device=device)
        self.last_terms={}
    def reset(self,mask):
        for name in self.names:getattr(self,name)[mask]=0
        self.touched[mask]=False
    def update(self,x,before,after,active,time,apex,recovery_ticks):
        force=x['wheel_force_n'].clamp_min(0);total=force.sum(1)
        first=active & ~self.touched & apex & (before==2) & (force.max(1).values>=1.)
        height=x['leg_height_m'].mean(1)
        self.touch_time[first]=time[first];self.pre_touch_vz[first]=self.previous_com_vz[first]
        self.touch_leg_height[first]=height[first];self.min_leg_height[first]=height[first]
        self.touched |= first
        age=time-self.touch_time
        post=active & self.touched
        cushion=post & (age<=SPEC['impact_window_s'])
        # Force peak continues over the complete landing, so shifting an impact
        # outside the0.25s absorption window does not hide the hard landing.
        self.peak_force_n=torch.where(post,torch.maximum(self.peak_force_n,total),self.peak_force_n)
        self.min_leg_height=torch.where(cushion,torch.minimum(self.min_leg_height,height),self.min_leg_height)
        negative_power=(-x['motor_torque_nm']*x['motor_speed_rad_s']).clamp_min(0).sum(1)
        contact=cushion & (total>=1.)
        self.negative_leg_work_j += contact*negative_power*self.dt
        self.impulse_sum_ns += cushion*total*self.dt
        self.impulse_difference_ns += cushion*(force[:,0]-force[:,1]).abs()*self.dt
        # A controlled rise while still supported is recovery, not a rebound.
        unsupported=post & (force.max(1).values<=.5)
        self.peak_rebound_vz=torch.where(unsupported,torch.maximum(self.peak_rebound_vz,x['com_vz_mps'].clamp_min(0)),self.peak_rebound_vz)
        moving=(x['tilt_rad']/math.radians(5)).square()
        moving+=(x['gyro_norm_rad_s']/.5).square()
        moving+=(torch.linalg.vector_norm(x['body_vxy_mps'],dim=1)/.05).square()
        moving+=(x['body_vz_mps']/.05).square()
        motion_window=post & (age<=SPEC['motion_window_s'])
        self.motion_integral+=motion_window*(moving/4).clamp(max=9)*self.dt
        self.motion_duration+=motion_window*self.dt
        self.best_stable_s=torch.where(post,torch.maximum(self.best_stable_s,recovery_ticks*self.dt),self.best_stable_s)
        self.previous_com_vz=torch.where(active,x['com_vz_mps'],self.previous_com_vz)
    def metrics(self):
        compression=(self.touch_leg_height-self.min_leg_height).clamp_min(0)
        incident=.5*self.mass_kg*self.pre_touch_vz.clamp_max(0).square()
        absorption=(self.negative_leg_work_j/incident.clamp_min(.05)).clamp(0,1)
        absorption*=compression>=SPEC['minimum_compression_for_absorption_m']
        return dict(peak_force_n=self.peak_force_n,peak_force_bodyweights=self.peak_force_n/self.weight_n,
            peak_rebound_vz_mps=self.peak_rebound_vz,compression_m=compression,
            negative_leg_work_j=self.negative_leg_work_j,incident_vertical_energy_j=incident,
            absorption_fraction_proxy=absorption,
            impulse_asymmetry=(self.impulse_difference_ns/self.impulse_sum_ns.clamp_min(1e-6)).clamp(0,1),
            mean_settling_motion=self.motion_integral/self.motion_duration.clamp_min(self.dt),
            best_continuous_stable_s=self.best_stable_s)
    def score(self,task,ticks):
        m=self.metrics()
        retained=task.height_score.apex & (task.height_score.peak>=SPEC['min_com_rise_m']) & (task.peak_clearance_m>=SPEC['min_wheel_m'])
        failed=task.phase==5
        valid=retained & self.touched & ~failed
        passed=valid & (task.phase==4) & (ticks>=2000)
        settling=(task.success_time-self.touch_time).clamp_min(0)
        terms=dict(finish=100*passed.float(),stable_progress=30*valid*self.best_stable_s.clamp(max=1),
            quick_settle=10*passed*torch.exp(-settling/.8),
            absorption=5*valid*m['absorption_fraction_proxy'],
            impact=-15*((m['peak_force_bodyweights']-2).clamp_min(0)/4).square().clamp(max=4),
            rebound=-25*(m['peak_rebound_vz_mps']/.25).square().clamp(max=4),
            settling_motion=-10*m['mean_settling_motion'].clamp(max=3),
            uneven_impulse=-5*m['impulse_asymmetry'],
            insufficient_height=-100*(~retained).float(),failure=-100*failed.float())
        self.last_terms=terms
        return sum(terms.values()),passed,retained

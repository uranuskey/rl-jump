"""CPU behavioral tests: rank landings and reject concrete reward shortcuts."""
import copy
from types import SimpleNamespace
import torch
from landing_reward import FlatLandingReward

def task(n=1):
    return SimpleNamespace(height_score=SimpleNamespace(apex=torch.ones(n,dtype=torch.bool),peak=torch.full((n,),.1198)),
        peak_clearance_m=torch.full((n,),.0925),phase=torch.full((n,),4,dtype=torch.long),success_time=torch.full((n,),2.5))
def metric():
    m=FlatLandingReward(1,'cpu',.0025,7.228699)
    m.touched[:]=True;m.touch_time[:]=1.3;m.pre_touch_vz[:]=-1.3
    m.touch_leg_height[:]=.19;m.min_leg_height[:]=.16
    m.peak_force_n[:]=2*m.weight_n;m.negative_leg_work_j[:]=4
    m.impulse_sum_ns[:]=30;m.motion_duration[:]=1;m.best_stable_s[:]=1
    return m
def value(m,t=None,ticks=2000):return float(m.score(t or task(),torch.tensor([ticks]))[0])
def main():
    checks=[]
    soft=metric();hard=copy.deepcopy(soft);hard.peak_force_n[:]=6*hard.weight_n
    assert value(soft)>value(hard);checks.append('same-height soft impact outranks hard impact')
    bounce=copy.deepcopy(soft);bounce.peak_rebound_vz[:]=.4
    assert value(soft)>value(bounce);checks.append('same-height rebound loses reward')
    unstable=copy.deepcopy(soft);unstable.motion_integral[:]=3;unstable.best_stable_s[:]=.1
    assert value(soft)>value(unstable);checks.append('residual motion and short stable hold lose reward')
    unbraked=copy.deepcopy(soft);unbraked.negative_leg_work_j.zero_()
    assert value(soft)>value(unbraked)
    rigid=copy.deepcopy(soft);rigid.min_leg_height.copy_(rigid.touch_leg_height)
    assert value(rigid)==value(unbraked);checks.append('absorption credit requires measured compression and braking work')
    low=task();low.peak_clearance_m[:]=.05
    assert value(soft,low)<0
    tuck=task();tuck.height_score.peak[:]=.05;tuck.peak_clearance_m[:]=.15
    assert value(soft,tuck)<0
    standing=task();standing.height_score.apex[:]=False
    assert value(soft,standing)<0;checks.append('low jump, wheel tuck and no jump receive no landing credit')
    failed=task();failed.phase[:]=5
    assert value(soft,failed)<0 and soft.last_terms['finish'].item()==0
    assert value(soft,ticks=1200)<value(soft);checks.append('late failure and early end cannot claim complete success')
    fast=task();slow=task();slow.success_time[:]=3.7
    assert value(soft,fast)>value(soft,slow);checks.append('faster stable recovery is preferred')
    m=FlatLandingReward(2,'cpu',.0025,7.228699)
    def sample(force=35.,vz=0.):
        return dict(wheel_force_n=torch.full((2,2),force),leg_height_m=torch.full((2,2),.18),
            motor_torque_nm=torch.full((2,4),-2.),motor_speed_rad_s=torch.ones(2,4),com_vz_mps=torch.full((2,),vz),
            tilt_rad=torch.zeros(2),gyro_norm_rad_s=torch.zeros(2),body_vxy_mps=torch.zeros(2,2),body_vz_mps=torch.zeros(2))
    m.previous_com_vz[:]=-1.2
    m.update(sample(),torch.full((2,),2),torch.full((2,),3),torch.ones(2,dtype=torch.bool),torch.full((2,),1.3),torch.ones(2,dtype=torch.bool),torch.zeros(2,dtype=torch.long))
    # Spike is only one2.5ms tick and must not disappear under50Hz sampling.
    m.update(sample(250.),torch.full((2,),3),torch.full((2,),3),torch.ones(2,dtype=torch.bool),torch.full((2,),1.3025),torch.ones(2,dtype=torch.bool),torch.zeros(2,dtype=torch.long))
    assert bool((m.peak_force_n==500).all());checks.append('one-physics-tick force spike is retained')
    m.update(sample(35.,.4),torch.full((2,),3),torch.full((2,),3),torch.ones(2,dtype=torch.bool),torch.full((2,),1.4),torch.ones(2,dtype=torch.bool),torch.ones(2,dtype=torch.long))
    assert bool((m.peak_rebound_vz==0).all());checks.append('grounded recovery rise is not classified as rebound')
    before={k:getattr(m,k).clone() for k in m.names}
    m.update(sample(1000.,2.),torch.full((2,),3),torch.full((2,),5),torch.zeros(2,dtype=torch.bool),torch.full((2,),2.),torch.ones(2,dtype=torch.bool),torch.ones(2,dtype=torch.long))
    assert all(torch.equal(getattr(m,k),v) for k,v in before.items());checks.append('paused worlds cannot accumulate reward metrics')
    m.reset(torch.tensor([True,False]))
    assert all(getattr(m,k)[0]==0 and getattr(m,k)[1]==v[1] for k,v in before.items());checks.append('selective reset preserves neighbouring world')
    return checks
if __name__=='__main__':print(main())

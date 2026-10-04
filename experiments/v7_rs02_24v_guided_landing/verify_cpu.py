"""Behavioral checks for impact guidance, shortcut resistance and actor reuse."""
import copy
import json
import guided_paths
import torch
from guided_reward import GuidedReward
from guided_learning import Policy,PPO,apply_offset,set_exploration
from guided_search import PROPOSALS,Proposals
from guided_runtime import verify
from test_reward import task,metric,value,main as old_checks


def main():
    torch.set_num_threads(1)
    torch.manual_seed(104042)
    checks=list(old_checks())
    def guided():
        m=GuidedReward(1,'cpu',.0025,7.228699)
        old=metric()
        for name in old.names:
            getattr(m,name).copy_(getattr(old,name))
        m.touched[:]=True
        m.first_stop_seen[:]=True
        m.first_stop_com_drop_m[:]=.03
        return m
    soft,hard=guided(),guided()
    soft.peak_force_n[:]=300;hard.peak_force_n[:]=427
    assert value(soft)>value(hard)+40
    checks.append('300 N stable landing is strongly preferred to 427 N at the same height')
    deeper=copy.deepcopy(hard);deeper.first_stop_com_drop_m[:]=.045
    assert value(deeper)>value(hard)
    saturated=copy.deepcopy(deeper);saturated.first_stop_com_drop_m[:]=.20
    topped=copy.deepcopy(deeper);topped.first_stop_com_drop_m[:]=.06
    assert value(saturated)==value(topped)
    failed=task();failed.phase[:]=5
    assert value(soft,failed)<value(hard)-200
    low=task();low.peak_clearance_m[:]=.05
    assert value(soft,low)<0
    checks.append('stroke guidance is bounded; failure or height loss cannot buy a softer score')
    m=GuidedReward(2,'cpu',.0025,7.228699)
    def sample(force,z,vz):
        return dict(wheel_force_n=torch.full((2,2),force),leg_height_m=torch.full((2,2),.18),
            motor_torque_nm=torch.full((2,4),-2.),motor_speed_rad_s=torch.ones(2,4),
            com_z_m=torch.full((2,),z),com_vz_mps=torch.full((2,),vz),
            tilt_rad=torch.zeros(2),gyro_norm_rad_s=torch.zeros(2),
            body_vxy_mps=torch.zeros(2,2),body_vz_mps=torch.zeros(2))
    active=torch.ones(2,dtype=torch.bool)
    def tick(force,z,vz,time,phase=3,mask=active):
        m.update(sample(force,z,vz),torch.full((2,),phase),torch.full((2,),3),mask,
                 torch.full((2,),time),active,torch.zeros(2,dtype=torch.long))
    m.previous_com_z[:]=.30;m.previous_com_vz[:]=-1.45
    tick(210.,.296,-1.3,1.3,2)
    tick(250.,.29,-1.,1.3025)
    assert (m.peak_force_n==500).all() and (m.early_peak_force_n==500).all()
    tick(100.,.27,-.02,1.35)
    assert torch.allclose(m.first_stop_com_drop_m,torch.full((2,),.03),atol=1e-6)
    tick(35.,.22,-.3,1.5)
    assert torch.allclose(m.first_stop_com_drop_m,torch.full((2,),.03),atol=1e-6)
    checks.append('single-tick contact spikes retained; crouching after first arrest gets no stroke credit')
    before={name:getattr(m,name).clone() for name in m.names+m.extra_names+('first_stop_seen',)}
    tick(1000.,.1,1.,1.6,mask=~active)
    assert all(torch.equal(getattr(m,name),v) for name,v in before.items())
    m.reset(torch.tensor([True,False]))
    assert all(getattr(m,name)[0]==0 and getattr(m,name)[1]==v[1] for name,v in before.items())
    checks.append('extra 400 Hz metrics respect inactive worlds and selective reset')
    policy=Policy();obs=torch.randn(512,17)
    original=policy.actor(obs).detach().clone()
    profile=PROPOSALS[9]
    offset=list(profile[1])+[0.]*8
    proposed=Proposals(policy,[profile]*11,512).distribution(obs).mean.detach()
    apply_offset(policy,offset)
    assert torch.allclose(policy.actor(obs),proposed,atol=1e-6)
    assert torch.equal(policy.actor(obs)[:,5:],original[:,5:])
    set_exploration(policy,0)
    ppo=PPO(policy)
    with torch.no_grad():actions=policy.distribution(obs).sample()
    before=copy.deepcopy(policy.actor.state_dict())
    stats=ppo.update(obs,actions,torch.linspace(-300,150,512),torch.ones(512,dtype=torch.bool))
    assert stats['actor_steps']>0 and any(not torch.equal(v,before[k]) for k,v in policy.actor.state_dict().items())
    checks.append('proposal bias transfer is exact; subsequent PPO uses fresh on-policy samples')
    assert len(PROPOSALS)==21
    print(json.dumps(dict(status='PASS_OFFLINE_ONLY',frozen_sha256=verify(),checks=checks),indent=2))


if __name__=='__main__':
    main()

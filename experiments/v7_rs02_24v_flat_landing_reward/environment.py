"""Independent symmetric jump controller, retaining only the standing prefix."""
import bootstrap
from dataclasses import dataclass
import torch
from controlled_env import HeightEnv
from settings import JumpConfig
from jump_task import ProbeTask,FAILED,REASONS
from height_reward import FlightHeight
from landing_reward import FlatLandingReward
from seed_environment import reset_cases
from velocity_contract import reference_motor_velocity
from voltage_curve import no_load
from plan_contract import HANDOFF,decode,initial_raw,reference

@dataclass(frozen=True)
class Config(JumpConfig):
    episode_s:float=2.8
    target_height_m:float=.15
    maximum_height_m:float|None=None
    takeoff_deadline_s:float=2.
    leg_height_low_m:float=.09
    leg_height_high_m:float=.23
    leg_motor_torque_limit_nm:float=17.
    leg_motor_speed_limit_rad_s:float=no_load(24)
    def __post_init__(self):
        assert self.maximum_height_m is None and self.minimum_visible_height_m==.01
        assert self.leg_motor_torque_limit_nm==17 and self.leg_height_low_m==.09
    @property
    def height_bounds(self):return .01,float('inf')

class LearningTask(ProbeTask):
    def __init__(self,n,device,config,mass_kg=7.228699):
        self.height_score=FlightHeight(n,device)
        self.landing_reward=FlatLandingReward(n,device,config.physics_dt_s,mass_kg)
        super().__init__(n,device,config)
    def reset(self,mask,xy,height):
        super().reset(mask,xy,height)
        self.height_score.reset(mask)
        self.landing_reward.reset(mask)
    def update(self,x,active=None):
        if active is None:active=torch.ones_like(self.paired)
        before=self.phase.clone()
        super().update(x,active)
        time=self.steps*self.cfg.physics_dt_s
        self.height_score.update(x,before,self.phase,active,time)
        self.landing_reward.update(x,before,self.phase,active,time,self.height_score.apex,self.recovery_ticks)
        # PPO receives the episodic landing score, not the former height increments.
        return torch.zeros_like(time)

class JumpEnv(HeightEnv):
    def __init__(self,n,voltage=24,**kwargs):
        self.voltage=voltage
        super().__init__(n,config=Config(leg_motor_speed_limit_rad_s=no_load(voltage)),stage='jump',**kwargs)
        self.plan=decode(initial_raw(self.device).expand(n,-1)).clone()
        self.task=LearningTask(n,self.device,self.cfg,float(self.m.body_subtreemass[self.base]))
        self.terminal={}
        self.seen_terminal=torch.zeros(n,dtype=torch.bool,device=self.device)
        self.reset(torch.ones_like(self.paused))
    def terminal_mask(self):
        end=self.task.phase==FAILED
        if hasattr(self.task,'height_score'):end=end|self.task.height_score.apex
        return end
    def capture_terminal(self,x,mask,phase_before):
        if not hasattr(self,'seen_terminal'):return
        new=mask & ~self.seen_terminal
        fields=('motor_speed_rad_s','wheel_speed_rad_s','motor_torque_nm','motor_envelope_nm',
                'motor_peak_exposure_s','leg_height_m','wheel_force_n','com_vz_mps','tilt_rad','saturated',
                'mimic_q_error_rad','mimic_v_error_rad_s','nonwheel_force_n','self_contact')
        values={k:x[k] for k in fields}
        values.update(ticks=self.ticks,phase=self.task.phase,phase_before=phase_before,reason=self.task.reason,
                      leg_overspeed=x['motor_speed_rad_s'].abs()>self.cfg.leg_motor_speed_limit_rad_s,
                      wheel_overspeed=x['wheel_speed_rad_s'].abs()>self.cfg.wheel_speed_limit_rad_s,
                      reward_apex=self.task.height_score.apex)
        for k,v in values.items():
            if k not in self.terminal:self.terminal[k]=torch.zeros_like(v)
            self.terminal[k][new]=v[new]
        self.seen_terminal |= new
    def reset(self,mask,**kwargs):
        if kwargs.get('poses') is None and hasattr(self,'training_poses'):
            kwargs.update(poses=self.training_poses,velocities=self.training_velocities)
        if hasattr(self,'plan'):self.plan[mask]=decode(initial_raw(self.device))
        if hasattr(self,'seen_terminal'):
            self.seen_terminal[mask]=False
            for v in self.terminal.values():v[mask]=0
        return super().reset(mask,**kwargs)
    def command(self):
        age=((self.ticks-200).clamp_min(0)*.0025/10).clamp(max=1)
        height=.18+self.curve_state[:,0]
        return torch.stack((torch.full_like(age,3.),(self.ticks>=200).float(),age,height/.25),1)
    def push_motor_velocity(self,height):
        return self.curve_state[:,3,None]*reference_motor_velocity(height,self.curve_state[:,1])
    def push_force(self):return self.curve_state[:,2]
    def step(self,standing_actions,*,auto_reset=False):
        # The teacher is completely absent after handoff; wheel speed target is zero.
        if hasattr(self,'plan'):
            state=reference(self.plan,self.ticks*.0025)
            self.curve_state=torch.where((self.ticks*.0025>=HANDOFF)[:,None],state,self.curve_state)
        actions=torch.where((self.ticks*.0025<HANDOFF)[:,None],standing_actions[:,:6],torch.zeros_like(standing_actions[:,:6]))
        return super().step(actions,auto_reset=auto_reset)
    def plan_observation(self):
        q,v,gyro,linear,g,height,tilt=self.state()
        from height_contract import to_motor
        return torch.cat((g,.2*gyro,to_motor(q[:,:4])/3.,.1*to_motor(v[:,:4]),.05*v[:,4:],self.assist_strength[:,None]),1)

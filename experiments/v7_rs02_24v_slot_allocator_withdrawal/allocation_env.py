"""Isolated residual-window revision; same physics, FIFO and pre-apex actions."""
import allocation_paths
import torch
from clearance_env import ClearanceEnv
from compliant_control import tick
from compliant_servo import payload
from velocity_contract import reference_motor_velocity
from precontact_control import modulate
from allocation_control import feedback,table
class AllocationEnv(ClearanceEnv):
    def __init__(self,n,profiles,**kw):
        super().__init__(n,profiles,**kw)
        self.allocation_config=table(profiles,n,self.device)

    def control_tick(self,held):
        q,vel,gyro,linear,gravity,height,_=self.state()
        args=dict(ticks=self.ticks,apex=self.task.height_score.apex,terminal=self.terminal_mask(),
            launch_height=.18+.03*held[:,6],touchdown=self.task.touchdown_time,leg_height=height,
            incident_vz=self.task.landing_reward.pre_touch_vz,gravity=gravity,gyro=gyro,
            linear=linear,wheel_speed=vel[:,4:],mass=self.mass)
        def request(raw):
            c=tick(raw,self.control_state,**args)
            c,diag,progress=modulate(c,self.config,self.pre_state,clearance=self.clearance,
                surface_z=self.world_floor_positions[:,self.ground,2],com_vz=self.com_v[:,2],
                touchdown=self.task.touchdown_time,leg_height=height,protect_landing=True)
            bounded,slot=feedback(c,self.allocation_config,q,vel);diag.update(slot)
            mv=reference_motor_velocity(c['height'],c['velocity'])
            new=payload(bounded,c['height'],mv,c['force'],c['kp'],c['kd'],c['enabled'])
            return c,torch.where(c['enabled'][:,None],new,held),diag,progress
        c,requested,diag,progress=request(self.parameter_action)
        if self.proof:
            _,alternate,_,_=request(self.parameter_action+1.1)
            before=~self.task.height_score.apex
            assert torch.equal(requested[before],alternate[before])
            assert torch.equal(requested[before,:12],held[before,:12])
            self.proof_samples+=int(before.sum())
            self.shadow_fifo.delay_steps.copy_(self.fifo.delay_steps)
            self.expected_prefix=self.shadow_fifo.push(held[:,:12]).clone()
            self.prefix_mask=before.clone()
            if not bool(before.any()):self.proof=False
        self.control_state=c['state'];self.pre_state=progress;self.pre_diagnostics=diag
        self.gate_tick.copy_(self.control_state['gate_tick']);self.control_gate.copy_(c['enabled'])
        self.effective_action.copy_(torch.where(c['enabled'][:,None],self.parameter_action,torch.zeros_like(self.parameter_action)))
        state=torch.stack((c['height']-.18,c['velocity'],c['force'],torch.ones_like(c['height'])),1)
        self.curve_state.copy_(torch.where(c['enabled'][:,None],state,self.curve_state))
        return requested

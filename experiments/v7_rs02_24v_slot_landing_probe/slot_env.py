"""Isolated controller over the unchanged 24 V / assisted physical environment."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'v7_rs02_24v_compliant_landing_v3'
PRE=HERE.parent/'v7_rs02_24v_precontact_landing_probe'
sys.path.insert(0,str(PARENT));import compliant_paths
sys.path.insert(0,str(PRE))
import torch
import warp as wp
from compliant_env import CompliantEnv
from compliant_control import tick
from compliant_servo import payload
from velocity_contract import reference_motor_velocity
from precontact_control import modulate,table as pre_table,initial
from slot_control import feedback,table as slot_table


class SlotEnv(CompliantEnv):
    def __init__(self,n,profiles,*,height_m=0.,**kw):
        self.pre_diagnostics=None;self.policy_calls=0;self.terrain_receipt=None
        self.height_m=height_m
        super().__init__(n,**kw)
        self.config=pre_table(profiles,self.device)
        if len(profiles)==1:self.config=self.config[:1].expand(n,-1).clone()
        assert self.config.shape[0]==n
        self.slot_config=slot_table(profiles,n,self.device)
        self.pre_state=initial(n,self.device)
        self.ground=self.m.geom('ground').id
        self.floor_positions=wp.to_torch(self.gm.geom_pos)
        self.world_floor_positions=wp.to_torch(self.gd.geom_xpos)

    def set_floor(self,height):
        self.m.geom_pos[self.ground,2]=height
        if self.floor_positions.ndim==3:self.floor_positions[:,self.ground,2]=height
        else:self.floor_positions[self.ground,2]=height
        self.world_floor_positions[:,self.ground,2]=height

    def reset(self,mask,**kw):
        if hasattr(self,'pre_state'):
            for value in self.pre_state.values():value[mask]=0
            self.set_floor(0.)
        self.policy_calls=0
        return super().reset(mask,**kw)

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
            bounded,slot=feedback(c,self.slot_config,q,vel);diag.update(slot)
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

    def sensors(self,motor=None):
        x=super().sensors(motor)
        if self.pre_diagnostics is not None:x.update({k:v.clone() for k,v in self.pre_diagnostics.items()})
        h=(self.world_floor_positions[:,self.ground,2].clone() if hasattr(self,'world_floor_positions') else torch.zeros(self.n,device=self.device))
        x.update(landing_surface_z_m=h,clearance_above_surface_m=x['wheel_clearance_m']-h[:,None])
        return x

    def step(self,standing_actions,**kw):
        if self.height_m and self.policy_calls==63:
            assert bool(self.task.height_score.apex.all()) and float(self.support.max())<=.5
            assert float(self.clearance.min())>self.height_m+.03
            q,v=self.q.clone(),self.v.clone();self.set_floor(self.height_m)
            self.terrain_receipt=dict(height_m=self.height_m,at_s=1.26,qv_unchanged=torch.equal(q,self.q) and torch.equal(v,self.v))
        self.policy_calls+=1
        out=super().step(standing_actions,**kw)
        if self.height_m and self.policy_calls==64:
            e=float((self.world_floor_positions[:,self.ground,2]-self.height_m).abs().max())
            self.terrain_receipt.update(actual_plane_error_m=e,first_step_force_n=float(self.support.max()))
            assert e<1e-7 and float(self.support.max())<=.5
        return out

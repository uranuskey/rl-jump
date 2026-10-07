"""Wheel targets before apex; original launch, FIFO, servo and actuator map remain."""
import twpd_runtime
import torch
from zsnap_env import BalancedAssistEnv as Base
from twpd_control import wheel_feedback

class BalancedAssistEnv(Base):
    def control_tick(self,held):
        _,_,gyro,_,gravity,_,_=self.state()
        pitch=torch.atan2(gravity[:,0],-gravity[:,2])
        terminal=self.terminal_mask();apex=self.task.height_score.apex
        action,enabled,ramp=wheel_feedback(pitch,gyro[:,1],self.ticks,apex,terminal,torch)
        revised=held.clone()
        revised[:,4:6]=torch.where(enabled[:,None],action[:,None].expand(-1,2),held[:,4:6])
        self.twpd_diag=dict(twpd_pitch=pitch.clone(),twpd_gyro=gyro[:,1].clone(),twpd_ticks=self.ticks.clone(),
            twpd_q=self.q.clone(),twpd_v=self.v.clone(),twpd_apex=apex.clone(),twpd_terminal=terminal.clone(),
            twpd_action=action.clone(),twpd_enabled=enabled.clone(),twpd_ramp=ramp.clone(),twpd_held=held.clone(),
            twpd_revised=revised.clone())
        # Existing proof checks that landing actor cannot change this revised
        # takeoff prefix, and all 15 channels still traverse the original FIFO.
        return super().control_tick(revised)

    def sensors(self,motor=None):
        out=super().sensors(motor)
        if motor is not None and self.record and hasattr(self,'twpd_diag'):
            out.update(self.twpd_diag)
        return out

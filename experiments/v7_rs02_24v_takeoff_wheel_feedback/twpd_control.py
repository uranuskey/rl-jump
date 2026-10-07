"""A bounded wheel-motor feedback hypothesis; no generalized/body force output."""
PROFILE=dict(name='wheel_pitch_pd_14_1',kp_nm_rad=14.,kd_nm_s_rad=1.,handoff_s=.6,
             ramp_s=.05,action_bound=1.,wheel_servo_nm_per_action=2.)

def wheel_feedback(pitch,gyro_y,ticks,apex,terminal,xp):
    assert bool(xp.isfinite(pitch).all()) and bool(xp.isfinite(gyro_y).all()) and bool(xp.isfinite(ticks).all())
    t=ticks*.0025
    enabled=(t>=PROFILE['handoff_s']) & ~apex & ~terminal
    ramp=xp.clip((t-PROFILE['handoff_s'])/PROFILE['ramp_s'],0.,1.)
    raw=(PROFILE['kp_nm_rad']*pitch+PROFILE['kd_nm_s_rad']*gyro_y)/PROFILE['wheel_servo_nm_per_action']
    action=xp.where(enabled,xp.clip(raw,-PROFILE['action_bound'],PROFILE['action_bound'])*ramp,raw*0.)
    return action,enabled,ramp

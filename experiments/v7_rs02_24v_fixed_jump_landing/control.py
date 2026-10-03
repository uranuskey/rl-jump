"""Causal 50 Hz landing actions, gated separately for each world's COM apex."""
import torch

DT = .02
HEIGHT_LOW, HEIGHT_HIGH = .095, .225
HEIGHT_VELOCITY_SCALE = 1.0
ACTION_DIM = 7
ACTOR_DIM, CRITIC_DIM = 157, 189
ACTION_STD = (.10, .12, .12, .12, .12, .10, .10)


def reference(plan, time, start, touchdown):
    # Zero-action reference is exactly the previously validated landing controller.
    h_air, duration, h_bottom, absorb, recover = .18, .22, .16, .14, .45
    u = ((time-start)/duration).clamp(0, 1)
    h = plan[:, 3]+(h_air-plan[:, 3])*u*u*(3-2*u)
    velocity = (h_air-plan[:, 3])*6*u*(1-u)/duration
    age = (time-touchdown).clamp_min(0)
    a = (age/absorb).clamp(0, 1)
    b = ((age-absorb)/recover).clamp(0, 1)
    ht = h_air+(h_bottom-h_air)*a*a*(3-2*a)+(.18-h_bottom)*b*b*(3-2*b)
    vt = (h_bottom-h_air)*6*a*(1-a)/absorb+(.18-h_bottom)*6*b*(1-b)/recover
    return torch.where(touchdown>0, ht, h), torch.where(touchdown>0, vt, velocity)


def advance(raw, enabled, offset, base_height, base_velocity):
    if raw.shape != (len(offset), ACTION_DIM) or not bool(torch.isfinite(raw).all()):
        raise ValueError('Expected finite [worlds, 7] continuous landing actions')
    bounded = torch.tanh(raw)
    wanted = offset+DT*HEIGHT_VELOCITY_SCALE*bounded[:, 0]
    lo, hi = HEIGHT_LOW-base_height, HEIGHT_HIGH-base_height
    candidate = torch.maximum(torch.minimum(wanted, hi), lo)
    new_offset = torch.where(enabled, candidate, offset)
    height = base_height+new_offset
    velocity = base_velocity+(new_offset-offset)/DT
    velocity = torch.where(((height<=HEIGHT_LOW+1e-7)&(velocity<0)) |
                           ((height>=HEIGHT_HIGH-1e-7)&(velocity>0)), 0., velocity)
    # HeightEnv applies tanh exactly once to these six motor/wheel latents.
    motor_latent = torch.where(enabled[:, None], raw[:, 1:], torch.zeros_like(raw[:, 1:]))
    effective = torch.where(enabled[:, None], bounded, torch.zeros_like(bounded))
    return new_offset, height, velocity, motor_latent, effective


def features(env):
    time = env.ticks*.0025
    touched = env.task.touchdown_time>0
    apex_age = torch.where(env.gate_tick>=0, (time-env.landing_start).clamp(0, 4)/4, 0.)
    extra = torch.stack((apex_age,
                         torch.where(touched, (time-env.task.touchdown_time).clamp(0, 4)/4, 0),
                         touched.float(), env.height_offset/.08, time/5), 1)
    extra = torch.cat((extra, env.previous_policy_action), 1)
    actor = torch.cat((env.obs, extra), 1)
    critic = torch.cat((env.critic_obs, extra), 1)
    assert actor.shape[1] == ACTOR_DIM and critic.shape[1] == CRITIC_DIM
    return actor, critic

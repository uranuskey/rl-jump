"""One 13D landing plan per episode; its effects start at each world's apex."""
import torch

ACTION_DIM, OBS_DIM = 13, 17
LOW = (.165, .12, .135, .08, .25)
HIGH = (.205, .35, .18, .25, .8)
INITIAL = (.18, .22, .16, .14, .45)
STD = (.05,)*5+(.035,)*8


def initial_raw(device='cpu'):
    low, high, initial = [torch.tensor(x, device=device) for x in (LOW, HIGH, INITIAL)]
    return torch.cat((torch.logit((initial-low)/(high-low)), torch.zeros(8, device=device)))


def decode(raw):
    low, high = [raw.new_tensor(x) for x in (LOW, HIGH)]
    return low+(high-low)*raw[:, :5].sigmoid(), 4*raw[:, 5:].tanh()


def reference(parameters, plan, time, start, touchdown):
    h_air, duration, h_bottom, absorb, recover = parameters.unbind(-1)
    u = ((time-start)/duration).clamp(0, 1)
    h = plan[:, 3]+(h_air-plan[:, 3])*u*u*(3-2*u)
    velocity = (h_air-plan[:, 3])*6*u*(1-u)/duration
    age = (time-touchdown).clamp_min(0)
    a = (age/absorb).clamp(0, 1)
    b = ((age-absorb)/recover).clamp(0, 1)
    ht = h_air+(h_bottom-h_air)*a*a*(3-2*a)+(.18-h_bottom)*b*b*(3-2*b)
    vt = (h_bottom-h_air)*6*a*(1-a)/absorb+(.18-h_bottom)*6*b*(1-b)/recover
    return torch.where(touchdown>0, ht, h), torch.where(touchdown>0, vt, velocity)


def prepare(plan, raw, ticks, apex, terminal, gate_tick, start, curve_state,
            standing, launch_state, touchdown, gravity, gyro, linear, wheel_speed):
    time = ticks*.0025
    enabled = apex & ~terminal & (ticks<2000)
    new = enabled & (gate_tick<0)
    gate_tick = torch.where(new, ticks, gate_tick)
    start = torch.where(new, time, start)
    parameters, gains = decode(raw)
    height, velocity = reference(parameters, plan, time, start, touchdown)
    landing = torch.stack((height-.18, velocity, torch.zeros_like(height), torch.ones_like(height)), 1)
    state = torch.where(apex[:, None], landing, launch_state)
    wheel = gains[:, 0]*gravity[:, 0]*10+gains[:, 1]*gyro[:, 1]*.2
    wheel += gains[:, 2]*linear[:, 0]*.5+gains[:, 3]*wheel_speed.mean(1)*.05
    pitch = gains[:, 4]*gravity[:, 0]*10+gains[:, 5]*gyro[:, 1]*.2
    roll = gains[:, 6]*gravity[:, 1]*10+gains[:, 7]*gyro[:, 0]*.2
    correction = torch.stack((pitch+roll, pitch-roll, pitch-roll, pitch+roll, wheel, wheel), 1)
    correction = torch.where(enabled[:, None], correction, torch.zeros_like(correction))
    return dict(curve_state=torch.where((time>=.6)[:, None], state, curve_state),
        actions=torch.where((time<.6)[:, None], standing[:, :6], correction),
        gate_tick=gate_tick, landing_start=start, control_gate=enabled,
        effective_action=torch.where(enabled[:, None], raw-initial_raw(raw.device), torch.zeros_like(raw)))

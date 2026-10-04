"""Causal 400 Hz landing reference with an independent compression distance."""
import torch

ACTION_DIM, OBS_DIM = 16, 17
NAMES = ('air_height_m', 'approach_s', 'stroke_m', 'absorb_s', 'recover_s',
         'kp_scale', 'kd_scale', 'weight_support')
LOW = (.18, .10, .03, .07, .35, .20, .15, .0)
HIGH = (.215, .26, .07, .16, .85, .85, .85, 1.2)
INITIAL = (.20, .16, .05, .12, .60, .45, .30, .50)
STATE_NAMES = ('gate_tick', 'air_start', 'touch_seen', 'touch_height', 'touch_velocity')


def encode(values, device='cpu'):
    p, lo, hi = [torch.as_tensor(x, device=device, dtype=torch.float32) for x in (values, LOW, HIGH)]
    return torch.logit(((p-lo)/(hi-lo)).clamp(1e-5, 1-1e-5))


def initial_raw(device='cpu'):
    return torch.cat((encode(INITIAL, device), torch.zeros(8, device=device)))


def decode(raw):
    lo, hi = raw.new_tensor(LOW), raw.new_tensor(HIGH)
    return lo+(hi-lo)*raw[:, :8].sigmoid(), 4*raw[:, 8:].tanh()


def smooth(u):
    return u*u*(3-2*u)


def hermite_down(start, bottom, initial_velocity, duration, age):
    """Monotone compression with a matched downward starting velocity."""
    distance = (start-bottom).clamp_min(0)
    v0 = torch.maximum(initial_velocity.clamp_max(0), -3*distance/duration)
    u = (age/duration).clamp(0, 1)
    h = start-distance*smooth(u)+v0*duration*u*(1-u).square()
    v = -distance*6*u*(1-u)/duration+v0*(1-4*u+3*u.square())
    return h, torch.where(age>=duration, torch.zeros_like(v), v)


def tick(raw, state, *, ticks, apex, terminal, launch_height, touchdown, leg_height,
         incident_vz, gravity, gyro, linear, wheel_speed, mass):
    """Pure function. All state is per-world and only advances after its apex."""
    p, gains = decode(raw)
    hair, approach, stroke, absorb, recover, kp, kd, support = p.unbind(-1)
    time = ticks*.0025
    enabled = apex & ~terminal & (ticks<2000)
    new = enabled & (state['gate_tick']<0)
    gate = torch.where(new, ticks, state['gate_tick'])
    air_start = torch.where(new, launch_height, state['air_start'])
    air_age = (time-gate*.0025).clamp_min(0)
    u = (air_age/approach).clamp(0, 1)
    ha = air_start+(hair-air_start)*smooth(u)
    va = (hair-air_start)*6*u*(1-u)/approach
    first = enabled & (touchdown>0) & ~state['touch_seen']
    h0 = torch.where(first, leg_height.mean(1).clamp(.115, .225), state['touch_height'])
    v0 = torch.where(first, incident_vz.clamp(-2., 0), state['touch_velocity'])
    seen = state['touch_seen'] | first
    bottom = (h0-stroke).clamp_min(.105)
    age = (time-touchdown).clamp_min(0)
    hc, vc = hermite_down(h0, bottom, v0, absorb, age)
    r = ((age-absorb)/recover).clamp(0, 1)
    hr = bottom+(.18-bottom)*smooth(r)
    vr = (.18-bottom)*6*r*(1-r)/recover
    landed_h = torch.where(age<absorb, hc, hr)
    landed_v = torch.where(age<absorb, vc, vr)
    height = torch.where(seen, landed_h, ha)
    velocity = torch.where(seen, landed_v, va)
    # Prepare compliance before contact; retain the real motor FIFO delay.
    soften = smooth((air_age/.04).clamp(0, 1))
    restore = torch.where(seen, smooth(r), torch.zeros_like(r))
    blend = soften*(1-restore)
    kp = 1+(kp-1)*blend
    kd = 1+(kd-1)*blend
    force = mass*9.81*support*seen*(1-restore)
    wheel = gains[:, 0]*gravity[:, 0]*10+gains[:, 1]*gyro[:, 1]*.2
    wheel += gains[:, 2]*linear[:, 0]*.5+gains[:, 3]*wheel_speed.mean(1)*.05
    pitch = gains[:, 4]*gravity[:, 0]*10+gains[:, 5]*gyro[:, 1]*.2
    roll = gains[:, 6]*gravity[:, 1]*10+gains[:, 7]*gyro[:, 0]*.2
    correction = torch.stack((pitch+roll, pitch-roll, pitch-roll, pitch+roll, wheel, wheel), 1)
    correction = torch.where(enabled[:, None], correction, torch.zeros_like(correction))
    next_state = dict(gate_tick=gate, air_start=air_start, touch_seen=seen,
                      touch_height=h0, touch_velocity=v0)
    return dict(state=next_state, enabled=enabled, height=height, velocity=velocity,
                kp=kp, kd=kd, force=force, correction=correction, bottom=bottom)

# Bounded slot allocator and immediate assistance withdrawal

At 52.5% assistance the frozen clearance125 controller requested 50 N per leg,
but its shared 0.06 rad motor-offset window admitted only about 15 N. Case 3
reached approximately 97.2 mm and self-contacted 65 ms after touchdown.
This revision changes the slot residual allocation window, not the motor model.

Four predeclared candidates use total motor-offset bounds 0.06, 0.075, 0.09,
and 0.12 rad. The original policy correction remains tanh bounded. The slot
increment uses a common scale per leg and preserves its Jacobian direction;
wheel commands remain unchanged. Corrections enter the existing 15-channel
FIFO, so they have the same real delay. There is no added external force,
contact masking, state projection, larger motor torque envelope, or new sensor.
The actual 24 V estimated speed/torque envelope, 17 Nm peak, sustained-limit
stop, rotor5ms and all collision/height/stroke/impact/recovery gates remain.

The fixed 180-world screen compares all four at 52.5%. The first smallest
window passing strict screening and reducing the worst slot error by at least
1 mm is eligible for independent validation. Screening is not qualification.
If none meets this criterion, preserve the results and stop for analysis.

The selected controller is freshly tested at 55%, then 52.5%, then 50%.
Each level requires native45 with independent physical, FIFO, assistance and
NumPy allocator audits plus exactly three512 deterministic replays. Complete
qualification advances immediately even at zero PPO updates. A failed actor
is never repeatedly tested until passing. Bounded learning entry permits
rebound at most0.10 m/s, but never self-contact. Final rebound remains1e-6.
When learning is admissible, use2 then8-update chunks, up to126 new updates;
two prior updates still count toward the shared128. Stop at50%, entry failure,
or budget exhaustion. Preserve old55% configurations and every failure.

A gain/controller change and subsequent PPO learning must be reported separately.
The source weights still contain the old profile; deployment requires the
new declared profile as well as the weights. No unassisted or hardware claim.

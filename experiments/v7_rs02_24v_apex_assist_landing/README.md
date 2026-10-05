# Assistance withdrawal only where the landing policy can act

The previous uniform 61.25% experiment stopped before PPO: one of512 worlds
crossed the unchanged rotor-proxy velocity residual limit during takeoff,
before the landing policy could affect commands. Existing native45 traces
show that this numerical equality constraint is close to its0.01rad/s limit;
they do not establish the full cause of between-replay variation. We retain
all guards and preserve the failed run. No timestep, integrator, solver,
rotor inertia, motor curve, collision shape or tolerance is changed here.

Keep the source slot selected112 and original62.5% assistance through the
observed COM apex. Only then lower assistance smoothly over100ms to60%; use
61.25% as a bounded fallback if60% fails. The original0.5–0.6s entry ramp is
unchanged. This is partial withdrawal in the airborne/landing/recovery phase,
not a uniformly lower assistance level and not zero-assistance qualification.
All linear virtual forces remain zero.

The frozen launch and the landing plan sampled at0.6s see original assistance.
The existing16D landing policy retains400Hz sensor feedback; PPO may only
adapt its landing parameters and gains. Keep the original13-term reward,
height gates and all-case landing admission. Independent audits reconstruct
the causal apex event and every400Hz wrench, and check the original physics,
FIFO, slot feedback and controller schedule. Add per-case residual maxima and
terminal diagnostics even when a rollout is not recorded as a full trace.

Require native45 admission followed by smoke and three consecutive512-world
checks before formal PPO. Any failure stops this attempt; never retry until a
random replay happens to pass. The first training rollout is checked again.
512 worlds repeat45 initial-condition/delay cases, not512 distinct scenarios.
Code goes through GitHub; generated data remains in ignored runs directories.

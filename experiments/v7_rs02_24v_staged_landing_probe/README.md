# Staged impedance diagnostic

Fixed source: compliant v3 train_02 update 56 plus the independently evaluated
air-target -5 mm offset. No PPO updates and no automatic checkpoint replacement.
24 V estimated envelope, 62.5% attitude assistance, launch, collisions, solver,
400 Hz physics, reference trajectory, and all 15 delayed motor payload channels
retain the frozen v3 implementation. Frozen sources are not edited.

Eleven profiles (495 worlds) screen lower pre-contact position stiffness with
unchanged air Kd, so its velocity-tracking term is preserved. After measured
touchdown, stiffness, damping and weight-support scale can rise smoothly with
actual COM compression. An ideal vertical stopping-force estimate based on
measured downward velocity and remaining nominal stroke accelerates the gain
recovery when it rises from 250 to 300 N. This is an urgency signal, NOT a clamp
on contact force or an external vertical force. Stage progress cannot decrease.
Original gains return during the original recovery trajectory. Lower stiffness
also affects air position tracking; unchanged velocity gain does not prove
unchanged actual contact velocity, which the native traces separately measure.

Baseline has scheduling disabled and is bitwise unchanged at the payload level.
All profiles are gated after each world's apex and share the original motor
FIFO. Profiles stay fixed within an episode. Only Kp/Kd/support scheduling is
new; no reference height, stroke, velocity, motor limits, or assistance changes.
The controller gain lower bound is extended to 5% of original standing gains
for this experiment, with gains never above the original standing gains.

Pipeline: native baseline45, native soft60 smoke45, fast search495, selected
native45, independent selected repeat45, selected native +10mm landing plane45.
Real exits are recorded by the PowerShell wrapper; failed artifacts are kept.
All stages have a 1800-second bound, shared exclusive-physics lock and GPU guards.

Candidate admission is predeclared: all45 pass, actual mean COM first-stop stroke
>=50mm, mean peak at least3% below baseline, worst peak no worse, wheel and COM
height at least99% of baseline, stable confirmation no more than100ms slower.
The qualified candidate with lowest mean peak proceeds. If none qualifies,
later stages are explicitly skipped. Report mean<=300N, mean<=250N and ALL
peaks<=250N separately. Native admission and the independent repeat decide
whether the improvement survives screening; screening alone is not promotion.
All later force peaks and first20ms peaks are reported to detect peak shifting.

The +10mm fixture raises the plane only while all worlds are airborne at1.26s,
updates Warp's static geometry cache, and checks actual final elevated support.
It does not test a stair edge or a single-wheel collision. Surface height is
not an input to the policy or new controller schedule.

CPU contracts cover causal gating, unchanged air velocity gain/reference,
compression/speed response, monotone braking, recovery and exact audit import.
Native audits retain all physical limits, FIFO equality and COM stroke checks,
and add correspondence between staged requests and delayed motor payloads.
Source hashes are captured and checked again at completion. Runs/checkpoints/
traces/reports stay local to the remote machine and are not publicly uploaded.

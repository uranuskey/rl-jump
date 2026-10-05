# Lower takeoff assistance after rotor constraint correction

This independent experiment continues the user's authorized assistance withdrawal
at61.25% before the observed COM apex and60% after the100ms apex ramp. It uses
source80 weights and the explicitly audited rotor5ms numerical coupling revision.
The old current625_600 result remains unqualified: one of its1536 batch samples
had0.075854331m/s rebound. There is no claim that the parent successfully withdrew
assistance, and that failure is not overwritten or retried until passing.

The corrected reference625 passed native45 and512. The new, lower target must
independently pass learning entry in native45,512, two-update45-world smoke and
a fresh512 replay immediately before formal PPO. Learning permits at most0.10m/s
rebound as previously authorized; every other old qualification bound remains,
and takeoff residual must stay within0.008rad/s. Both original-physics and
corrected-physics reference limits apply. The final rebound limit remains1e-6.

Formal PPO uses512 environments,128 new updates, source80 weights and fresh Adam.
Launch network weights, motor limits, controller, original13-term landing reward,
physics gates and the5ms coupling are frozen. The landing actor remains gated
until the actual apex: this adapts landing to a lower takeoff-assistance setting;
it is not takeoff-policy learning. Assistance torque is independently reconstructed
from400Hz traces. No linear assistance or yaw torque is added.

Selection takes the latest strictly qualified trained checkpoint at this lower
setting, without requiring or ranking impact improvement. Deterministic checks
run at update2 and every8 updates. Initial/unqualified policies cannot be selected.
Native45 and3x512 final replays of selected/latest must all pass, along with their
training deterministic qualification. Reference and seed use one512 comparison
each.512 worlds repeat45 conditions; they are not512 unique test conditions.

Preflight, smoke, training, independent evaluation and audit have separate real
process exits. The audit reconstructs all128 update admissions and accepted actor
steps, selected history, raw evaluation metrics and physical/assistance traces.
Old source and failures remain immutable. This pipeline does not automatically
reduce assistance again.24V is still an estimated motor curve; the result remains
assisted simulation and never establishes hardware or zero-assistance qualification.

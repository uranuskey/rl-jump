# Contact-force spreading and 10 mm landing-height probe

Source: compliant v3 train_02 selected update 56. This is a bounded deterministic
parameter search, with zero PPO updates. Existing frozen experiments are read-only.
24 V estimated motor envelope, 62.5% attitude assistance, fixed launch, collisions,
400 Hz integration, all 15 FIFO channels and original physical guards are retained.

Nine profiles test air retraction target -5 mm, duration -8/-16 ms and damping
0.8/1.2 times the selected values. No other controller outputs are changed.
Search uses 405 worlds: nine full 45-condition groups. Native independent checks
use 45 worlds. A candidate needs all 45 passes, mean actual COM first-stop stroke
at least 50 mm, mean peak force at least 3% lower, no worse worst peak, height
retention of at least 99%, and no more than 100 ms slower standing confirmation.
Selection failure is retained and reported; no automatic retraining follows.

The +10 mm case changes only the landing plane while ALL worlds are in free
flight at 1.26 s, with at least 30 mm gap above the new plane. q/v are unchanged;
there must be no immediate contact force. The policy receives no terrain-height
input. This isolates earlier touchdown while preserving launch: it is NOT a
fixed stair edge or a single-wheel obstacle test. Static stair placement under
the existing near-in-place jump was found unable to both avoid pre-apex wheel
geometry and fully cover all landing patches; the saved CPU analysis records this.

Jump height keeps its original takeoff-ground datum. A separate sensor records
actual clearance over the elevated plane, and final elevated support is checked.
Native traces retain all original force/actuator/contact/assistance audits.
All artifacts stay under ignored runs/, never in the public source repository.

Probe_01 stopped before simulation due to an audit-module name collision.
Probe_02 completed flat and screening, then stopped before landing because the
requested ground position did not propagate into Warp's static world-geometry
cache. Neither failed raised run is evidence about obstacle buffering. The
fixture now updates both authored placement and the static geom_xpos cache,
records measured plane height at every sample, and audits its final support.

run_height_followup.ps1 reuses the verified flat baseline and retains all failed
artifacts. It tests native raised selected56, then native flat/raised air_minus5mm.
That profile passed 45/45 in screening but narrowly missed the declared 3%
improvement threshold. This follow-up does not relabel that search as admitted;
native comparison and any failure remain explicit. No PPO or auto-promotion.

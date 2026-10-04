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

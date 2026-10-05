# Clearance repair and automatic withdrawal

The previous immutable course retained55% assistance and stopped at52.5% when
three512-world replays had1/5/3 illegal contacts. This study diagnoses those
contacts and changes only the existing encoder-based slot tracking gains and
bounded motor-offset force cap. Robot geometry, collision detection, contact
solver, rotor5ms coupling,24V estimated curve, fixed launch network, landing
trajectory, FIFO and the13 reward terms are unchanged.

The diagnostic replay records exact GPU geometry pairs, signed penetration and
contact normal force on first termination, plus q/v. CPU reconstruction confirms
base_link versus left_lower/right_lower at97mm leg height; measured fore-aft
position is about40.5mm versus the53mm CAD path. At the same height, a static
1mm correction removes those overlaps; that is not a dynamic qualification.

A fixed six-profile270-world screen includes the original controller. The first
smallest-gain candidate meeting the original strict bounds and improving maximum
slot tracking error is taken forward. A passing screen does not qualify anything.
Each actual level requires independent native45 with complete physical/controller/
slot/assistance audits and three512 deterministic replays, plus audit reconstruction.

The changed controller is requalified first at55%, then52.5%, then50%. Advance
immediately when tests pass, including with zero PPO. If bounded learning entry
passes, use2 updates then8-update chunks with qualification after each chunk.
The remaining update budget is126: the earlier course already used2 of the
shared128. Fresh Adam per level; exact optimizer/RNG resume within a level.
The same failed actor cannot be repeatedly evaluated until it passes.

Any collision or other failure outside learning entry stops this bounded course.
No collision/clearance gate is relaxed; the old55% controller/model remains
available independently. A controller-only gain improvement is reported separately
from PPO learning. All failures and exact process exits are preserved.

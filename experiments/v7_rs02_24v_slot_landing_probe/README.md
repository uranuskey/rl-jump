# Slot-aligned compliant landing

Independent successor to the precontact and geometry diagnostics. Keep selected56,
the fixed launch, the 24 V estimated torque-speed curve, 62.5% attitude assistance,
all physical guards, geometry, and the original 15-channel delayed motor payload.

The original CAD safe crouch is a path, not a height-only workspace. Use measured
hip/knee angles and velocities to calculate each wheel's fore/aft position relative
to the hip. A Cartesian spring/damper follows the CAD path at the measured height.
Map its force through the motor-coordinate fore/aft Jacobian. Encode this torque
increment as an additional motor position offset inside the existing +/-0.06 rad
correction range; allocate one common scale per leg to preserve force direction.
This is a motor controller, never an external assistance force. Actual total motor
torques still use the original limiter, speed envelope, and saturation-stop guards.
The fore/aft feedback uses existing encoders; precontact proximity remains ideal
simulated sensing, with hardware realization unqualified.

Nine fixed profiles / 405 worlds screen baseline, slot-only feedback, and three
precontact settings with low/medium/high fore/aft gains. Native45 baseline and
slot-only checks precede screening. Check two fully passing screening profiles
independently; select the best stable seed and repeat it on native45.

PPO seed qualification: every case passes, average actual first-stop COM travel
at least 50 mm, wheel/COM heights at least 99% of matched baseline, recovery at
most 100 ms slower, mean and worst impact at most 2% above baseline. This tolerates
small replay variation for a training start; it does not claim an improvement.
Policy promotion retains the stricter prior gate: at least 3% lower mean peak,
worst peak no higher, and the same landing/height/stroke/recovery requirements.

The air-minus-5 mm transform is embedded inside the actor, so PPO's old log
probability, new log probability, and KL calculation use the identical mean.
Controller and source hashes are recorded. All results are ignored under runs/.
Preserve failed runs, stop on process/audit failures, and never weaken collision
checks or consume shorter jumps as apparent impact improvement.

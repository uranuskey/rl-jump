# Compliant landing controller, revision 3

Revision 2 retained passing landings and about 5.2 cm of real deceleration
travel, but its best passing candidate still averaged about 422 N. Some
early-retraction profiles caused illegal contact. This third isolated revision
moves the air retraction later: with s=clamp(age/duration), reference height is
start+(target-start)*smoothstep(s*s), with its exact analytic velocity. This
preserves more leg extension early in descent and concentrates retraction
velocity closer to contact. It does not predict future contact or change the
frozen launch. The actuator FIFO still delays every command.

Screen only nine focused profiles: target 150/165/180 mm by duration
180/200/220 ms, with all post-contact parameters fixed to revision 2's
50 mm compression, 120 ms absorption, kp scale .45, kd scale .40, and
weight-support .70. Preserve both failed earlier revisions. The same 10%
force reduction and all height/stability/stroke admission gates still apply.

## Earlier diagnostic evidence

The first isolated revision opened about 6.1 cm of actual COM deceleration
travel, but all 27 profiles failed the force-reduction admission. Independent
native traces of its best passing profile measured a 483 N mean peak on the
first contact sample, with only 0.44 Nm maximum motor torque on that sample.
Pre-contact leg retraction was 0.182 m/s versus the parent's 0.361 m/s; its
200 mm air target had slowed the useful retraction. More post-contact travel
alone therefore did not remove the initial contact spike.

The second revision retained the first revision unchanged. It expanded only the
air-reference bounds to 145..215 mm and approach time to 100..320 ms, then
screens 27 combinations of air target (155/170/185 mm), approach time
(160/220/280 ms), and damping scale (0.20/0.40/0.65). Compression remains an
independent 50 mm in this screening. Physics, motor limits, launch, reward,
and the original admission threshold are unchanged. Gain trace metadata now
reports the actual legacy gain before the delayed compliance gate arrives.

The prior controller generated touchdown references at 50 Hz, then delayed them
through a 0..20 ms FIFO. In the audited parent traces, the impact peaked before
the first touchdown-generated command arrived in all 45 conditions. Guided PPO
also reduced the nominal air-to-bottom reference distance to about 3.9 mm while
motor PD remained fixed at 60/2.

This isolated experiment preserves the fixed launch, full collision/inertial
model, original 400 Hz physics and guards, RS02 24 V estimated envelope, and
62.5% attitude assistance. No contact material, timestep, motor limit or external
force is changed. The old stopped experiment and all its artifacts are retained.

The new 16D episode policy selects eight bounded controller parameters and eight
attitude/wheel feedback gains. The first eight are air height, approach time,
independent 3..7 cm leg-reference compression distance, compression time,
recovery time, motor stiffness scale, motor damping scale, and weight-support
feedforward. Reference compression is not claimed to equal actual COM travel.

After each world's measured COM apex, stiffness/damping ramp to the selected
levels over 40 ms, before contact. A causal 400 Hz controller updates height,
velocity, gains and feedback every physical step. All 15 command channels share
the original physical FIFO. Before apex, the exact legacy servo is used.

At first contact, compression starts from measured leg height, using the recorded
pre-contact downward COM velocity as the initial reference velocity. A monotone
Hermite segment stops at the independent compression distance without overshoot.
Recovery smoothly returns to 180 mm and the original PD. Weight support is motor
torque feedforward, allocated inside the unchanged RS02 envelope; it is never an
external upward force. Actual COM travel and all force peaks are measured by the
unchanged guided landing reward, including the early 20 ms window.

The 13D old action meanings are not loaded into the 16D new output. Initialization
copies only the compatible 17D-observation hidden layers and eight attitude
feedback output rows from guided update 80. The eight new controller outputs,
critic and optimizer are deliberately new.

## Admission and execution

1. CPU behavioral checks, then an independent 45-condition old-controller baseline.
2. Screen nine physical profiles in one 405-world batch, nine full 45-condition
   groups. Prefix proof compares counterfactual parameters at the same
   state and verifies the actual delayed native command readback before apex.
3. Require all 45 conditions to pass, retain mean wheel height >=9.065 cm and COM
   rise >=11.74 cm, obtain >=3.5 cm actual first-deceleration COM travel, and reduce
   mean peak force by at least 10%. Independently validate the chosen profile with
   the original native physics-loop guard path and audit its 400 Hz traces.
4. Two smoke PPO updates at 45 worlds must retain admission. Recheck admission
   on 512 worlds, then run 128 new on-policy updates. Physical search samples
   never enter PPO. Evaluate deterministically every eight updates.
5. Independently evaluate old baseline, admitted seed, selected and latest models;
   audit physical limits, actual exits, update counts, real COM travel and all
   15 FIFO channels. Report controller improvements separately from PPO gains.

Failed gates stop the pipeline. These are assisted simulation conditions, not
unassisted or hardware qualification. Source is synchronized through GitHub;
all generated models, traces and logs remain off the public repository.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File experiments\v7_rs02_24v_compliant_landing_v3\run_pipeline.ps1 -PythonPath D:\RL_JUMP\.venv\python.exe -RunId train_01 -Checkpoint D:\RL_JUMP\repo\experiments\v7_rs02_24v_guided_landing\runs\train_01\model_0080.pt
```

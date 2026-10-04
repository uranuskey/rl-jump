# Compliant landing controller

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
2. Screen 27 physical profiles in three 405-world batches, nine full 45-condition
   groups per batch. Prefix proof compares counterfactual parameters at the same
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
powershell -NoProfile -ExecutionPolicy Bypass -File experiments\v7_rs02_24v_compliant_landing\run_pipeline.ps1 -PythonPath D:\RL_JUMP\.venv\python.exe -RunId train_01 -Checkpoint D:\RL_JUMP\repo\experiments\v7_rs02_24v_guided_landing\runs\train_01\model_0080.pt
```

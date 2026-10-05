# Rotor constraint diagnosis and continued attitude-assistance withdrawal

The user authorized fixing the takeoff constraint residual and continuing
withdrawal. The previous original-physics experiment remains immutable and
unqualified at61.25% takeoff/60% after-apex assistance: world204/case24 stopped
at1.08s with velocity residual0.010017395rad/s against the0.01 limit.

Four ideal rotor proxies use q_rotor =7.75 q_joint, with solimp0.9999 and the
default solref20ms/critical damping. This independent physics revision compares:

| Variant | Newton iterations/tolerance | Line-search iterations/tolerance | Rotor solref |
| --- | --- | --- | --- |
| original | 100 /1e-8 | 50 /.01 | .020,1 |
| solve_strict | 200 /1e-10 | 100 /.0001 | .020,1 |
| rotor10ms | 100 /1e-8 | 50 /.01 | .010,1 |
| rotor5ms, conditional fallback | 100 /1e-8 | 50 /.01 | .005,1 |

The coupling change is explicit modeling of a closer approximation to ideal
rigid gearing, not a measured mechanical compliance or a mere relabeling of
the old physics. Original XML, geometry, inertias, motor capability, contact
parameters, timestep2.5ms, refsafe, policy weights and all physical gates remain
intact. There is no q/v projection, masking, velocity clipping or threshold
relaxation. Only the listed runtime model parameters may change.

MuJoCo's [solver-parameter documentation](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters)
describes positive solref as time constant and damping ratio. Euler's refsafe
lower bound is twice the timestep; even the5ms candidate respects it. Installed
mujoco-warp3.14.0 constraint.py independently confirms that lower bound.

Diagnosis uses three512-world1.20s prefixes per variant at61.25/60. It saves
the0.90–1.20s joint/rotor q,v,acceleration, coupling force, integration-solve and
post-forward iteration counts. Offline float64 reconstruction separates the
reported float32 residual from final measurement rounding. These short trials
are never landing qualifications. A correction needs no physical/task failures,
velocity residual at most0.008rad/s and the unchanged position limit in all
three prefixes. Stricter solve is preferred if it has that margin. The10ms
comparison always runs;5ms is tested only when both other candidates lack margin.

Then independent full native45 and512-world evaluations qualify the chosen
revision against the saved original-physics reference as well as its own
reference. Full native traces verify actual GPU coupling time constants,
assistance, motor/physics, controller, slot path and landing. Native45 plus
three512 repeats must all pass at each candidate level:
62.5/60 ->61.25/60 ->60/60 ->57.5/57.5 ->55/55 ->52.5/52.5 ->50/50.
The original slot11262.5% reference uses one512 replay. World batches repeat the
same45 initial/delay conditions. First failed qualification stops withdrawal;
failed trials are retained and never retried until passing. Force is an upper
constraint, not an improvement objective. No PPO updates occur in this pipeline.

The supervisor runs through WMI, captures real stage/wrapper exits and survives
SSH closure. Run directory names are unique. Audit independently reconstructs
prefix residuals, correction choice and per-level qualification from saved data.
AUDITED means evidence checked, not necessarily a successful fix or withdrawal.
Results retain the estimated24V motor and nonzero assistance boundaries; there
is no unassisted or hardware claim and no automated continuation below50%.

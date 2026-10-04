# Precontact landing diagnostic

Independent successor to the staged-impedance diagnostic. Preserve the frozen
v3 controller, selected56 checkpoint, fixed takeoff, estimated 24 V motor curve,
62.5% attitude assistance, physical limits and 15-channel motor FIFO.

First reconstruct wheel-to-ground closing speed and remaining leg travel from
saved 45-case native traces. This CPU analysis does not run or train a policy.
Subsequent candidates will change precontact preparation, with native landing
acceptance and actual COM stroke checked independently of commanded leg travel.

Reference: Sato et al., IEEE Access 2022, DOI 10.1109/ACCESS.2022.3153127:
precontact foot velocity and postcontact support are separate control problems.
This diagnostic does not claim that their single-leg hardware reduction carries
over to this wheeled biped. All outputs under `runs/` remain ignored and private.

## Frozen trial specification

- Baseline: selected56 plus the previously checked air-height minus 5 mm setting.
- 11 profiles (495 worlds in the fast screening backend); 45 worlds per independent native check.
- Closing-speed feedback targets 0.25/0.45/0.65 m/s, approaching over 30/45/60 ms.
- Extra retraction limited to 3/6/9 mm with a nominal 160 mm reference floor.
- No changes to stiffness, damping, support feedforward, landing recovery, takeoff,
  contacts, solver settings, actuation envelope, delays, or attitude assistance.
- Ideal proximity/causal mesh-bottom velocity is new controller information. Its
  sensor noise, latency, and hardware realizability are not established here.
- Screening is only directional evidence. Independently check the best two
  fully passing profiles even if they miss the screening improvement threshold.
- Native admission: 45/45 pass, mean peak at least 3% below fresh baseline, worst
  peak no higher, mean actual first-stop COM stroke at least 50 mm, wheel/COM
  heights at least 99% of baseline, recovery at most 100 ms slower.
- Only an admitted native result gets a repeat and a +10 mm uniform-plane
  perturbation. This is not a static step-edge or single-wheel obstacle test.
- No PPO updates, model promotion, or relaxation of gates in this diagnostic.

Native audits reconstruct motor FIFO, reference velocities, causal measurement
history, precontact-only activation, unchanged impedance, and actual COM stroke.
Process exit receipts are independent of progress/result status. Existing runs
must never be overwritten; any error retains its artifacts and stops the pipeline.

## Baseline failure and isolated protection revision

`probe_01` stopped before testing the new precontact controller: the unchanged
baseline passed 44/45 and world 0 triggered self-contact at 1.4475 s. GPU flags
and CPU reconstruction both locate lower-link/body contact near 101.3 mm actual
leg height (about 0.108 mm penetration on the left). Previous passing native
replays and this failure demonstrate inadequate margin; their numerical
cross-run discrepancy is not explained or declared fixed.

Optional `--protect-landing` adds postcontact gain scheduling based on measured
leg height: smooth onset at 140 mm, full strength at 115 mm, up to 1.20 times
the inherited Kp and 1.10 times Kd, capped at original standing gains. References,
support feedforward, collision criteria, physics and actuator limits stay fixed.
This is a new motor control change, not hardware clearance or an external force.
Its baseline must pass two independent native45 runs before the precontact
screening proceeds. All screened profiles share this same protection so that
the effect of precontact feedback can be assessed separately. No candidate gate
is relaxed. Preserve `probe_01` and use a new run id for the protected experiment.

## Bounded coordinated braking follow-up

`probe_02` completed without an eligible candidate. Both protected native
baselines passed 45/45 (331.74 / 331.67 N average peak), but the initial
precontact profile passed only 38/45 at 292.22 N. All ten new screening profiles
had illegal contacts. CPU reconstruction of its world 0 confirms lower-link/
body contact at 98.8 mm leg height while COM velocity was still -0.387 m/s.
Do not report the failed groups' recovery averages as an improvement: terminated
worlds have success_s=0 in the inherited metric. Their force means also include
truncated failures and are not qualified full-land-and-recover performance.

`--coupled --protect-landing` selects a separate fixed nine-profile / 405-world
follow-up, retaining the current protected baseline and two postcontact-only
controls. Earlier braking starts at 160 mm and reaches full strength at 125 mm:
Kp/Kd multipliers are either 1.35/1.15 or 1.60/1.30, still capped at standing
gains and sent through the original delayed motor FIFO. Three precontact
settings (late/3 mm, mid/6 mm, late/6 mm) pair with both braking schedules.
This tests coordinating wheel preparation with earlier motor braking; it is not
a force-tracking controller or a claim of 250 N performance. The same admission
thresholds, native baselines, independent repeats, and no-PPO boundary apply.

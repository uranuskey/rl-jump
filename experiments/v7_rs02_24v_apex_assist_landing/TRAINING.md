# Landing adaptation after the observed apex

Keep takeoff assistance at 62.5%. After the physical COM apex, smoothly lower
it over 100 ms to the target admitted by the native 45-case probe. This run
uses a single target: it never advances to another assistance level or
restarts itself. It does not qualify uniform 60% or zero-assistance jumping.
The 24 V estimated motor curve, physics, collisions, equality constraints,
launch, slot controller, height gates and 13-term landing reward are frozen.

The pipeline requires a completed, audited probe with a real zero exit code,
then a 45-world two-update smoke and three consecutive 512-world deterministic
seed replays. All three must pass; any failure stops the attempt. The 512
worlds repeat the original 45 initial-condition/delay cases, rather than
introducing 512 distinct scenarios. Three passes are an admission check,
not proof that a rare numerical failure cannot recur. A same-size 62.5%
reference controls for the unequal weighting of those cases. The formal
training process checks its initial policy once again before PPO.

Initialize from slot selected112 with fresh Adam optimizers. The smoke model
is not used to initialize formal training. Run 512 environments and 128 new
PPO updates; keep the 16 landing parameters, 400 Hz feedback, full actor and
Adam rollback, halved-step retries and 0.03 KL limit. Exploration standard
deviations decrease from .035 to .025 for the first eight action components
and from .012 to .008 for the eight feedback gains. Save actual accepted
actor steps, stochastic results and deterministic evaluations separately.

Evaluate every eight updates. Preserve the initial checkpoint when no
qualified improvement is found. Independent native 45-case evaluation then
compares reference625, the reduced-assistance seed, selected and latest.
Reconstruct each causal apex and every recorded 400 Hz assistance wrench;
repeat the existing physical, FIFO, schedule, slot and controller audits.
Audit real stage exits, source and checkpoint hashes, all three admission
replays and accepted actor steps. Report selected and latest independently,
and distinguish phase-schedule effects from PPO improvement. Baseline failure
retains every case, residual diagnostics and explicit zero update counters.

Use a WMI-created hidden supervisor for persistent Windows OpenSSH launch.
The unchanged supervisor holds the child process handle and records its
actual exit code separately from the Python stage receipts. The shared
physics lock prevents simultaneous simulator jobs. Preserve all failed runs.
Only source is synchronized through GitHub; checkpoints, traces and reports
remain in ignored run directories. No further withdrawal follows this run.

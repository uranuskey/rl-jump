# PPO continuation after native slot-path admission

The user authorized automatic review and starting PPO after the bounded diagnostic.
The probe's two native candidate checks and repeat must have real exit 0, all four
audits PASS, unchanged source hashes, and the documented physical seed gates.

`run_training.ps1` performs these stages in order:

1. Fresh source selected56 weights, approved fixed slot/precontact profile, and
   two new PPO updates on 45 cases. The final deterministic replay must retain
   seed admission. This smoke does not supply weights to formal training.
2. Independent 512-world deterministic seed admission with fresh source weights.
3. Fresh source weights and Adam again, 512 worlds, exactly 128 new updates.
4. Independent native 45-case baseline/seed/selected/latest evaluation without
   exploration. The baseline is the protected air-minus-5 mm controller, with
   neither slot feedback nor extra precontact retraction. It uses the same jump.
5. An independent audit verifies true process receipts, all 128 updates, executed
   actor steps, source hashes, fixed launch, physical telemetry, COM stroke,
   all 15 delayed payload channels, causal slot feedback and precontact schedule.

One 16-dimensional landing-plan action is sampled at 0.6 s per episode. Effects
remain gated until COM apex. Gaussian standard deviations decrease linearly from
0.060 to 0.035 for eight parameter logits and 0.020 to 0.012 for eight feedback
gain logits over 128 updates. This is plan exploration, not sensor noise or
independent noise every servo tick. PPO actor LR is 5e-5, critic LR 3e-4, clipping
0.2, and attempted actor updates with mean KL above 0.03 are rolled back along
with their optimizer state. The original guided 13-term full-landing reward is
unchanged (including 300 N force target and actual first-stop COM stroke).

The air-height minus 5 mm change remains inside the actor mean and therefore
inside both PPO log-probability and KL calculations. All checkpoints explicitly
record this contract; loading them with an unmodified parent Policy is invalid.

Deterministic evaluation occurs every eight updates. `selected` starts at the
qualified seed and changes only if all worlds pass, height/stroke/recovery gates
hold, mean impact improves by at least 3% against the native control and at least
0.25 N against the incumbent, and worst impact does not exceed the native control.
`latest` always records the last update regardless of qualification. Independent
evaluation reports each separately. Training completion is not policy acceptance.

The wrapper writes actual exit receipts for each process, including a distinct
audit receipt, and native failure evidence when nonzero. Read receipts/processes
before stale RUNNING markers. Stop on errors; preserve all failed products. No
automatic restart or further 128-update budget is authorized by this launcher.
Runtime source is frozen in TRAINING_FROZEN.json. Run artifacts are never uploaded.
24 V is an estimated voltage curve; 62.5% attitude assistance remains active.

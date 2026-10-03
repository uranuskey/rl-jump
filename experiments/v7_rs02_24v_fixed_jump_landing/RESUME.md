# 512-world continuation and synchronization reduction

The original `FROZEN.json`, source, checkpoints and runs remain unchanged.
`RESUME_FROZEN.json` separately freezes the additive implementation. Both manifests
are verified before continuation and after each update.

`FastLandingEnv` uses the same 400 Hz physics, limits, force-path guards and sensor
formulas. External-force predicates share one host read; numerical and actuator-path
predicates share another. Every guard still executes at every physics tick. Overflow
stays on the device until the combined guard is read. Guard failure precedence is
unchanged. Torch and Warp use the same existing CUDA stream. Launch-plan and apex
checks avoid dynamically sized boolean indexing. Reporting copies per-world metrics
and summary scalars in two batches; collection counts transitions on the device.
PPO, exploration, rewards and selection gates are unchanged.

The stopped 2048-world run supplies the model, both Adam states and CPU/CUDA RNG.
Episodes reset after changing batch size; this is not bitwise trajectory continuation.
Completed update JSON files are copied with unchanged hashes, and the mixed environment
schedule is recorded. The budget ends at update 128, rather than adding 128 updates.
The old zero-action model is retained, and the historical best policy is evaluated
again at 512 worlds before selection. Latest checkpoints are saved atomically before
periodic evaluation, so a STOP during evaluation preserves the finished update.

Before any resumed PPO update, CPU regressions compare guard/report behavior against
the frozen implementation and verify identical next PPO updates after a complete
state restore. Native checks compare sensors at identical state, verify force readback
and selective reset, then repeat the original 45-world apex/FIFO isolation proof.
The 512-world zero-action baseline must pass the unchanged height/landing gates.

Use `run_resume.ps1 -PythonPath ... -SourceRun ... -Checkpoint ... -ExpectedUpdate 38
-RunId train_512_01`. The pipeline records actual process exit, runs the original
45-condition independent evaluation and 400 Hz trace audit, then audits continuation
provenance and the mixed batch schedule. Generated runs, logs and resumed checkpoints
are local only and must not be committed. `rollout_s` and `ppo_s` measure separate
parts of each resumed update; compare total time with awareness that both batch size
and host synchronization changed.

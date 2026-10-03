# Fixed jump, continuous landing PPO

This independent task keeps the existing 24 V launch model and standing prefix frozen. A separate actor makes a new landing decision every 20 ms, beginning only after each world's confirmed airborne COM apex. A 400 Hz paired native check compares zero versus distinct future landing commands through each world's apex. The earlier flat-landing experiment is unchanged.

The seven Gaussian actions are symmetric reference-height velocity (1 m/s times tanh, integrated at 50 Hz), four motor-position corrections (the existing 0.06 rad limits), and two wheel-velocity corrections (20 rad/s limits). The validated landing curve is the zero-action reference; the height offset can evolve throughout landing, bounded by requested height 0.095 to 0.225 m. Existing mechanical workspace, contact, torque/speed, FIFO, and exposure checks remain enabled. This is continuous residual reference control, not direct learned torque control. The 17 Nm estimated RS02 24 V envelope and 62.5% attitude assistance are unchanged; no upward virtual force is added.

Actor observations are the existing 145 causal history/reference features plus time since apex/touchdown, touchdown state, learned height offset, episode time, and seven previous actions (157 total). The critic additionally retains 32 privileged simulator features (189 total). Task phase/contact detection is simulated; this is not a hardware-qualified estimator.

PPO uses post-apex transitions only. The ten existing landing reward terms become differences of the measured episode score at successive control steps. Their undiscounted sum is checked against the final score; small action change (0.02) and magnitude (0.001) costs are added. Reward scale is 0.01, gamma 0.995, GAE lambda 0.95, clip 0.2, two epochs and sixteen minibatches, actor LR 3e-5 and critic LR 3e-4. Fixed independent per-step Gaussian standard deviations are [0.10, 0.12, 0.12, 0.12, 0.12, 0.10, 0.10]. The launch has no trainable parameters in either optimizer. Zero-action initial behavior preserves the previous stable reference.

Selection still requires all worlds to pass the 5 s landing task, the first 45 cases to pass, mean wheel clearance at least 9.065 cm and mean COM flight rise at least 11.74 cm. Only a higher measured landing return can replace a qualifying baseline. Stochastic exploration failure is reported separately from deterministic evaluation.

From the repository root on the configured remote PC:

```powershell
$py='D:\RL_JUMP\.venv\python.exe'
$task='experiments\v7_rs02_24v_fixed_jump_landing'
& $py "$task\verify_cpu.py"
powershell -NoProfile -ExecutionPolicy Bypass -File "$task\run_checked.ps1" -Stage smoke -RunId smoke_01 -PythonPath $py -NumEnvs 45
powershell -NoProfile -ExecutionPolicy Bypass -File "$task\run_pipeline.ps1" -RunId train_01 -PythonPath $py -NumEnvs 2048 -SmokeResult "$task\runs\smoke_01\result.json"
```

Use fresh run IDs. Smoke performs the 45-world paired prefix proof and two PPO updates; the formal run performs 128 updates, independent 45-condition baseline/selected evaluation with 400 Hz traces, then a CPU audit. GPU stages are serial and each real exit code is recorded. Training is bounded to 12 hours, smoke/evaluation to 30 minutes. STOP files and GPU reserve checks remain active. No automatic retry, assistance withdrawal, driver changes, or training-output upload occurs. The original 45 conditions are known cases, not held-out perturbation qualification.

Source files in this new directory are explicitly force-added during publication because the repository's original exact allowlist is frozen. Runs, logs, models produced by training, and caches stay ignored. Do not force-add a whole working directory.

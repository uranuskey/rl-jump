# Method 1: one landing-parameter action per jump

The 24 V launch and standing prefix remain frozen. At 0.6 s a new actor selects
one 13D action: five landing-curve parameters plus eight causal feedback gains.
Both the curve and all feedback are gated separately by each world's confirmed
airborne COM apex. The parameters remain locked for the complete episode. Feedback
uses current IMU, encoder and body-velocity estimates at 50 Hz; no further policy
sample occurs during landing. All 400 Hz physics guards and the ten original
flat-ground landing reward terms remain active. Assistance stays at 62.5 percent
attitude torque, with no added vertical or horizontal force.

| Parameter | Range | Initial value |
|---|---|---|
| Descent leg-height target | 0.165–0.205 m | 0.180 m |
| Time to descent target | 0.12–0.35 s | 0.22 s |
| Touchdown compression target | 0.135–0.180 m | 0.160 m |
| Compression duration | 0.08–0.25 s | 0.14 s |
| Recovery duration | 0.25–0.80 s | 0.45 s |
| Eight signed feedback gains | `4*tanh(raw)` | 0 |

The recovery reference always ends at 0.18 m. The eight gains control wheel
feedback from pitch gravity, pitch angular velocity, forward body velocity and
wheel velocity, plus leg pitch/roll feedback from gravity and angular velocity.
They retain the existing tanh motor/wheel limits. A settled reference alone is
not proof that the physical robot settled; the original measured success gate
must still pass.

This actor is new: 17 causal inputs, two 64-unit tanh layers and 13 outputs.
It cannot load the previous 7D continuous controller's weights. Its initial output
reproduces the validated zero-feedback landing curve. The old continuous best and
failed latest checkpoints are preserved in their original experiment directories.
The frozen launch actor is separate from both optimizers.

The Gaussian standard deviations are 0.05 for the five raw curve parameters and
0.035 for the eight raw feedback gains, sampled once per episode. Episodic PPO
uses final undiscounted landing score times 0.01, actor LR 5e-5, critic LR 3e-4,
two epochs, four minibatches and the existing global 0.03 KL rollback. Episodes
whose landing parameters never activate are excluded from PPO. There is no
continuous height-offset integration, 50 Hz exploration noise or extra action
regularization in this episodic objective.

Formal training is 512 environments and 128 updates. The 45 delay/initial-state
conditions repeat across these worlds. Deterministic evaluation occurs every
eight updates; checkpoints only replace the selected baseline if every world
passes, mean wheel clearance remains at least 9.065 cm, mean COM rise remains
at least 11.74 cm, and landing return improves. Latest-policy failures are always
reported even if an older qualifying model is retained.

The source contract includes both unchanged parent manifests and this directory.
CPU checks cover reward semantics, initial-reference equivalence, bounded recovery,
mixed-world gates, parameter locking and a real masked PPO update. The 45-world
native smoke includes two updates, readback/selective reset, and same-state command
and 400 Hz FIFO checks. Repeated GPU trajectories are diagnostics, not a claim of
bitwise replay. Independent post-training evaluation uses the original physics
implementation and records baseline, selected and latest policies separately.
All three get 400 Hz trace audits and actual process exit receipts.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File experiments\v7_rs02_24v_param_landing\run_pipeline.ps1 -PythonPath D:\RL_JUMP\.venv\python.exe -RunId train_01
```

Use fresh run IDs. STOP files, the shared exclusive physics lock and GPU reserve
checks remain enabled. Code syncs through GitHub; runs, logs, traces, caches and
generated checkpoints remain local and must never be force-added as a directory.
No zero-assistance, unseen-condition or hardware qualification is claimed.

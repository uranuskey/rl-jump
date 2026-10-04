# Guided continuation of method 1

Start from the independently audited parent update 120 (mean peak 427.23 N,
45/45 complete landings). Preserve the 13D actor's action meaning and all bounds,
fixed launch, apex gate, contact physics, 400 Hz guards, 50 Hz control, 0..20 ms
FIFO latency, 24 V envelope and 62.5% attitude assistance. No vertical force is
added. The old experiment and its checkpoints remain unchanged.

First evaluate 20 deterministic curve proposals plus matched baselines in two
512-world batches. Each complete proposal group contains all 45 conditions.
Proposals change only five bounded curve logits: approach height/time, cushion
depth/time, and standing recovery time. Feedback gains are inherited. A candidate
must retain height, pass all 45 cases and lower mean peak by at least 1 N. Transfer
its exact logit offset into the actor's final bias, then validate the resulting
conditional actor on all 512 worlds. Revert to the parent actor if that validation
does not pass or fails to improve force by 0.5 N. Search samples are never passed
to on-policy PPO. Search and PPO improvements are reported separately.

The old ten reward terms remain, with impact weight increased from 15 to 90 and
failure/height-loss penalties increased from 100 to 300. Three guidance terms add:

- Up to 20 points for actual COM descent during first deceleration, gated by a
  complete stable landing. Stop measuring when COM velocity first reaches -0.05
  m/s, and in all cases after 250 ms. A later crouch receives no extra credit.
  The target stroke is 1.25 times incident vertical energy divided by (300 N minus
  body weight), bounded to 4..6 cm. More stroke beyond the target gains nothing.
- A penalty for squared force excess above 300 N integrated over the first 20 ms.
  The original full-landing maximum remains penalized, so a delayed spike is not
  hidden and 2.5 ms contact spikes are retained.
- Up to 20 points for approaching a 300 N peak while completing the landing.

300 N is a research target, not a proven mechanical limit or hardware rating.
All rewards use measured trajectories, not requested knee/leg extension.
Selection requires every world to pass with mean wheel clearance >=9.065 cm and
COM rise >=11.74 cm, then lower actual mean peak force by at least 0.25 N.

Continue for 128 NEW PPO updates at 512 environments. Keep the actor from parent
update 120, but reset critic and Adam because the score changes; this is not a
bitwise resume. Curve exploration standard deviations begin at
[0.18,0.18,0.22,0.20,0.10] raw logits and decrease during training; the 8 gain
standard deviations stay 0.035. Actor LR remains 5e-5, critic LR 3e-4, and the
global 0.03 KL rollback remains. Exploration is still sampled once per jump.

Two smoke updates at 45 worlds must end with 45/45 deterministic qualification
before formal training. Formal evaluation occurs every 8 updates; zero-pass
latest evaluations stop execution. Selected and latest results are always
reported separately. The final original-backend evaluation records four sets:
inherited parent baseline, guided seed, selected model, and latest model. All four
have 45 cases, full traces, frozen-source/checkpoint checks and an independent
first-deceleration-stroke audit. Actual train/eval/audit exit receipts are required.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File experiments\v7_rs02_24v_guided_landing\run_pipeline.ps1 -PythonPath D:\RL_JUMP\.venv\python.exe -RunId train_01 -Checkpoint D:\RL_JUMP\repo\experiments\v7_rs02_24v_param_landing\runs\train_01\model_0120.pt
```

Only source goes to GitHub. Generated models, search trials, traces and logs stay
local. A successful result is limited to the retained assisted simulation cases.

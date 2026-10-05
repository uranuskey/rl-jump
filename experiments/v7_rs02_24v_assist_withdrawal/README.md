# Gradual removal of virtual attitude assistance

The source is slot-landing train_02 selected update 112. Keep its landing
controller, fixed takeoff network and observation override, 13 reward terms,
24 V estimated motor envelope, FIFO, collision and landing gates unchanged.
No source checkpoints or training outputs belong in the public repository.

First reproduce the source at 62.5% assistance in native45. Evaluate the same
weights at 60%; if this is not qualified, try 61.25%. These are whole-episode
strengths multiplied by the existing 0.50-0.60 s ramp. They act on pitch/roll
torque only: the cap changes from 12.5 Nm to 12.0/12.25 Nm. There is no virtual
vertical lift. Fixed network weights do not imply identical takeoff plans
when the physical state changes.

Admission requires all cases passing, at most 1% mean height loss versus the
matched 62.5% replay, retained actual COM deceleration stroke, at most 2% mean
and 3% worst peak-force increase, no rebound, a full second of continuous
stability, and recovery no more than 0.10 s slower. All old per-case physical
height and landing requirements remain active. Passing native45 alone is not
permission to skip a smoke test, the 512-world validation, or final evaluation
before promoting a later trained policy. A failed reduction leaves the old
qualified model available. No automatic larger assistance drop occurs here.

The independent rollout avoids the legacy helper that resets assistance to
62.5%. It records actual strength, ramp, orientation and applied wrench at
400 Hz. The audit recomputes every assistance torque, rejects any linear/yaw
force, and runs the existing physical, FIFO, schedule and slot-path audits.
512 worlds repeat the same 45 initial-condition/delay cases; they are not 512
distinct real-world scenarios. None of these tests qualifies real hardware.

Run the bounded probe on the configured remote environment:

```powershell
.\run_probe.ps1 -PythonPath D:\RL_JUMP\.venv\python.exe -RunId probe_01
```

Results, traces and real child exit receipts are saved under ignored `runs/`.

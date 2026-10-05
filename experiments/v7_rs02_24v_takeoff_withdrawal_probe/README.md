# Takeoff and landing assistance withdrawal

The objective is the lowest **qualified attitude assistance**, including crouch
and takeoff. Further reduction of impact force is not required. The existing
24 V estimated motor envelope, physics, collisions, constraint tolerances,
controller, policy weights, height/stroke/impact/stability/recovery gates and
strict zero-rebound requirement are unchanged.

The preceding landing-only PPO was deliberately stopped at update 80 when the
user changed priority. Its real exit code 1 and original `ERROR_STOPPED` result
remain intact; this is not a completed 128-update run. The newest intermediate
strict-passing checkpoint is only a candidate until independent qualification.

This probe checks the original 62.5% reference and then these bounded levels:

| Step | Crouch/takeoff to COM apex | After apex |
| --- | --- | --- |
| Current checkpoint qualification | 62.5% | 60% |
| First takeoff reduction | 61.25% | 60% |
| Uniform reduction | 60% | 60% |
| Next uniform reduction | 57.5% | 57.5% |

The existing entry ramp from 0.5 to 0.6 s is preserved. The post-apex transition
uses the observed COM apex and a 100 ms smooth ramp. Only roll/pitch virtual
torque is present; there is no linear lifting force or yaw assistance.

Each level needs native45 with full 400 Hz physical/controller/assistance audits
and all three fresh 512-world replays. The original reference needs native45 and
one 512 replay. The 512 worlds repeat 45 cases. The first failed level stops
further withdrawal; failures are retained and are never retried until passing.
Impact remains bounded by the inherited limits; no improvement threshold or
force-based ranking controls advancement. Assistance traces independently
recompute the actual applied torque, including the pre-apex interval.

The test holds launch and landing weights fixed. Lower assistance may change
the state and therefore the frozen launch policy's output; numerical identity
of the takeoff trajectory across assistance levels is not claimed. The old
landing actor is causally inactive before the apex, so a failed takeoff probe
does not justify more landing-only PPO as takeoff adaptation.

Run the supervisor through WMI so it survives SSH closure, passing `-RunId`
and `-Handoff`. Code is synchronized through GitHub; handoffs, checkpoints,
traces, results and receipts stay in ignored `runs/` directories. `AUDITED`
confirms checked evidence, not necessarily a successful withdrawal. There is
no optimizer in this probe, no automatic further training or lower level beyond
the table, and no unassisted or hardware qualification.

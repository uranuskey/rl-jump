# Learn small rebounds without declaring withdrawal qualified

The user authorized this change after the parent apex-assistance run stopped
before PPO. That run completed its two-update smoke (exit 0), then its first
512-world validation stopped (exit 1) because two worlds recorded unsupported
upward COM speeds of 0.07815 and 0.06662 m/s. All 512 worlds completed the
task, retained height and recovered stability; only the additional zero-
rebound gate failed. The old failed run and all frozen parent code remain.

Separate learning entry from promotion. Entry retains every existing
physical, all-case completion, height, stroke, impact, recovery and stability
gate, but permits rebound velocity up to 0.10 m/s. This is an explicit
simulation training bound, not a hardware safety limit. The inherited
13-term reward still penalizes rebound. No reward, policy topology, physics,
controller or assistance schedule changes. Takeoff uses 62.5% attitude
assistance; the observed COM apex triggers a 100 ms ramp to 60%. It is not
uniform 60% assistance. The 24 V estimated motor curve is unchanged.

Reuse the completed parent smoke and its complete 512-world evidence, with
source, checkpoint, result and receipt verification. The preflight preserves
the historical validation's ERROR_STOPPED and real exit 1; it does not call
that old run passed. It records a new learning-entry decision under this
explicitly different contract. Formal PPO performs a fresh 512-world initial
replay before accepting any update. No repeated replay-until-pass loop exists.

Train 512 worlds for 128 new PPO updates from slot selected112 with fresh
Adam, unchanged exploration and 0.03 KL rollback/backtracking. The parent
smoke checkpoint is not the initial policy. Evaluate at update 2 and then
every 8 updates. Only a checkpoint meeting the original strict zero-rebound
qualification may become selected. The first qualified checkpoint need not
beat an unqualified seed's force. If none qualifies, selected remains null;
an explicitly diagnostic candidate is evaluated instead, without promotion.

After the budget, independently evaluate reference625, seed, selected (or
diagnostic candidate) and latest in native 45-world runs with full physical,
controller, slot and 400 Hz assistance audits. Then replay reference and
seed once at 512 worlds, and selected and latest three times each at 512
worlds. All three final batches plus the native45 audit must meet the
original strict qualification. A 512-world batch repeats 45 initial-condition
and delay cases; it does not introduce 512 distinct scenarios. The batch
replays save complete per-case metrics and constraint diagnostics, not full
per-step traces. Report every failure; do not cherry-pick a passing repeat.

The audit independently checks actual exits, frozen hashes, reused historical
evidence, accepted actor steps, selection history and final replays. AUDITED
means evidence was checked, not that the policy qualified. Report PPO rebound
and force changes separately from the phase-assistance change. Do not claim
zero-assistance, hardware or calibrated-motor qualification. No additional
assistance reduction or automatic retraining follows. Reuse the hidden WMI
supervisor, preserve failed artifacts and synchronize source only via GitHub.

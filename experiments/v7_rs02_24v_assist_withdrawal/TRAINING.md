# One admitted assistance step

The PPO pipeline consumes a completed, audited withdrawal probe with a real
zero exit receipt. It never tries a lower assistance level on its own. A
45-world two-update smoke and a 512-world deterministic validation must pass
before 128 new PPO updates can start. All outputs use a fresh run directory.

Initialize the actor/critic from slot selected112, keep the 16D compliant
controller and fixed launch, and create fresh Adam optimizers. Exploration
standard deviations gradually decrease from .035 to .025 for the first eight
action components and from .012 to .008 for the eight feedback gains. The
existing PPO optimizer retains its full parameter/Adam rollback and halved
learning-rate retries under the .03 KL limit. The 13-term reward is unchanged.

The deterministic initial policy must pass admission at the selected lower
assistance. Evaluation runs every eight updates, with a separate selection
gate that retains the initial model if no qualified improvement is found.
Exploratory training samples are never counted as deterministic qualification.
Final native45 evaluation compares the source at62.5%, initial lower-assist
seed, selected and latest, including recorded 400Hz assistance audits. A final
audit verifies accepted actor steps, checkpoint hashes and real stage exits.
Latest may be worse than selected and must be reported separately. No second
assistance reduction or automatic retraining follows this pipeline.

`supervise.ps1` records the wrapper's real process exit separately from all
Python stage exits. When launched over Windows OpenSSH, use a WMI-created
supervisor outside the SSH process job, then verify it survives connection
closure; ordinary `Start-Process` alone did not persist in the first probe
launch. The `-SelfTest` mode only waits12seconds and records a child exit (0,
or deliberately7 with `-SelfTestExitCode 7`); it
does not import the simulator, acquire the physics lock or start training.
Use hidden windows and preserve all failed launch/run directories.

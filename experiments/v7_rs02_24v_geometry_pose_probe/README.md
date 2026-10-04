# Landing pose / slot alignment diagnostic

Read saved failed trajectories and compare their actual configuration with the
original CAD squat path at exactly the same leg heights. Reconstruct contacts
using the unchanged full-collision MuJoCo model on CPU. Keep the original robot
collision filters; report all robot penetrations below -10 micrometres without
requiring a force threshold. This is static geometry, not a new dynamic policy.

`cad_path.json` contains the original 81 CAD input configurations and provenance,
not training trajectories. The original validation covers upright synchronized
legs, 2 mm height samples, and +/-3 mm fore-aft offsets at five heights. It does
not establish a rectangular safe workspace in height and joint angle.

The script preserves outputs and hashes the source trace and geometry. All
results stay ignored under `runs/`. Do not edit geometry, collision filters, or
the frozen controller to make this diagnostic pass.

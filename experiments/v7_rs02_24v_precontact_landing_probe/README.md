# Precontact landing diagnostic

Independent successor to the staged-impedance diagnostic. Preserve the frozen
v3 controller, selected56 checkpoint, fixed takeoff, estimated 24 V motor curve,
62.5% attitude assistance, physical limits and 15-channel motor FIFO.

First reconstruct wheel-to-ground closing speed and remaining leg travel from
saved 45-case native traces. This CPU analysis does not run or train a policy.
Subsequent candidates will change precontact preparation, with native landing
acceptance and actual COM stroke checked independently of commanded leg travel.

Reference: Sato et al., IEEE Access 2022, DOI 10.1109/ACCESS.2022.3153127:
precontact foot velocity and postcontact support are separate control problems.
This diagnostic does not claim that their single-leg hardware reduction carries
over to this wheeled biped. All outputs under `runs/` remain ignored and private.

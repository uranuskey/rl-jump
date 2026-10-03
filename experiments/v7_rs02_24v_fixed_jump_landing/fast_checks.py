"""Device-side predicates preserve the old guard order and per-tick coverage."""
import torch


def external_fault(external, base, assist):
    return (external[:, base]!=assist).any() | (external[:, :, :3]!=0).any()


def post_step_faults(overflow, q, v, acc, active, force, main, effort, aux, actuator_force):
    numeric = (overflow | ~torch.isfinite(q).all(1) | ~torch.isfinite(v).all(1)
               | ~torch.isfinite(acc).all(1)) & active
    force_path = ((force[:, main]-effort).abs().max()>1e-6) | (
        force[:, aux].abs().max()>0) | (actuator_force.abs().max()>0)
    return torch.stack((numeric.any(), force_path))

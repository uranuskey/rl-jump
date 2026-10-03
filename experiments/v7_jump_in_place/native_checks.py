"""Read back native geometry, COM and contacts without advancing time."""
import numpy as np
import torch
import mujoco
import mujoco_warp as mjw
from geometry import mesh_bottom_clearance


def readback(env, world=0):
    torch.cuda.synchronize()
    copied = mujoco.MjData(env.m)
    if int(env.ticks[world]) == 0:
        # Selective reset deliberately clears the mass/contact workspace. The
        # generic get_data_into calls mj_factorM even for this uncomputed matrix;
        # read only the valid reset kinematic fields until the first full step.
        copied.qpos[:] = env.q[world].cpu().numpy()
        copied.qvel[:] = env.v[world].cpu().numpy()
        copied.geom_xpos[:] = __import__('warp').to_torch(env.gd.geom_xpos)[world].cpu().numpy()
        copied.time = float(env.time[world])
    else:
        mjw.get_data_into(copied, env.m, env.gd, world_id=world)
    # Independent kinematic/velocity recomputation at exactly the GPU q/v.
    reference = mujoco.MjData(env.m)
    reference.qpos[:] = copied.qpos
    reference.qvel[:] = copied.qvel
    # Only kinematics/COM are compared with this independent CPU reference.
    # Contact forces below are reconstructed from the copied GPU constraint data.
    mujoco.mj_kinematics(env.m, reference)
    mujoco.mj_comPos(env.m, reference)
    mujoco.mj_comVel(env.m, reference)
    mujoco.mj_subtreeVel(env.m, reference)
    bottoms = []
    for ids in env.wheel_geoms:
        values = []
        for gid in ids:
            mesh = env.m.geom_dataid[gid]
            adr, count = env.m.mesh_vertadr[mesh], env.m.mesh_vertnum[mesh]
            values.append(mesh_bottom_clearance(env.m.mesh_vert[adr:adr+count], reference.geom_xmat[gid].reshape(3, 3), reference.geom_xpos[gid]))
        bottoms.append(min(values))
    support = np.zeros(2)
    for idx, contact in enumerate(copied.contact):
        a, b = (int(env.m.geom_bodyid[int(g)]) for g in contact.geom)
        if a != 0 and b != 0:
            continue
        body = b if a == 0 else a
        if body not in env.wheels:
            continue
        f = np.zeros(6)
        mujoco.mj_contactForce(env.m, copied, idx, f)
        world_force = contact.frame.reshape(3, 3).T@f[:3]
        support[env.wheels.index(body)] += max(0., world_force[2]*(1 if a == 0 else -1))
    metrics = dict(
        q_copy_error=float(abs(copied.qpos-env.q[world].cpu().numpy()).max()),
        geometry_error_m=float(abs(reference.geom_xpos-copied.geom_xpos).max()),
        clearance_error_m=float(abs(np.array(bottoms)-env.clearance[world].cpu().numpy()).max()),
        com_position_error_m=float(abs(reference.subtree_com[env.base]-env.com[world].cpu().numpy()).max()),
        com_velocity_error_mps=float(abs(reference.subtree_linvel[env.base]-env.com_v[world].cpu().numpy()).max()),
        contact_force_readback_error_n=float(abs(support-env.support[world].cpu().numpy()).max()),
        native_time_s=float(copied.time), logical_time_s=float(env.ticks[world])*.0025,
        wheel_clearance_m=bottoms, wheel_support_n=support.tolist())
    limits = dict(q_copy_error=1e-8, geometry_error_m=2e-6, clearance_error_m=2e-6,
                  com_position_error_m=2e-6, com_velocity_error_mps=2e-5, contact_force_readback_error_n=2e-3)
    bad = {k: v for k, v in metrics.items() if k in limits and v > limits[k]}
    if bad:
        raise RuntimeError('Native post-step sensor readback failed: '+str(bad))
    if abs(metrics['native_time_s']-metrics['logical_time_s']) > 2e-5:
        raise RuntimeError('Native clock does not match sampled physics ticks')
    return metrics


def selective_reset(env):
    """Reset one world and require every neighbour's dynamic/history/FIFO state unchanged."""
    neighbours = torch.arange(1, env.n, device=env.device)
    names = ('q', 'v', 'warm', 'time', 'previous', 'ticks', 'raw_history', 'command_history', 'episode_id')
    old = {name: getattr(env, name)[neighbours].clone() for name in names}
    fifo = env.fifo.history[neighbours].clone()
    delays = env.fifo.delay_steps[neighbours].clone()
    mask = torch.zeros(env.n, dtype=torch.bool, device=env.device)
    mask[0] = True
    env.reset(mask, fixed_delay=8)
    for name in names:
        if not torch.equal(old[name], getattr(env, name)[neighbours]):
            raise RuntimeError('Selective reset modified neighbouring '+name)
    if not torch.equal(fifo, env.fifo.history[neighbours]) or not torch.equal(delays, env.fifo.delay_steps[neighbours]):
        raise RuntimeError('Selective reset modified neighbouring FIFO')
    if env.fifo.history[0].abs().max() != 0 or env.previous[0].abs().max() != 0:
        raise RuntimeError('Reset did not clear the selected action/FIFO history')
    return dict(status='PASS', neighbours=env.n-1, fields=list(names)+['FIFO', 'delay'], selected_delay_ticks=8)

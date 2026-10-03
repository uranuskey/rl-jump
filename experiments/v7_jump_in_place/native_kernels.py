"""GPU kernels for the isolated jump environment."""
import warp as wp


@wp.kernel
def restore_paused(mask: wp.array[bool], sq: wp.array2d[float], sv: wp.array2d[float],
                   sw: wp.array2d[float], st: wp.array[float], q: wp.array2d[float],
                   v: wp.array2d[float], warm: wp.array2d[float], t: wp.array[float]):
    i = wp.tid()
    if mask[i]:
        for j in range(q.shape[1]):
            q[i, j] = sq[i, j]
        for j in range(v.shape[1]):
            v[i, j] = sv[i, j]
            warm[i, j] = sw[i, j]
        t[i] = st[i]


@wp.kernel
def reset_pose(mask: wp.array[bool], poses: wp.array2d[float], q: wp.array2d[float]):
    i, j = wp.tid()
    if mask[i]:
        q[i, j] = poses[i, j]


@wp.kernel
def contact_metrics(ncon: wp.array[int], geom: wp.array[wp.vec2i], world: wp.array[int], body: wp.array[int],
                    frame: wp.array[wp.mat33], dist: wp.array[float], force: wp.array[wp.spatial_vector],
                    left: int, right: int, support: wp.array2d[float], bad: wp.array2d[float]):
    c = wp.tid()
    if c < ncon[0]:
        w = world[c]
        a = int(body[geom[c][0]])
        b = int(body[geom[c][1]])
        f = wp.spatial_top(force[c])
        fw = f @ frame[c]
        if a == 0 or b == 0:
            rb = int(0)
            sign = float(1.)
            if a == 0:
                rb = b
            else:
                rb = a
                sign = -1.
            wheel = int(-1)
            if rb == left:
                wheel = 0
            elif rb == right:
                wheel = 1
            if wheel >= 0:
                wp.atomic_add(support, w, wheel, wp.max(0., sign*fw[2]))
            else:
                wp.atomic_max(bad, w, 0, wp.length(f))
        elif f[0] > 1.e-4 and dist[c] < -1.e-5:
            wp.atomic_max(bad, w, 1, f[0])


@wp.kernel
def wheel_bottom(vertices: wp.array[wp.vec3], geomids: wp.array[int], wheelids: wp.array[int],
                 xpos: wp.array2d[wp.vec3], xmat: wp.array2d[wp.mat33], clearance: wp.array2d[float]):
    world, vertex = wp.tid()
    g = geomids[vertex]
    p = xmat[world, g] @ vertices[vertex] + xpos[world, g]
    wp.atomic_min(clearance, world, wheelids[vertex], p[2])

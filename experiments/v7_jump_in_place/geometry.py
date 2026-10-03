"""Pure geometry helpers; native collider transforms still need simulator parity."""
import numpy as np


def mesh_bottom_clearance(vertices, rotation_world, origin_world, ground_z=0.0):
    """Exact minimum vertex z for one convex collider over a flat plane.

    vertices and transform MUST share the collider's local frame. MuJoCo mesh
    compilation can recenter/rotate vertices; use compiled mesh + geom pose
    together, not source vertices combined with a compiled geom pose.
    For each wheel, take the minimum over all its collision geoms.
    """
    v, r, p = (np.asarray(x, dtype=float) for x in (vertices, rotation_world, origin_world))
    if v.ndim != 2 or v.shape[1] != 3 or len(v) == 0 or r.shape != (3, 3) or p.shape != (3,):
        raise ValueError("Expected nonempty vertices[N,3], R[3,3], position[3]")
    if not all(np.isfinite(x).all() for x in (v, r, p)) or not np.isfinite(ground_z):
        raise ValueError("Nonfinite geometry")
    if not np.allclose(r.T@r, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(r), 1, atol=1e-6):
        raise ValueError("Expected proper rotation matrix")
    return float((v@r[2]).min()+p[2]-ground_z)

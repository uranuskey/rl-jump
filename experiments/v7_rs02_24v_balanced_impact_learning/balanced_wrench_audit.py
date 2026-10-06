"""Independent actual applied-wrench audit without GPU dependencies."""
import numpy as np

def external_arrays(z,before,after):
    from balanced_contract import checked_levels
    checked_levels(before,after)
    body=z['sensor_external_body_wrench']
    root=z['sensor_external_root_generalized_force']
    assert body.ndim==4 and body.shape[-1]==6 and root.shape==(*body.shape[:2],6)
    assert np.isfinite(body).all() and np.isfinite(root).all()
    ids=z['sensor_external_base_body_id']
    assert np.all(ids==ids.flat[0])
    base=int(ids.flat[0])
    assert np.array_equal(body[...,base,:],z['assist_wrench'])
    nonbase=body.copy();nonbase[...,base,:]=0
    assert np.all(nonbase==0), 'Hidden body assistance'
    assert np.all(root==0), 'Hidden generalized free-base assistance'
    assert np.all(body[...,:3]==0), 'External linear assistance'
    if before==after==0:
        assert np.all(body==0), 'Nonzero actual wrench at zero-assistance level'
    return dict(status='PASS',every_body_checked=True,free_base_checked=True,
        max_abs_linear_force_n=float(abs(body[...,:3]).max(initial=0)),
        max_abs_body_torque_nm=float(abs(body[...,3:]).max(initial=0)),
        max_abs_root_generalized_force=float(abs(root).max(initial=0)),
        zero_external_wrench=bool(np.all(body==0) and np.all(root==0)))

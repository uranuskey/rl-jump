"""Small deterministic pose evidence checks independent of the GPU runtime."""
import hashlib
import numpy as np

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def check_pose(path,expected_digest,worlds):
    assert digest(path)==expected_digest
    with np.load(path,allow_pickle=False) as z:
        for key in ('q','v','ticks','minimum_q'):
            assert len(z[key])==worlds and np.isfinite(z[key]).all(),key
        assert z['q'].ndim==z['v'].ndim==z['minimum_q'].ndim==2
        assert z['ticks'].shape==(worlds,) and (z['ticks']>0).all()
        assert z['q'].shape==z['minimum_q'].shape

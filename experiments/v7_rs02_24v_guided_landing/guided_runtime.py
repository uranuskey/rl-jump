"""Own immutable source contract; all parent source contracts remain unchanged."""
import time
import guided_paths
from guided_paths import HERE,ROOT,PARENT
from param_runtime import verify as verify_parent, sha, write, read, exclusive, now


def verify():
    parent=verify_parent()
    manifest=read(HERE/'FROZEN.json')
    assert manifest['parent_sha256']==parent
    for rel,digest in manifest['sha256'].items():
        if sha(ROOT/rel)!=digest:
            raise RuntimeError('Guided source changed: '+rel)
    return sha(HERE/'FROZEN.json')


def resource_limit(started,seconds):
    def limit():
        import torch
        if (HERE/'STOP').exists() or (HERE.parent/'v7_jump_in_place/STOP').exists():
            raise RuntimeError('STOP requested')
        if time.monotonic()-started>seconds:
            raise RuntimeError('Wall time bound')
        if torch.cuda.mem_get_info()[0]<512*1024**2:
            raise RuntimeError('GPU reserve')
    return limit

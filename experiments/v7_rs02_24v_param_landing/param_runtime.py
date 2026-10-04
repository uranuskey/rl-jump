"""New source contract, existing exclusive physics lock and atomic reporting."""
from datetime import datetime, timezone
import time
import path_setup
from path_setup import HERE, ROOT
from runtime import sha, write, exclusive
from resume_state import read, verify_resume


def verify():
    policy_hash, fast_hash = verify_resume()
    path = HERE/'FROZEN.json'
    manifest = read(path)
    assert manifest['parent_policy_sha256']==policy_hash and manifest['parent_fast_sha256']==fast_hash
    for rel, digest in manifest['sha256'].items():
        if sha(ROOT/rel)!=digest:
            raise RuntimeError('Parameter task source changed: '+rel)
    return sha(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def resource_limit(started, seconds):
    def limit():
        import torch
        if (HERE/'STOP').exists() or (HERE.parent/'v7_jump_in_place/STOP').exists():
            raise RuntimeError('STOP requested')
        if time.monotonic()-started>seconds:
            raise RuntimeError('Wall time bound')
        if torch.cuda.mem_get_info()[0]<512*1024**2:
            raise RuntimeError('GPU reserve')
    return limit

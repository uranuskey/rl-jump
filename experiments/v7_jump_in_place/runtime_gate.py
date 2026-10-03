"""Fail-closed local execution scope and one-physics-process guard."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def processes():
    script = "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; @(Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CommandLine) | ConvertTo-Json -Compress"
    r = subprocess.run(['powershell.exe', '-NoProfile', '-Command', script], capture_output=True, encoding='utf-8', timeout=20, check=True)
    return json.loads(r.stdout.lstrip('\ufeff'))


def runtime_conflicts():
    rows = processes()
    parents = {r['ProcessId']: r['ParentProcessId'] for r in rows}
    ancestors, current = set(), os.getpid()
    while current and current not in ancestors:
        ancestors.add(current)
        current = parents.get(current)
    return [r for r in rows if r['ProcessId'] not in ancestors
            and (r['Name'].lower().startswith(('python', 'kit', 'isaac')))]


@contextmanager
def exclusive_runtime(stage):
    scope = json.loads((HERE/'EXECUTION_SCOPE.json').read_text(encoding='utf-8'))
    expected = {'admission': 'bounded_physics_admission', 'smoke': '16_env_2_update_ppo_smoke'}
    if stage not in expected or expected[stage] not in scope['authorized_now']:
        raise RuntimeError('Stage 4 requires a later explicit user command; this runner only implements stages 1--3')
    if scope['stage4']['authorized'] is not False:
        raise RuntimeError('Unexpected stage4 scope mutation')
    conflicts = runtime_conflicts()
    if conflicts:
        raise RuntimeError('Other Python/physics process still running: '+json.dumps(conflicts, ensure_ascii=False))
    free = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                          capture_output=True, text=True, check=True, timeout=10)
    if int(free.stdout.splitlines()[0]) < 512:
        raise RuntimeError('GPU free memory below 512 MiB')
    lock = HERE/'PHYSICS_LOCK.json'
    receipt = dict(pid=os.getpid(), stage=stage, utc=datetime.now(timezone.utc).isoformat(), free_gpu_mib=int(free.stdout.splitlines()[0]))
    with lock.open('x', encoding='utf-8') as f:
        json.dump(receipt, f, indent=2)
    try:
        yield receipt
    finally:
        if lock.exists() and json.loads(lock.read_text(encoding='utf-8'))['pid'] == os.getpid():
            lock.unlink()

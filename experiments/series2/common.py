"""Small, dependency-free utilities shared by the overnight jobs."""
from __future__ import annotations
import hashlib, json, os, signal, time
from pathlib import Path

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    temporary.replace(path)

def read_json(path, default=None):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else default

def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def tensor_id(path, expert):
    return hashlib.sha256(path.encode()).hexdigest()[:16] + f'-e{expert:03d}'

def event(name, **fields):
    print(json.dumps({'event': name, 'unix_time': time.time(), **fields}, allow_nan=False), flush=True)

def install_deadline(seconds):
    def stop(signum, frame): raise TimeoutError('Phase deadline or termination signal reached')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM): signal.signal(sig, stop)
    if seconds: signal.alarm(max(1, int(seconds)))

def directory_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())

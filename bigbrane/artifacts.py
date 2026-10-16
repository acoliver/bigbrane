"""Atomic records, frozen inputs, and reproducibility metadata."""
from __future__ import annotations
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import platform
from pathlib import Path
import resource
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')


def configure_threads():
    for key in THREAD_ENV:
        os.environ[key] = '1'


def hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def save_array(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, value, allow_pickle=False)
    temporary.replace(path)


def peak_rss_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == 'darwin' else rss*1024)


def environment():
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        np.show_config()
    cpu = platform.processor()
    if sys.platform == 'darwin':
        cpu = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    return {'python': sys.version, 'executable': sys.executable, 'os': platform.platform(),
            'cpu': cpu, 'machine': platform.machine(), 'numpy': np.__version__, 'torch': torch.__version__,
            'packages': dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions())),
            'thread_environment': {k: os.environ.get(k) for k in THREAD_ENV},
            'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
            'numpy_build': config.getvalue(), 'torch_build': torch.__config__.show(),
            'dtypes': {'hmm': 'float64', 'neural_train': 'float32', 'neural_log_normalization': 'float64'},
            'prng': 'NumPy PCG64(SeedSequence(integer tuple))',
            'neural_seed_conversion': 'SeedSequence(tuple).generate_state(1, dtype=uint64)[0] -> int -> torch.manual_seed',
            'initializers': ['torch.nn.init.normal_(mean=0,std=0.1)', 'torch.nn.init.orthogonal_(gain=1)',
                             'torch.nn.init.xavier_uniform_(gain=1)'],
            'device': 'cpu', 'deterministic_algorithms': True,
            'thread_enforcement': 'backend environment set before imports; torch intra/inter-op=1; see recorded build settings'}


def source_files():
    return sorted([ROOT/'EXPERIMENT.md', ROOT/'RESEARCH.md', ROOT/'README.md', ROOT/'pyproject.toml',
                   *ROOT.joinpath('bigbrane').glob('*.py'), *ROOT.joinpath('tests').glob('*.py')])


def freeze_sources(out):
    inventory = {}
    for source in source_files():
        relative = str(source.relative_to(ROOT)); target = out/'frozen'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        inventory[relative] = hash_file(target)
    return inventory


def verify_sources(out, inventory):
    current = {str(p.relative_to(ROOT)): hash_file(p) for p in source_files()}
    if current != inventory:
        raise ValueError('source or protocol changed since freeze; start a separately labeled run')
    for name, digest in inventory.items():
        if hash_file(out/'frozen'/name) != digest:
            raise ValueError(f'frozen source changed: {name}')


def verify_inventory(base, inventory):
    for relative, digest in inventory.items():
        if hash_file(Path(base)/relative) != digest:
            raise ValueError(f'artifact hash mismatch: {relative}')


class Ownership:
    """An inherited flock keeps the run owned even if its driver is killed."""

    def __init__(self, path):
        self.path = Path(path)
        self.fd = None

    def __enter__(self):
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(self.fd); self.fd = None
            raise RuntimeError(f'directory is owned by a live driver or worker: {self.path.parent}') from error
        return self

    def __exit__(self, *_):
        # Do not LOCK_UN: a surviving child shares this open file description.
        os.close(self.fd)
        self.fd = None


def process_identity():
    import uuid
    return {'pid': os.getpid(), 'token': uuid.uuid4().hex}


def verify_provenance(out, manifest):
    verify_sources(out, manifest['source_hashes'])
    for stage_id, stage in manifest['data_stages'].items():
        if read_json(Path(out)/f'data-stage-{stage_id}.json') != stage:
            raise ValueError('stage registry changed since freeze')
        verify_inventory(out, stage['hashes'])
    if 'environment_hash' in manifest:
        verify_inventory(out, {'frozen/environment.json': manifest['environment_hash']})
    for selection in manifest.get('selections', {}).values():
        verify_inventory(out, {selection['path']: selection['hash']})

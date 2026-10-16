"""Command line composition root. CPU thread settings precede backend imports."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'


def main():
    parser = argparse.ArgumentParser(description='Fixed version-one HMM, RNN and attention experiment')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--root', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prepare-only', action='store_true', help='freeze inputs without fitting or test evaluation')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        from .fitting import worker
        worker(args.worker)
        return
    from .runner import run
    root = args.root if args.root is not None else (20260931 if args.pilot else 20260930)
    out = args.output or Path('results')/(('pilot-' if args.pilot else 'experiment-001-')+time.strftime('%Y%m%d-%H%M%S'))
    run(root, args.pilot, out, resume=args.resume, prepare_only=args.prepare_only)


if __name__ == '__main__':
    main()

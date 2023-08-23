import argparse
from dataclasses import dataclass
import json
import os
from typing import List

from KAS import Path


@dataclass
class TrialMetadata:
    path: Path
    accuracy: float
    flops: int
    params: int
    time: float

def get_trial_metadata(dir: os.PathLike) -> List[TrialMetadata]:
    # Get Path
    parser = argparse.ArgumentParser(description='KAS session plot')
    args = parser.parse_args()
    assert args.dirs is not None
    for dir in args.dirs:
        assert os.path.exists(dir) and os.path.isdir(dir)

    # Read
    kernels = []
    for kernel_fmt in os.listdir(dir):
        kernel_dir = os.path.join(dir, kernel_fmt)
        if not os.path.isdir(kernel_dir):
            continue
        if 'ERROR' in kernel_dir:
            continue
        files = list(os.listdir(kernel_dir))
        assert 'graph.dot' in files and 'loop.txt' in files and 'meta.json' in files

        meta_path = os.path.join(kernel_dir, 'meta.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        kernels.append(TrialMetadata(
            path=Path.deserialize(meta['path']),
            accuracy=meta['accuracy'],
            flops=meta['flops'],
            params=meta['params'],
            time=meta['time'],
        ))
    kernels.sort(key=lambda x: x.time)
    return kernels

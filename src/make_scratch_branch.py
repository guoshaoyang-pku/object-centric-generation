"""Write a step-0 frozen joint_flow.py checkpoint so stable_flow_ddp.py can train a new size from scratch (amendment G).

stable_flow_ddp.py only starts from a branch checkpoint: it loads a plain
SmallObjectFlow, calibrates the QK-norm gains so the first pass reproduces it,
and checks that identity. This script builds that plain flow with joint_flow.build
(same seeding as every earlier run), at any width/depth/heads, without training it,
and saves it with step = samples = train_seconds = 0. Nothing is inherited, so the
resulting run is a from-scratch run with the stable recipe, and all its GPU-hours
are its own.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joint_flow as jf  # noqa: E402
from codec_research import CacheBank, canonical_sha, digest  # noqa: E402
from codec_flow_research import flow_state  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['data', 'cache', 'codec-checkpoint', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--codec-sha', required=True)
    p.add_argument('--seed', type=int, default=713)
    p.add_argument('--width', type=int, required=True)
    p.add_argument('--depth', type=int, required=True)
    p.add_argument('--heads', type=int, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit('Refusing to overwrite ' + str(a.out))
    bank = CacheBank(a.data, a.cache)
    build_args = argparse.Namespace(codec_checkpoint=a.codec_checkpoint, codec_sha=a.codec_sha, seed=a.seed,
                                    small_width=a.width, small_depth=a.depth, small_heads=a.heads, mode='frozen')
    model, codec_identity = jf.build(build_args, bank, torch.device('cpu'))
    protocol = {'format': jf.FORMAT, 'mode': 'frozen', 'seed': a.seed, 'init': 'scratch (joint_flow.build, untrained)',
                'small_architecture': [a.width, a.depth, a.heads], 'codec_sha256': a.codec_sha,
                'script_sha256': digest(Path(__file__))}
    params = sum(q.numel() for q in model.flow.parameters())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'format': jf.FORMAT, 'signature_sha256': canonical_sha(protocol), 'protocol': protocol,
                'flow': flow_state(model.flow), 'step': 0, 'samples_seen': 0, 'train_seconds': 0.,
                'codec_identity': codec_identity}, a.out)
    sha = hashlib.sha256(a.out.read_bytes()).hexdigest()
    print(json.dumps({'out': str(a.out), 'sha256': sha, 'flow_parameters': params}))


if __name__ == '__main__':
    main()

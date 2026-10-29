#!/usr/bin/env python3
"""Score one stable_flow_ddp.py checkpoint on a fixed cohort with the trainer's own evaluate().

Reports training-set memory (pilot_train1024[:32]) next to heldout generation (pilot32, confirmation64)
for the same weights. Read-only on the run directory; writes only under --out.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joint_flow as jf  # noqa: E402  (also puts the code_v4 base on sys.path)
from codec_research import CacheBank, digest  # noqa: E402
from codec_flow_research import evaluate, load_flow_state  # noqa: E402
from stable_flow import StableObjectFlow  # noqa: E402

FORMAT = 'claude_stable_flow_ddp_v1'
SPLIT = {'pilot_train1024': 'train', 'pilot32': 'validation', 'confirmation64': 'validation'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['data', 'cache', 'codec-checkpoint', 'cohorts', 'checkpoint', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--codec-sha', required=True)
    p.add_argument('--which', choices=['ema', 'raw'], required=True)
    p.add_argument('--cohort', choices=sorted(SPLIT), required=True)
    p.add_argument('--count', type=int, default=32)
    p.add_argument('--eval-size', type=int, default=64)
    p.add_argument('--deadline-unix', type=float, required=True)
    a = p.parse_args()
    device = torch.device('cuda')
    bank = CacheBank(a.data, a.cache)
    document = json.loads(a.cohorts.read_text())
    if document['manifest_sha256'] != bank.manifest_sha:
        raise ValueError('Cohort and source manifest hashes differ')
    ids = []
    for r in document[a.cohort][:a.count]:
        source = bank.records[r['id']]
        if source['split'] != SPLIT[a.cohort] or source['array_sha256'] != r['sha256']:
            raise ValueError('Cohort record differs from source: ' + r['id'])
        ids.append(r['id'])
    if len(ids) != a.count:
        raise ValueError('Cohort shorter than --count')
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if ckpt.get('format') != FORMAT:
        raise ValueError('Not a stable_flow_ddp.py checkpoint')
    fixed = ckpt['protocol']
    build_args = argparse.Namespace(codec_checkpoint=a.codec_checkpoint, codec_sha=a.codec_sha, seed=fixed['seed'],
                                    small_width=fixed['small_architecture'][0], small_depth=fixed['small_architecture'][1],
                                    small_heads=fixed['small_architecture'][2], mode='frozen')
    model, _ = jf.build(build_args, bank, device)
    codec = model.codec
    flow = StableObjectFlow(codec.objects, codec.latent_dim, codec.config['feature_width'], codec.decoder.anchors,
                            *fixed['small_architecture']).to(device)
    if a.which == 'ema':
        flow.load_state_dict(ckpt['ema_flow'], strict=True)
    else:
        load_flow_state(flow, ckpt['flow'])
    flow.requires_grad_(False)
    result = evaluate(jf.JointFlow(codec, flow, False), bank, ids, [], a.out, a.eval_size, int(ckpt['step']),
                      device, True, a.deadline_unix)
    meta = {'checkpoint': str(a.checkpoint), 'checkpoint_sha256': digest(a.checkpoint), 'which': a.which,
            'cohort': a.cohort, 'split': SPLIT[a.cohort], 'count': a.count, 'step': int(ckpt['step']),
            'train_gpu_hours': float(ckpt['train_seconds']) / 3600,
            'cohort_selection': document['selection'], 'elapsed_seconds': result['elapsed_seconds']}
    (a.out / 'memory_eval_meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps({**meta, 's16': result['means']['sample_16_seed_average']['mse'],
                      's4': result['means']['sample_4_seed_average']['mse']}))


if __name__ == '__main__':
    main()

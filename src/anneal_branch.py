#!/usr/bin/env python3
"""Turn a stable_flow_ddp.py checkpoint into the resume point of a short annealing branch.

The branch keeps the source's flow, EMA, AdamW moments and per-rank data/noise RNG, so after the branch
point it draws exactly the batches the source run drew; only the LR schedule differs: cosine from the
source's LR at the branch point to --lr-min over --anneal-hours training GPU-hours. Writes
<out>/checkpoints/latest.pt and <out>/protocol.json; `stable_flow_ddp.py --resume` with the matching
arguments (printed as JSON) then trains it unchanged. Budget counters restart at zero for the branch.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stable_flow_ddp as sfd  # noqa: E402
from codec_research import digest, canonical_sha, atomic_json  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--source-sha', required=True)
    p.add_argument('--cohorts', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--anneal-hours', type=float, required=True)
    p.add_argument('--lr-min', type=float, default=1e-5)
    a = p.parse_args()
    if digest(a.source) != a.source_sha:
        raise ValueError('Source checkpoint SHA mismatch')
    folder = a.out / 'checkpoints'
    if folder.exists() and any(folder.iterdir()):
        raise SystemExit('Branch checkpoint directory is not empty')
    ckpt = torch.load(a.source, map_location='cpu', weights_only=False)
    if ckpt.get('format') != sfd.FORMAT:
        raise ValueError('Source is not a stable_flow_ddp.py checkpoint')
    old = ckpt['protocol']
    source_signature = ckpt['signature_sha256']
    if digest(a.cohorts) != old['cohorts_sha256']:
        raise ValueError('Cohort file differs from the source run')
    branch_h = ckpt['train_gpu_seconds'] / 3600
    lr = ckpt['optimizer']['param_groups'][0]['lr']
    # Same namespace fields stable_flow_ddp.py reads into protocol_fields(); everything but the schedule is inherited.
    ns = argparse.Namespace(seed=old['seed'], batch_per_rank=old['batch_per_rank'], lr=lr, lr_min=a.lr_min,
                            warmup_steps=old['warmup_steps'], decay_end_hours=round(branch_h + a.anneal_hours, 6),
                            weight_decay=old['weight_decay'], ema_decay=old['ema_decay'],
                            small_width=old['small_architecture'][0], small_depth=old['small_architecture'][1],
                            small_heads=old['small_architecture'][2],
                            diverge_loss=old['divergence_rule']['mean_loss_1000_steps_above'],
                            diverge_grad=old['divergence_rule']['median_grad_500_steps_above'],
                            codec_sha=old['codec_sha256'], eval_cohort=old['eval_cohort'], eval_count=old['eval_count'])
    if ckpt['stable_steps'] < ns.warmup_steps:
        raise ValueError('Branch point is still inside warmup')
    here = Path(sfd.__file__).resolve().parent
    source_sha = {name: digest(here / name) for name in sfd.SOURCE_FILES}
    fixed = sfd.protocol_fields(ns, old['world_size'], source_sha, old['cohorts_sha256'], a.source_sha)
    ckpt.update(protocol=fixed, signature_sha256=canonical_sha(fixed), start_train_gpu_seconds=ckpt['train_gpu_seconds'],
                milestones_done=[], pins_done=[], allocated_gpu_seconds=0., eval_seconds=0., allocated_seconds=0.)
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f'branch.pt.{os.getpid()}.tmp'
    torch.save(ckpt, tmp)
    os.replace(tmp, folder / 'branch.pt')
    os.symlink('branch.pt', folder / 'latest.pt')
    atomic_json(a.out / 'protocol.json', {**fixed, 'signature_sha256': ckpt['signature_sha256'],
                'branch': {'kind': 'anneal', 'source': str(a.source), 'source_sha256': a.source_sha,
                           'source_signature': source_signature,
                           'branch_step': int(ckpt['step']), 'branch_train_gpu_hours': branch_h,
                           'lr_at_branch': lr, 'anneal_hours': a.anneal_hours,
                           'inherited': 'flow, ema_flow, AdamW state, per-rank data/noise RNG'}})
    print(json.dumps({'lr': lr, 'lr_min': a.lr_min, 'decay_end_hours': ns.decay_end_hours,
                      'target_h': ns.decay_end_hours, 'branch_h': branch_h, 'step': int(ckpt['step']),
                      'signature': ckpt['signature_sha256']}))


if __name__ == '__main__':
    main()

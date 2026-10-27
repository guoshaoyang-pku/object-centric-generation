"""Diagnostic only: optimize each heldout clip's latent using its TRUE future.

This measures an amortization/latent-search gap for a frozen trained decoder.
It is neither video prediction nor a decoder capacity bound. Refined latents
can leave the learned posterior distribution, so improvement does not establish
that a larger encoder alone would recover the same improvement.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from codec_research import CacheBank, atomic_json, sampled_queries, sync, SHOWCASE, digest
from codec_flow_research import load_codec, state_sha
from diagnose_persistence import save_gif, time_metrics


def run(a):
    if a.out.exists() and any(a.out.iterdir()):
        raise ValueError('Use a fresh output directory')
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    bank = CacheBank(a.data, a.cache)
    codec, identity = load_codec(a.checkpoint, bank)
    codec = codec.to(device).eval()
    before_sha = state_sha(codec)
    ids = SHOWCASE
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        _, mean, _, _ = codec.encode_future(*bank.batch(ids, device))
    initial = mean.float().detach()
    latent = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([latent], lr=a.lr)
    query_rng = torch.Generator().manual_seed(a.seed)
    histories = []
    eval_seconds = train_seconds = 0.
    started = time.perf_counter()
    deadline = min(a.deadline_unix, time.time() + a.max_hours * 3600)
    initial_video = None

    @torch.no_grad()
    def evaluate(step):
        nonlocal initial_video, eval_seconds
        began = time.perf_counter()
        tt, yy, xx = torch.meshgrid(torch.linspace(-1, 1, 20, device=device),
                                    torch.linspace(-1, 1, a.size, device=device),
                                    torch.linspace(-1, 1, a.size, device=device), indexing='ij')
        queries = torch.stack((xx, yy, tt), -1).reshape(1, -1, 3).expand(len(ids), -1, -1)
        params = codec.decoder.decode(latent)
        chunks = []
        for query in queries.split(256, dim=1):
            if time.time() >= deadline:
                eval_seconds += time.perf_counter() - began
                raise TimeoutError('Probe deadline during dense evaluation')
            chunks.append(codec.decoder.render(params, query))
        prediction = torch.cat(chunks, 1).reshape(len(ids), 20, a.size, a.size, 3)
        prediction = prediction.permute(0, 1, 4, 2, 3).cpu().numpy()
        if initial_video is None:
            initial_video = prediction.copy()
        records, rows = [], []
        for i, identifier in enumerate(ids):
            full = torch.from_numpy(np.array(bank.rgb(identifier), copy=True)).permute(0, 3, 1, 2).float() / 255
            full = F.interpolate(full, (a.size, a.size), mode='bilinear', align_corners=False).numpy()
            records.append({'id': identifier, 'metrics': time_metrics(prediction[i], full[4:], full[3])})
            rows.append((identifier, [full[4:].transpose(0, 2, 3, 1),
                                      initial_video[i].transpose(0, 2, 3, 1),
                                      prediction[i].transpose(0, 2, 3, 1)]))
        directory = a.out / f'step_{step:05d}'
        directory.mkdir()
        save_gif(directory / 'comparison.gif', rows,
                 ['Ground truth', 'Encoded AE / FUTURE SEEN', 'Optimized z / FUTURE SEEN'], 12)
        summary = {'step': step, 'records': records,
                   'mse': float(np.mean([r['metrics']['mse'] for r in records])),
                   'latent_rms_change': float((latent - initial).square().mean().sqrt()),
                   'latent_rms': float(latent.square().mean().sqrt()),
                   'train_gpu_hours': train_seconds / 3600, 'future_seen': True}
        atomic_json(directory / 'summary.json', summary)
        histories.append(summary)
        sync(device)
        eval_seconds += time.perf_counter() - began

    step = 0
    reason = 'step_limit'
    try:
        evaluate(0)
        while step < a.steps:
            if time.time() >= deadline:
                raise TimeoutError('Probe deadline before update')
            sync(device)
            began = time.perf_counter()
            coords, target = sampled_queries(bank, ids, a.points, query_rng, device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(codec.render_queries(latent, coords), target)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite latent search')
            loss.backward()
            torch.nn.utils.clip_grad_norm_([latent], 1., error_if_nonfinite=True)
            optimizer.step()
            sync(device)
            train_seconds += time.perf_counter() - began
            step += 1
            if step in {100, 500, a.steps}:
                evaluate(step)
                print(json.dumps({'step': step, 'mse': histories[-1]['mse'],
                                  'train_gpu_hours': train_seconds / 3600}), flush=True)
    except TimeoutError:
        reason = 'deadline'
    after_sha = state_sha(codec)
    if before_sha != after_sha or any(p.grad is not None for p in codec.parameters()):
        raise RuntimeError('Frozen codec changed')
    torch.save({'ids': ids, 'initial': initial.cpu(), 'refined': latent.detach().cpu(),
                'codec_identity': identity}, a.out / 'latents.pt')
    atomic_json(a.out / 'summary.json', {
        'status': 'complete', 'stop_reason': reason,
        'last_evaluated_step': histories[-1]['step'] if histories else None,
        'probe_source_sha256': digest(Path(__file__)),
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        'codec_identity': identity, 'ids': ids,
        'method': 'Frozen codec, independent latent per clip, supervised by true future RGB',
        'claim_limit': __doc__, 'metric_size': a.size, 'lr': a.lr, 'points': a.points,
        'steps': step, 'seed': a.seed, 'train_gpu_hours': train_seconds / 3600,
        'eval_gpu_hours': eval_seconds / 3600, 'wall_seconds': time.perf_counter() - started,
        'codec_unchanged': before_sha == after_sha, 'codec_state_sha256': before_sha,
        'evaluations': histories})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint', 'data', 'cache', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=.03)
    parser.add_argument('--points', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=764)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--max-hours', type=float, default=.15)
    parser.add_argument('--deadline-unix', type=float, required=True)
    run(parser.parse_args())

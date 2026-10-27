"""Future-seen oracle diagnostic with the posterior-mean normalization constraint.

The posterior mean is per-object layer-normalized, but q(z|video) also adds
learned diagonal Gaussian noise. Only the mean's realizable set is constrained
here; this is NOT a claim that the full posterior has bounded support.
Neither optimized latent reconstruction nor its improvement proves forecasting,
semantic identity, physical compliance, or recoverability by a larger encoder.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from codec_research import CacheBank, SHOWCASE, atomic_json, digest, sampled_queries, sync
from codec_flow_research import load_codec, state_sha
from diagnose_persistence import save_gif, time_metrics


def normalized_mean(raw):
    return F.layer_norm(raw.float(), (raw.shape[-1],), eps=1e-5)


def latent_stats(z, initial=None, require_normalized=False):
    value = z.detach().float()
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError('Nonfinite latent')
    means, rms = value.mean(-1), value.square().mean(-1).sqrt()
    result = {'global_rms': float(value.square().mean().sqrt()),
              'max_abs_object_mean': float(means.abs().max()),
              'object_rms_min': float(rms.min()), 'object_rms_max': float(rms.max()),
              'max_object_rms_deviation_from_one': float((rms - 1).abs().max())}
    if initial is not None:
        result['rms_change_from_encoded'] = float((value - initial).square().mean().sqrt())
    if require_normalized and (result['max_abs_object_mean'] > 2e-5 or
                               result['max_object_rms_deviation_from_one'] > 2e-4):
        raise AssertionError('Per-object mean/RMS constraint violated')
    return result


def verified_load(path, expected_sha, binary=False):
    before = digest(path)
    if before != expected_sha:
        raise ValueError('Input SHA mismatch: ' + str(path))
    value = torch.load(path, map_location='cpu', weights_only=False) if binary else json.loads(path.read_text())
    if digest(path) != before:
        raise ValueError('Input changed while loading: ' + str(path))
    return value


def self_test():
    torch.manual_seed(764)
    u = torch.randn(2, 5, 128, requires_grad=True)
    z = normalized_mean(u)
    latent_stats(z, require_normalized=True)
    torch.testing.assert_close(z, F.layer_norm(u, (128,)), rtol=0, atol=0)
    target = normalized_mean(torch.randn_like(u))
    optimizer = torch.optim.Adam([u], lr=.03)
    before = float((z - target).square().mean().detach())
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        loss = (normalized_mean(u) - target).square().mean()
        loss.backward()
        assert u.grad is not None and bool(torch.isfinite(u.grad).all())
        optimizer.step()
        latent_stats(normalized_mean(u), require_normalized=True)
    assert float((normalized_mean(u) - target).square().mean().detach()) < before
    print('CPU normalization/gradient/optimization checks passed; no CUDA calls')


def run(a):
    started = time.perf_counter()
    deadline = min(a.deadline_unix, time.time() + a.max_hours * 3600)
    if time.time() >= deadline:
        raise TimeoutError('Deadline already passed')
    if a.out.exists() and any(a.out.iterdir()):
        raise ValueError('Use a fresh output directory')
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device(a.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    previous = verified_load(a.source_summary, a.source_summary_sha)
    stored = verified_load(a.latents, a.latents_sha, binary=True)
    bank = CacheBank(a.data, a.cache)
    codec, identity = load_codec(a.checkpoint, bank, a.checkpoint_sha)
    if stored.get('codec_identity') != identity or previous.get('codec_identity') != identity:
        raise ValueError('Stored latents and summary must bind the identical frozen codec checkpoint')
    ids = stored['ids']
    if ids != SHOWCASE or ids != previous['ids'] or previous.get('metric_size') != a.size:
        raise ValueError('Requires the original ordered four cases and native metric size')
    if previous.get('stop_reason') != 'step_limit' or previous.get('last_evaluated_step') != previous.get('steps'):
        raise ValueError('Source refinement lacks a fully evaluated final latent')
    if any(previous.get(k) != getattr(a, k) for k in ['lr', 'points', 'seed']):
        raise ValueError('Match source optimizer LR/query seed/point count for this diagnostic')
    codec = codec.to(device).eval()
    before_sha = state_sha(codec)
    expected_shape = (len(ids), codec.objects, codec.latent_dim)
    if tuple(stored['initial'].shape) != expected_shape or tuple(stored['refined'].shape) != expected_shape:
        raise ValueError('Stored latent shape differs from frozen codec')
    initial = stored['initial'].to(device=device, dtype=torch.float32)
    refined = stored['refined'].to(device=device, dtype=torch.float32)
    latent_stats(initial, require_normalized=True)
    latent_stats(refined)
    projected = normalized_mean(refined).detach()
    latent_stats(projected, require_normalized=True)
    raw = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([raw], lr=a.lr)
    query_rng = torch.Generator().manual_seed(a.seed)
    train_seconds = eval_seconds = 0.
    histories, baseline = [], {}
    step, reason = 0, 'step_limit'
    setup_seconds = time.perf_counter() - started
    truth, previous_frame = [], []
    for identifier in ids:
        video = torch.from_numpy(np.array(bank.rgb(identifier), copy=True)).permute(0, 3, 1, 2).float() / 255
        if a.size != 128:
            video = F.interpolate(video, (a.size, a.size), mode='bilinear', align_corners=False)
        truth.append(video[4:].numpy()); previous_frame.append(video[3].numpy())

    def check_deadline():
        if time.time() >= deadline:
            raise TimeoutError('Constraint probe deadline')

    @torch.no_grad()
    def render(z):
        check_deadline()
        tt, yy, xx = torch.meshgrid(torch.linspace(-1, 1, 20, device=device),
                                    torch.linspace(-1, 1, a.size, device=device),
                                    torch.linspace(-1, 1, a.size, device=device), indexing='ij')
        coords = torch.stack((xx, yy, tt), -1).reshape(1, -1, 3).expand(len(ids), -1, -1)
        p = codec.decoder.decode(z.float())
        chunks = []
        for q in coords.split(256, dim=1):
            check_deadline()
            chunks.append(codec.decoder.render(p, q))
        result = torch.cat(chunks, 1).reshape(len(ids), 20, a.size, a.size, 3)
        return result.permute(0, 1, 4, 2, 3).cpu().numpy()

    @torch.no_grad()
    def evaluate(z, label, constrained=False):
        nonlocal eval_seconds
        began = time.perf_counter()
        try:
            prediction = render(z)
            records = [{'id': identifier, 'metrics': time_metrics(prediction[i], truth[i], previous_frame[i])}
                       for i, identifier in enumerate(ids)]
            summary = {'label': label, 'step': step, 'records': records,
                       'mse': float(np.mean([r['metrics']['mse'] for r in records])),
                       'latent': latent_stats(z, initial, require_normalized=constrained),
                       'train_gpu_hours': train_seconds / 3600 if device.type == 'cuda' else 0.,
                       'future_seen': True, 'posterior_mean_constraint': constrained}
            atomic_json(a.out / (label + '.json'), summary)
            return prediction, summary
        finally:
            sync(device)
            eval_seconds += time.perf_counter() - began

    def comparison(path, panels, labels):
        nonlocal eval_seconds
        began = time.perf_counter()
        rows = [(identifier, [truth[i].transpose(0, 2, 3, 1)] +
                 [p[i].transpose(0, 2, 3, 1) for p in panels]) for i, identifier in enumerate(ids)]
        save_gif(path, rows, ['Ground truth'] + labels, 12)
        eval_seconds += time.perf_counter() - began

    try:
        encoded_video, baseline['encoded'] = evaluate(initial, 'encoded', True)
        refined_video, baseline['unconstrained'] = evaluate(refined, 'unconstrained')
        projected_video, baseline['projected_unconstrained'] = evaluate(projected, 'projected_unconstrained', True)
        for label, index in [('encoded', 0), ('unconstrained', -1)]:
            if abs(baseline[label]['mse'] - previous['evaluations'][index]['mse']) > 1e-6:
                raise ValueError('Rerender disagrees with original evidence: ' + label)
        comparison(a.out / 'baseline_comparison.gif', [encoded_video, refined_video, projected_video],
                   ['Encoded / FUTURE SEEN', 'Free z / FUTURE SEEN', 'LN(free z) / FUTURE SEEN'])
        pred, item = evaluate(normalized_mean(raw), 'constrained_00000', True)
        histories.append(item)
        while step < a.steps:
            check_deadline()
            sync(device)
            began = time.perf_counter()
            coords, target = sampled_queries(bank, ids, a.points, query_rng, device)
            optimizer.zero_grad(set_to_none=True)
            z = normalized_mean(raw)
            loss = F.mse_loss(codec.render_queries(z, coords), target)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite normalized latent search')
            loss.backward()
            torch.nn.utils.clip_grad_norm_([raw], 1., error_if_nonfinite=True)
            optimizer.step()
            sync(device)
            train_seconds += time.perf_counter() - began
            step += 1
            if step in {100, 500, a.steps}:
                pred, item = evaluate(normalized_mean(raw), f'constrained_{step:05d}', True)
                histories.append(item)
                comparison(a.out / f'constrained_{step:05d}.gif',
                           [encoded_video, refined_video, projected_video, pred],
                           ['Encoded / FUTURE SEEN', 'Free z / FUTURE SEEN',
                            'LN(free z) / FUTURE SEEN', 'LN search / FUTURE SEEN'])
                print(json.dumps({'step': step, 'mse': item['mse'],
                                  'train_gpu_hours': item['train_gpu_hours']}), flush=True)
    except TimeoutError:
        reason = 'deadline'
    final = normalized_mean(raw).detach()
    final_stats = latent_stats(final, initial, require_normalized=True)
    after_sha = state_sha(codec)
    if before_sha != after_sha or any(p.requires_grad or p.grad is not None for p in codec.parameters()):
        raise RuntimeError('Frozen codec changed or received gradients')
    target = a.out / 'latents.pt'
    tmp = a.out / 'latents.tmp'
    torch.save({'ids': ids, 'initial': initial.cpu(), 'unconstrained': refined.cpu(),
                'projected_unconstrained': projected.cpu(), 'constrained_raw': raw.detach().cpu(),
                'constrained': final.cpu(), 'codec_identity': identity, 'completed_steps': step}, tmp)
    os.replace(tmp, target)
    atomic_json(a.out / 'summary.json', {
        'status': 'complete' if reason == 'step_limit' else 'partial', 'stop_reason': reason,
        'completed_steps': step, 'last_evaluated_step': histories[-1]['step'] if histories else None,
        'probe_source_sha256': digest(Path(__file__)), 'codec_identity': identity,
        'inputs_sha256': {'checkpoint': a.checkpoint_sha, 'latents': a.latents_sha, 'source_summary': a.source_summary_sha},
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        'ids': ids, 'metric_size': a.size, 'constraint': 'z=F.layer_norm(u, [128], eps=1e-5), no affine; per-object mean zero and RMS approximately one',
        'constraint_scope': 'Posterior mean only; learned Gaussian noise gives q a different support.',
        'objective': 'True-future sampled-pixel RGB MSE, same LR/query seed/points as the unconstrained reference',
        'claim_limit': __doc__, 'baseline': baseline, 'evaluations': histories,
        'final_latent_statistics': final_stats, 'codec_unchanged': before_sha == after_sha,
        'codec_state_sha256': before_sha, 'train_gpu_hours': train_seconds / 3600 if device.type == 'cuda' else 0.,
        'eval_gpu_hours': eval_seconds / 3600 if device.type == 'cuda' else 0.,
        'setup_seconds': setup_seconds, 'wall_seconds': time.perf_counter() - started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        'saved_latents_sha256': digest(target)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--self-test', action='store_true')
    for name in ['checkpoint', 'data', 'cache', 'out', 'latents', 'source-summary']:
        parser.add_argument('--' + name, type=Path)
    for name in ['checkpoint-sha', 'latents-sha', 'source-summary-sha']:
        parser.add_argument('--' + name)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=.03)
    parser.add_argument('--points', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=764)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-hours', type=float, default=1 / 30)
    parser.add_argument('--deadline-unix', type=float)
    known, _ = parser.parse_known_args()
    if known.config:
        config = json.loads(known.config.read_text())
        admitted = {a.dest for a in parser._actions}
        if set(config) - admitted:
            parser.error('Unknown config fields: ' + ', '.join(sorted(set(config) - admitted)))
        parser.set_defaults(**config)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    required = ['checkpoint', 'data', 'cache', 'out', 'latents', 'source_summary',
                'checkpoint_sha', 'latents_sha', 'source_summary_sha', 'deadline_unix']
    if any(getattr(args, key) is None for key in required):
        parser.error('Provide input paths, their SHA256 values, and an absolute deadline')
    if min(args.steps, args.lr, args.points, args.size, args.max_hours) <= 0:
        parser.error('Numeric limits must be positive')
    run(args)


if __name__ == '__main__':
    main()

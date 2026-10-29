#!/usr/bin/env python3
"""Latent autoregressive long-video rollout (plan D1): 4 observed frames, then K chunks of 20.

Chunk 0 conditions on the cached 4-frame prefix and uses the evaluation noise,
so it is exactly the evaluated sample. Every later chunk re-encodes the last 4
generated frames with the frozen Wan2.2 VAE, with the padding, normalization
and token coordinates of cache_vae_features.py, and samples the next 20 frames
from fresh fixed-seed noise. No future frame is ever encoded. MOVi-E clips have
24 frames, so ground truth covers only the first chunk; later chunks are judged
by self-consistency: chunk-boundary jumps, colour drift and loss of edge
contrast relative to the last observed frame.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

CODE_BASE = Path(os.environ.get('PCV_CODE_BASE', Path(__file__).resolve().parent / 'code_base_v4'))
sys.path.insert(0, str(CODE_BASE))
from codec_research import CacheBank, digest, atomic_json  # noqa: E402
from codec_flow_research import load_codec, SmallObjectFlow, ConditionalObjectFlow, load_flow_state, prefix_batch  # noqa: E402
from eval_validation import clip_noise  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stable_flow import StableObjectFlow  # noqa: E402

SIZE = 128


def vae_encode(vae, frames, mean, std):
    """frames [B,T,3,H,W] in [0,1]; same as cache_vae_features.encode (posterior mode, FP32)."""
    x = frames.permute(0, 2, 1, 3, 4).float() * 2 - 1
    factor = int(getattr(vae.config, 'scale_factor_temporal', 4))
    pad = (factor - (x.shape[2] - 1) % factor) % factor
    if pad:
        x = torch.cat([x, x[:, :, -1:].expand(-1, -1, pad, -1, -1)], dim=2)
    with torch.autocast(device_type=x.device.type, enabled=False):
        latent = vae.encode(x, return_dict=True).latent_dist.mode()
        return (latent - mean) / std.clamp_min(1e-6)


def edge_strength(frame):
    """Mean absolute horizontal plus vertical difference of a [3,H,W] frame."""
    return float((frame[:, :, 1:] - frame[:, :, :-1]).abs().mean() + (frame[:, 1:] - frame[:, :-1]).abs().mean())


def load_model(spec, base_codec, device):
    label, path, key, *codec_key = spec.split('=')
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    codec = base_codec
    if codec_key:
        codec = copy.deepcopy(base_codec).cpu()
        codec.load_state_dict(ckpt[codec_key[0]])
        codec = codec.to(device).requires_grad_(False).eval()
    state = ckpt[key]
    cls = StableObjectFlow if any('.q_self.' in k for k in state) else SmallObjectFlow
    flow = cls(codec.objects, codec.latent_dim, codec.config['feature_width'], codec.decoder.anchors, 384, 8, 6)
    if cls is StableObjectFlow and key == 'ema_flow':
        flow.load_state_dict(state, strict=True)
    else:
        load_flow_state(flow, state)
    model = ConditionalObjectFlow(codec, flow).to(device).eval()
    identity = {'label': label, 'path': path, 'sha256': digest(path), 'state_key': key, 'flow_class': cls.__name__,
                'codec': codec_key[0] if codec_key else 'C2 frozen', 'step': ckpt.get('step')}
    return model, identity


def labelled(frame_rgb, text):
    img = Image.fromarray(frame_rgb)
    strip = Image.new('RGB', (img.width, 13), (250, 250, 250))
    ImageDraw.Draw(strip).text((2, 0), text, fill=(20, 20, 20))
    out = Image.new('RGB', (img.width, img.height + 13))
    out.paste(strip, (0, 0))
    out.paste(img, (0, 13))
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['data', 'cache', 'codec-checkpoint', 'pretrained']:
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--codec-sha', required=True)
    ap.add_argument('--cohorts', type=Path, required=True)
    ap.add_argument('--flow', required=True, help='label=checkpoint=state_key[=codec_state]')
    ap.add_argument('--chunks', type=int, default=4)
    ap.add_argument('--seeds', default='17,29')
    ap.add_argument('--pilot', type=int, default=4, help='First N pilot32 clips, after showcase4')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda')
    started = time.time()
    bank = CacheBank(a.data, a.cache)
    base_codec, codec_identity = load_codec(a.codec_checkpoint, bank, a.codec_sha)
    base_codec = base_codec.to(device)
    model, identity = load_model(a.flow, base_codec, device)
    codec = model.codec
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(a.pretrained, subfolder='vae', torch_dtype=torch.float32,
                                           local_files_only=True).to(device).float().eval().requires_grad_(False)
    ch = int(vae.config.z_dim)
    mean = torch.tensor(vae.config.latents_mean, device=device, dtype=torch.float32).reshape(1, ch, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device, dtype=torch.float32).reshape(1, ch, 1, 1, 1)
    cohorts = json.loads(a.cohorts.read_text())
    clips = [r['id'] for r in cohorts['showcase4']] + [r['id'] for r in cohorts['pilot32']][:a.pilot]
    seeds = [int(s) for s in a.seeds.split(',')]
    records = []
    with torch.inference_mode():
        for identifier in clips:
            cached_prefix, prefix_coords = prefix_batch(bank, [identifier], device)
            # Sanity check of the re-encoding path: encoding the real prefix must reproduce the cache.
            real = torch.from_numpy(np.array(bank.rgb(identifier), copy=True)).permute(0, 3, 1, 2).float().to(device) / 255
            check = vae_encode(vae, real[None, :4], mean, std).flatten(2).transpose(1, 2)
            reencode_error = float((check - cached_prefix.float()).abs().max())
            truth = real[4:].cpu()
            rollouts = {}
            for seed in seeds:
                frames = [real[:4].cpu()]
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    context = model.prefix_context(cached_prefix, prefix_coords)
                chunk_stats = []
                for k in range(a.chunks):
                    noise = clip_noise([identifier], seed + 1000 * k, codec.objects, codec.latent_dim, device, torch.float32)
                    z = model.sample(context, noise, 16)
                    video = codec.render_latent(z, SIZE)[0].clamp(0, 1)  # [20,3,H,W]
                    frames.append(video.cpu())
                    prev_last = frames[-2][-1]
                    within = (video[1:] - video[:-1]).abs().mean().item()
                    chunk_stats.append({'chunk': k, 'noise_seed': seed + 1000 * k,
                                        'boundary_jump_l1': float((video[0].cpu() - prev_last).abs().mean()),
                                        'within_chunk_step_l1': within,
                                        'mse_vs_truth': float((video.cpu() - truth).square().mean()) if k == 0 else None})
                    last4 = video[-4:][None]
                    tokens = vae_encode(vae, last4, mean, std).flatten(2).transpose(1, 2)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        context = model.prefix_context(tokens, prefix_coords)
                seq = torch.cat(frames)  # [4+20K,3,H,W]
                anchor = seq[3]
                rollouts[seed] = seq
                per_frame = [{'t': t, 'mean_rgb_drift': float((seq[t].mean((1, 2)) - anchor.mean((1, 2))).abs().mean()),
                              'edge_ratio': edge_strength(seq[t]) / max(edge_strength(anchor), 1e-8)}
                             for t in range(seq.shape[0])]
                records.append({'id': identifier, 'seed': seed, 'chunks': chunk_stats, 'per_frame': per_frame,
                                'prefix_reencode_max_abs_error': reencode_error})
            # GIF: GT (24 frames, then blank) | one panel per seed.
            total = 4 + 20 * a.chunks
            gif = []
            for t in range(total):
                k = (t - 4) // 20 if t >= 4 else -1
                tag = 'observed' if t < 4 else f'chunk {k}'
                gt = (real[t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) if t < 24 else \
                    np.full((SIZE, SIZE, 3), 40, np.uint8)
                panels = [labelled(gt, f'GT t={t}' if t < 24 else 'no GT')]
                for seed in seeds:
                    f = (rollouts[seed][t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    panels.append(labelled(f, f's{seed} {tag}'))
                gif.append(Image.fromarray(np.concatenate(panels, 1)).convert('P', palette=Image.ADAPTIVE, colors=255))
            gif[0].save(a.out / f'rollout_{identifier}.gif', save_all=True, append_images=gif[1:],
                        duration=1000 // 12, loop=0, optimize=True)
            # Strip: last observed frame and the last frame of each chunk, per seed.
            picks = [3] + [3 + 20 * (k + 1) for k in range(a.chunks)]
            rows = [np.concatenate([labelled((rollouts[s][t].permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                                             f's{s} t={t}') for t in picks], 1) for s in seeds]
            Image.fromarray(np.concatenate(rows, 0)).save(a.out / f'strip_{identifier}.png', optimize=True)
            atomic_json(a.out / 'progress.json', {'done': clips.index(identifier) + 1, 'total': len(clips)})
    summary = {'format': 'claude_long_rollout_v1', 'flow': identity, 'codec': codec_identity, 'chunks': a.chunks,
               'frames_total': 4 + 20 * a.chunks, 'seeds': seeds, 'clips': clips, 'size': SIZE,
               'mean_over_clips': {
                   f'chunk_{k}': {
                       'boundary_jump_l1': float(np.mean([r['chunks'][k]['boundary_jump_l1'] for r in records])),
                       'within_chunk_step_l1': float(np.mean([r['chunks'][k]['within_chunk_step_l1'] for r in records])),
                       'end_mean_rgb_drift': float(np.mean([r['per_frame'][3 + 20 * (k + 1)]['mean_rgb_drift'] for r in records])),
                       'end_edge_ratio': float(np.mean([r['per_frame'][3 + 20 * (k + 1)]['edge_ratio'] for r in records]))}
                   for k in range(a.chunks)},
               'chunk0_mse_vs_truth': float(np.mean([r['chunks'][0]['mse_vs_truth'] for r in records])),
               'max_prefix_reencode_error': max(r['prefix_reencode_max_abs_error'] for r in records),
               'records': records, 'elapsed_seconds': time.time() - started,
               'scope': 'Only chunk 0 has ground truth. Later chunks condition on re-encoded generated frames; '
                        'their metrics are self-consistency only. Fixed clips and seeds; no selection.'}
    atomic_json(a.out / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ['mean_over_clips', 'chunk0_mse_vs_truth', 'max_prefix_reencode_error',
                                              'elapsed_seconds']}, indent=1))


if __name__ == '__main__':
    main()

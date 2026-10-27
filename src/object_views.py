#!/usr/bin/env python3
"""Object-centric views of frozen-codec flow checkpoints.

For fixed clips (showcase4 plus pilot32, cohort order, no selection) this
renders the posterior-mean AE (future seen) and prefix-only samples (16 steps,
seeds 17 and 29, the same noise as the trainers' evaluation). For every
future pixel it records which parent owns it, i.e. the argmax of the
renderer's normalized mixing weights, and compares that ownership with the
official MOVi instance segmentation. A fixed nearest-anchor partition is the
reference for "parents do not follow objects". Read-only on checkpoints.
"""
import argparse
import colorsys
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
from codec_flow_research import (load_codec, SmallObjectFlow, ConditionalObjectFlow,  # noqa: E402
                                 load_flow_state, prefix_batch)
from eval_validation import clip_noise  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stable_flow import StableObjectFlow  # noqa: E402

SIZE, FRAMES, MIN_PIXELS = 128, 20, 20


def palette(n, seed=0):
    order = np.random.default_rng(seed).permutation(n)
    rows = [colorsys.hsv_to_rgb((k * .618034) % 1, .55 + .4 * (k % 2), .98 - .3 * ((k // 2) % 2)) for k in order]
    return (np.array(rows) * 255).astype(np.uint8)


def instance_colors(is_dynamic):
    colors = np.zeros((len(is_dynamic) + 1, 3), np.uint8)
    colors[0] = (35, 35, 35)
    for i, dyn in enumerate(is_dynamic):
        hue = (i * .37) % 1
        colors[i + 1] = np.array(colorsys.hsv_to_rgb(hue, .95, 1.) if dyn else
                                 colorsys.hsv_to_rgb(hue, .25, .55)) * 255
    return colors


def mixing_weights(decoder, p, coords):
    """The renderer's per-parent weights; mirrors TrajectoryDecoder.render."""
    local = decoder.local_coordinates(p, coords)
    delta = local[..., None, :] - p['child_center'][:, None]
    angle = p['child_angle'][:, None]
    c, s = angle.cos(), angle.sin()
    dx, dy = delta.unbind(-1)
    child = torch.stack((c * dx + s * dy, -s * dx + c * dy), -1)
    distance = (child / p['child_scale'][:, None]).square().sum(-1)
    log_density = -.5 * distance + p['child_log_amplitude'][:, None]
    return torch.logsumexp(log_density, -1).softmax(-1)


@torch.no_grad()
def views(codec, z):
    """RGB [20,S,S,3], owner ids [20,S,S], parent centers [20,N,2] in pixels."""
    decoder = codec.decoder
    p = decoder.decode(z.float())
    lin = torch.linspace(-1, 1, SIZE, device=z.device)
    yy, xx = torch.meshgrid(lin, lin, indexing='ij')
    rgb, owner = [], []
    times = torch.linspace(-1, 1, FRAMES, device=z.device)
    for t in times:
        q = torch.stack((xx, yy, torch.full_like(xx, t)), -1).reshape(1, -1, 3)
        rgb.append(decoder.render(p, q).reshape(SIZE, SIZE, 3))
        owner.append(mixing_weights(decoder, p, q).argmax(-1).reshape(SIZE, SIZE))
    centers, _ = decoder.motion_at(p, times[None])
    centers = (centers[0] + 1) * (SIZE - 1) / 2
    return (torch.stack(rgb).clamp(0, 1).cpu().numpy(), torch.stack(owner).cpu().numpy().astype(np.int16),
            centers.cpu().numpy())


def nearest_anchor_owner(anchors):
    lin = torch.linspace(-1, 1, SIZE)
    yy, xx = torch.meshgrid(lin, lin, indexing='ij')
    grid = torch.stack((xx, yy), -1).reshape(-1, 2)
    owner = torch.cdist(grid, anchors.cpu().float()).argmin(-1).reshape(SIZE, SIZE).numpy().astype(np.int16)
    return np.repeat(owner[None], FRAMES, 0)


def ownership_stats(owner, seg, is_dynamic, parents):
    """Per instance: parents needed for 80% of its pixels, and overlap of its parent set with its first visible frame."""
    groups = {'dynamic': [], 'static': [], 'background': []}
    for label in range(len(is_dynamic) + 1):
        hists = []
        for t in range(FRAMES):
            mask = seg[t] == label
            count = int(mask.sum())
            if count < MIN_PIXELS:
                continue
            hists.append(np.bincount(owner[t][mask], minlength=parents) / count)
        if len(hists) < 2:
            continue
        frag = [int(np.searchsorted(np.cumsum(np.sort(h)[::-1]), .8) + 1) for h in hists]
        overlap = [float(np.minimum(h, hists[0]).sum()) for h in hists[1:]]
        key = 'background' if label == 0 else ('dynamic' if is_dynamic[label - 1] else 'static')
        groups[key].append({'label': label, 'parents_for_80pct': float(np.mean(frag)),
                            'overlap_with_first_frame': float(np.mean(overlap)),
                            'overlap_last_frame': overlap[-1], 'frames': len(hists)})
    pure = []
    for t in range(FRAMES):
        for parent in np.unique(owner[t]):
            labels = seg[t][owner[t] == parent]
            if labels.size >= 10:
                pure.append((labels.size, np.bincount(labels).max() / labels.size))
    purity = float(sum(n * f for n, f in pure) / max(sum(n for n, _ in pure), 1))
    return {'groups': groups, 'parent_purity': purity,
            'active_parents_per_frame': float(np.mean([len(np.unique(o)) for o in owner]))}


def summarize(per_clip):
    out = {}
    methods = per_clip[0]['methods'].keys()
    for m in methods:
        row = {}
        for g in ['dynamic', 'static', 'background']:
            items = [x for c in per_clip for x in c['methods'][m]['groups'][g]]
            if items:
                row[g] = {k: float(np.mean([x[k] for x in items]))
                          for k in ['parents_for_80pct', 'overlap_with_first_frame', 'overlap_last_frame']}
                row[g]['instances'] = len(items)
        row['parent_purity'] = float(np.mean([c['methods'][m]['parent_purity'] for c in per_clip]))
        row['active_parents_per_frame'] = float(np.mean([c['methods'][m]['active_parents_per_frame'] for c in per_clip]))
        out[m] = row
    return out


def dynamic_tracks(ann, seg, is_dynamic):
    """GT dynamic-object centers in pixels. The (x, y) order of MOVi image_positions
    is checked against visible-mask centroids instead of assumed."""
    positions = ann['image_positions'][:, 4:] * (SIZE - 1)
    errors = {'xy': [], 'yx': []}
    for k in np.flatnonzero(is_dynamic):
        for t in range(FRAMES):
            ys, xs = np.nonzero(seg[t] == k + 1)
            if len(xs) >= MIN_PIXELS:
                errors['xy'].append(np.hypot(*(positions[k, t] - (xs.mean(), ys.mean()))))
                errors['yx'].append(np.hypot(*(positions[k, t, ::-1] - (xs.mean(), ys.mean()))))
    order = min(errors, key=lambda o: np.median(errors[o]) if errors[o] else np.inf)
    tracks = [positions[k] if order == 'xy' else positions[k, :, ::-1] for k in np.flatnonzero(is_dynamic)]
    check = {o: float(np.median(v)) if v else None for o, v in errors.items()}
    check['chosen'] = order
    return tracks, check


def label_strip(width, texts, height=14):
    strip = Image.new('RGB', (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(strip)
    w = width // len(texts)
    for i, text in enumerate(texts):
        draw.text((i * w + 3, 1), text, fill=(20, 20, 20))
    return strip


def save_gif(path, panels, names, scale=1):
    frames = []
    for t in range(FRAMES):
        row = np.concatenate([p[t] for p in panels], 1)
        img = Image.fromarray(row).resize((row.shape[1] * scale, row.shape[0] * scale), Image.NEAREST)
        canvas = Image.new('RGB', (img.width, img.height + 14), (250, 250, 250))
        canvas.paste(label_strip(img.width, [f'{n}  t+{t + 1}' if i == 0 else n for i, n in enumerate(names)]), (0, 0))
        canvas.paste(img, (0, 14))
        frames.append(canvas.convert('P', palette=Image.ADAPTIVE, colors=255))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=1000 // 6, loop=0, optimize=True)


def save_trajectories(path, first_frame, tracks, colors, gt_tracks, names, scale=2):
    """Parent center paths over the 20 future frames; moving parents drawn thick, GT dynamic objects in white."""
    panels = []
    for centers in tracks:
        img = Image.fromarray((first_frame * .45).astype(np.uint8)).resize((SIZE * scale, SIZE * scale), Image.BILINEAR)
        draw = ImageDraw.Draw(img)
        motion = np.linalg.norm(centers[-1] - centers[0], axis=-1)
        for n in np.argsort(motion):
            pts = [tuple(xy * scale) for xy in centers[:, n]]
            moving = motion[n] > 3
            draw.line(pts, fill=tuple(int(c) for c in colors[n]), width=3 if moving else 1)
            if moving:
                draw.ellipse([pts[-1][0] - 3, pts[-1][1] - 3, pts[-1][0] + 3, pts[-1][1] + 3],
                             fill=tuple(int(c) for c in colors[n]))
        for track in gt_tracks:
            pts = [tuple(xy * scale) for xy in track]
            draw.line(pts, fill=(255, 255, 255), width=2)
            draw.ellipse([pts[-1][0] - 4, pts[-1][1] - 4, pts[-1][0] + 4, pts[-1][1] + 4], outline=(255, 255, 255), width=2)
        panels.append(np.asarray(img))
    row = np.concatenate(panels, 1)
    canvas = Image.new('RGB', (row.shape[1], row.shape[0] + 14), (250, 250, 250))
    canvas.paste(label_strip(row.shape[1], names), (0, 0))
    canvas.paste(Image.fromarray(row), (0, 14))
    canvas.save(path, optimize=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--codec-checkpoint', required=True)
    ap.add_argument('--codec-sha', required=True)
    ap.add_argument('--cohorts', type=Path, required=True)
    ap.add_argument('--annotations', type=Path, nargs='+', required=True,
                    help='Directories holding <clip id>/annotations.npz')
    ap.add_argument('--flow', action='append', required=True,
                    help='label=checkpoint_path=state_key[=codec_state], e.g. latest=.../step.pt=ema_flow; '
                         'a fourth field loads that run\'s own trained codec from the same checkpoint')
    ap.add_argument('--gif-flow', required=True, help='Flow label used for the GIF and trajectory views')
    ap.add_argument('--gif-pilot', type=int, default=4, help='First N pilot32 clips also get views')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start = time.time()
    bank = CacheBank(a.data, a.cache)
    codec, codec_identity = load_codec(a.codec_checkpoint, bank, a.codec_sha)
    codec = codec.to(device)
    if codec.decoder.temporal_support:
        raise ValueError('Ownership view assumes no child time envelopes')
    flows, identities = {}, {}
    for spec in a.flow:
        label, path, key, *codec_key = spec.split('=')
        sha = digest(path)
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        own = codec
        if codec_key:
            own = copy.deepcopy(codec).cpu()
            own.load_state_dict(ckpt[codec_key[0]])
            own = own.to(device).requires_grad_(False).eval()
        state = ckpt[key]
        cls = StableObjectFlow if any('.q_self.' in k for k in state) else SmallObjectFlow
        flow = cls(own.objects, own.latent_dim, own.config['feature_width'], own.decoder.anchors, 384, 8, 6)
        if cls is StableObjectFlow and key == 'ema_flow':
            flow.load_state_dict(state, strict=True)
        else:
            load_flow_state(flow, state)
        flows[label] = ConditionalObjectFlow(own, flow).to(device).eval()
        identities[label] = {'path': path, 'sha256': sha, 'state_key': key, 'flow_class': cls.__name__,
                             'codec': codec_key[0] if codec_key else 'C2 frozen', 'step': ckpt.get('step')}
        del ckpt
    cohorts = json.loads(a.cohorts.read_text())
    showcase = [r['id'] for r in cohorts['showcase4']]
    pilot = [r['id'] for r in cohorts['pilot32']]
    clips = list(dict.fromkeys(showcase + pilot))
    view_ids = showcase + pilot[:a.gif_pilot]
    colors = palette(codec.objects)
    reference = nearest_anchor_owner(codec.decoder.anchors)
    per_clip = {'showcase4': [], 'pilot32': []}
    for identifier in clips:
        ann_dir = next(d / identifier for d in a.annotations if (d / identifier / 'annotations.npz').exists())
        ann = np.load(ann_dir / 'annotations.npz')
        seg = ann['segmentations'][4:].astype(np.int64)
        is_dynamic = ann['is_dynamic'].astype(bool)
        latents = {}
        # Prefix-only samples first; the future posterior is only for the AE rows.
        for label, model in flows.items():
            ctx = model.prefix_context(*prefix_batch(bank, [identifier], device))
            for seed in (17, 29):
                noise = clip_noise([identifier], seed, codec.objects, codec.latent_dim, device, torch.float32)
                with torch.no_grad():
                    latents[f'{label}_s{seed}'] = (model.codec, model.sample(ctx, noise, 16))
        for label, model in flows.items():
            _, mu, _, _ = model.target_latent(*bank.batch([identifier], device))
            latents[f'{label}_AE'] = (model.codec, mu)
        rendered = {name: views(c, z) for name, (c, z) in latents.items()}
        methods = {name: ownership_stats(r[1], seg, is_dynamic, codec.objects) for name, r in rendered.items()}
        methods['fixed_anchor_reference'] = ownership_stats(reference, seg, is_dynamic, codec.objects)
        record = {'id': identifier, 'num_instances': int(len(is_dynamic)),
                  'dynamic_instances': int(is_dynamic.sum()), 'methods': methods}
        for group, ids in (('showcase4', showcase), ('pilot32', pilot)):
            if identifier in ids:
                per_clip[group].append(record)
        if identifier in view_ids:
            truth = (np.asarray(bank.rgb(identifier))[4:]).astype(np.uint8)
            inst = instance_colors(is_dynamic)[seg]
            g = a.gif_flow
            panels = [truth, inst,
                      (rendered[f'{g}_AE'][0] * 255).astype(np.uint8), colors[rendered[f'{g}_AE'][1]],
                      (rendered[f'{g}_s17'][0] * 255).astype(np.uint8), colors[rendered[f'{g}_s17'][1]]]
            save_gif(a.out / f'owners_{identifier}.gif', panels,
                     ['GT', 'GT instances', 'AE (sees future)', 'AE owners', 'sample s17', 'sample owners'])
            gt_tracks, record['gt_track_check'] = dynamic_tracks(ann, seg, is_dynamic)
            save_trajectories(a.out / f'tracks_{identifier}.png', truth[0],
                              [rendered[f'{g}_AE'][2], rendered[f'{g}_s17'][2], rendered[f'{g}_s29'][2]],
                              colors, gt_tracks, ['AE (sees future)', 'sample s17', 'sample s29'])
        atomic_json(a.out / 'progress.json', {'done': clips.index(identifier) + 1, 'total': len(clips)})
    summary = {'format': 'claude_object_views_v2', 'codec': codec_identity, 'flows': identities,
               'gif_flow': a.gif_flow,
               'size': SIZE, 'min_pixels': MIN_PIXELS, 'showcase4': showcase, 'pilot32': pilot,
               'view_ids': view_ids, 'summary': {k: summarize(v) for k, v in per_clip.items()},
               'per_clip': per_clip, 'elapsed_seconds': time.time() - start, 'device': str(device),
               'definitions': {
                   'owner': 'argmax over parents of the renderer mixing weights at each 128px future pixel',
                   'parents_for_80pct': 'fewest parents whose owned pixels cover 80% of an instance in a frame, mean over frames',
                   'overlap_with_first_frame': 'histogram intersection between the instance parent distribution at frame t and at its first visible future frame, mean over later frames; 1 = the same parents keep the instance',
                   'parent_purity': 'pixel-weighted fraction of each parent\'s pixels inside its majority GT label',
                   'fixed_anchor_reference': 'nearest-anchor partition that never moves; parents that ignore objects',
                   'scope': '<label>_AE sees the true future (posterior mean of that run\'s codec); <label>_s17/s29 use only the 4-frame prefix. Single training seed.'}}
    atomic_json(a.out / 'summary.json', summary)
    print(json.dumps({'elapsed_seconds': summary['elapsed_seconds'], 'out': str(a.out)}))


if __name__ == '__main__':
    main()

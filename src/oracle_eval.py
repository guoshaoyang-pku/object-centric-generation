#!/usr/bin/env python3
"""Evaluations of an oracle flow checkpoint (amendment F1), matched to the patch-grid ones.

cohort:  codec_flow_research.evaluate on a fixed cohort (as memory_eval.py).
views:   object_views.py's ownership statistics and panels, plus slot identity:
         the fraction of each dynamic instance's GT pixels owned by its own slot.
rollout: long_rollout.py's protocol (showcase4 + pilot32[:4], seeds 17/29, noise
         seed + 1000k, 16 Euler steps, 4 chunks of 20 at 128 px). The next
         prefix is the last 4 rendered frames (uint8) and the rendered slot
         ownership; present slots are renumbered first, as in training. No GT
         is used after the 4 observed frames.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle as orc  # noqa: E402  (puts the code_v4 base on sys.path)
from codec_research import digest, atomic_json  # noqa: E402
from codec_flow_research import evaluate, load_flow_state, prefix_batch  # noqa: E402
from eval_validation import clip_noise  # noqa: E402
import object_views as ov  # noqa: E402
from long_rollout import edge_strength, labelled  # noqa: E402


def load(a, device):
    bank = orc.OracleBank(a.data, a.cache, a.seg)
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    proto = ckpt['protocol']
    if ckpt.get('format') != 'claude_oracle_v1' or proto['stage'] != 'flow':
        raise ValueError('Expected an oracle flow checkpoint')
    if digest(a.codec_checkpoint) != proto['codec_sha256']:
        raise ValueError('Codec checkpoint differs from the one the flow was trained on')
    ck = torch.load(a.codec_checkpoint, map_location='cpu', weights_only=False)
    cfg = ck['protocol']['model_config']
    codec = orc.OracleCodec(**{k: cfg[k] for k in ['slots', 'bg_side', 'latent_dim', 'subkernels', 'texture_kernels']})
    codec.load_state_dict(ck['model'], strict=True)
    codec = codec.to(device).requires_grad_(False).eval()
    w, d, h = proto['small_architecture']
    flow = orc.OracleObjectFlow(codec.objects, codec.latent_dim, cfg['feature_width'], codec.decoder.anchors, w, d, h)
    if a.which == 'ema':
        flow.load_state_dict(ckpt['ema_flow'], strict=True)
    else:
        load_flow_state(flow, ckpt['flow'])
    model = orc.OracleFlowModel(codec, flow.to(device)).eval()
    identity = {'checkpoint': str(a.checkpoint), 'sha256': digest(a.checkpoint), 'which': a.which,
                'step': ckpt['step'], 'train_gpu_hours': ckpt['train_gpu_seconds'] / 3600,
                'codec_sha256': proto['codec_sha256']}
    return bank, model, identity, json.loads(a.cohorts.read_text())


@torch.no_grad()
def render(codec, z, size=128, frames=20):
    """RGB [T,S,S,3] in [0,1], owner parent ids [T,S,S], parent centers [T,N,2] in pixels."""
    video, owner = codec.render_latent(z, size, frames, owners=True)
    p = codec.decoder.decode(z.float())
    centers, _ = codec.decoder.motion_at(p, torch.linspace(-1, 1, frames, device=z.device)[None])
    return (video[0].clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy(), owner[0].cpu().numpy().astype(np.int16),
            ((centers[0] + 1) * (size - 1) / 2).cpu().numpy())


def slot_identity(owner, lab, seg, is_dynamic, slots):
    """Per dynamic instance and frame (>= 20 px): share of its pixels owned by its own slot."""
    group = np.minimum(owner, slots)
    rows = []
    for k in np.flatnonzero(is_dynamic):
        values = []
        for t in range(seg.shape[0]):
            mask = seg[t] == k + 1
            if mask.sum() >= ov.MIN_PIXELS:
                values.append(float((group[t][mask] == lab[t][mask]).mean()))
        if values:
            rows.append(float(np.mean(values)))
    return {'dynamic_own_slot_share': float(np.mean(rows)) if rows else None, 'dynamic_instances': len(rows),
            'pixel_ownership_accuracy': float((group == lab).mean())}


def run_views(a, bank, model, identity, cohorts, device):
    codec = model.codec
    showcase = [r['id'] for r in cohorts['showcase4']]
    pilot = [r['id'] for r in cohorts['pilot32']]
    clips = list(dict.fromkeys(showcase + pilot))
    view_ids = showcase + pilot[:a.gif_pilot]
    colors = ov.palette(codec.objects)
    per_clip = {'showcase4': [], 'pilot32': []}
    for identifier in clips:
        with np.load(a.seg / bank.records[identifier]['split'] / (identifier + '.npz')) as f:
            ann = {k: f[k] for k in f.files}
        seg = ann['segmentations'][4:].astype(np.int64)
        is_dynamic = ann['is_dynamic'].astype(bool)
        lab = bank.label(identifier)[4:].astype(np.int64)
        ctx = model.prefix_context(*prefix_batch(bank, [identifier], device))
        latents = {}
        for seed in (17, 29):
            noise = clip_noise([identifier], seed, codec.objects, codec.latent_dim, device, torch.float32)
            latents[f's{seed}'] = model.sample(ctx, noise, 16)
        _, mu, _, _ = model.target_latent(*bank.batch([identifier], device))
        latents['AE'] = mu
        rendered = {name: render(codec, z) for name, z in latents.items()}
        methods = {}
        for name, r in rendered.items():
            methods[name] = {**ov.ownership_stats(r[1], seg, is_dynamic, codec.objects),
                             **slot_identity(r[1], lab, seg, is_dynamic, codec.slots)}
        record = {'id': identifier, 'num_instances': int(len(is_dynamic)),
                  'dynamic_instances': int(is_dynamic.sum()), 'slot_info': bank.slot_info[identifier], 'methods': methods}
        for group, ids in (('showcase4', showcase), ('pilot32', pilot)):
            if identifier in ids:
                per_clip[group].append(record)
        if identifier in view_ids:
            truth = np.asarray(bank.rgb(identifier))[4:].astype(np.uint8)
            inst = ov.instance_colors(is_dynamic)[seg]
            panels = [truth, inst, (rendered['AE'][0] * 255).astype(np.uint8), colors[rendered['AE'][1]],
                      (rendered['s17'][0] * 255).astype(np.uint8), colors[rendered['s17'][1]]]
            ov.save_gif(a.out / f'owners_{identifier}.gif', panels,
                        ['GT', 'GT instances', 'AE (sees future)', 'AE owners', 'sample s17', 'sample owners'])
            gt_tracks, record['gt_track_check'] = ov.dynamic_tracks(ann, seg, is_dynamic)
            # Only object slots that own pixels somewhere; background parents stay on their anchors.
            ov.save_trajectories(a.out / f'tracks_{identifier}.png', truth[0],
                                 [rendered[k][2] for k in ('AE', 's17', 's29')], colors, gt_tracks,
                                 ['AE (sees future)', 'sample s17', 'sample s29'])
        atomic_json(a.out / 'progress.json', {'done': clips.index(identifier) + 1, 'total': len(clips)})

    def extra(records, name):
        out = {}
        for key in ['dynamic_own_slot_share', 'pixel_ownership_accuracy']:
            vals = [r['methods'][name][key] for r in records if r['methods'][name][key] is not None]
            out[key] = float(np.mean(vals)) if vals else None
        return out

    summary = {'format': 'claude_oracle_views_v1', 'flow': identity, 'view_ids': view_ids,
               'summary': {g: {m: {**s, **extra(per_clip[g], m)} for m, s in ov.summarize(v).items()}
                           for g, v in per_clip.items()},
               'per_clip': per_clip,
               'definitions': {'dynamic_own_slot_share': 'per dynamic instance and frame with >=20 px, share of its GT pixels '
                                                         'whose owner is its own slot; mean over frames, then instances',
                               'pixel_ownership_accuracy': 'owner group (object slot, or background) equals the GT slot label',
                               'other keys': 'as object_views.py; AE sees the true future, s17/s29 only the 4-frame prefix'}}
    atomic_json(a.out / 'summary.json', summary)
    return summary['summary']


def relabel(group, slots):
    """Present slots first in their old order, then the absent ones: the training layout of a prefix."""
    present = [k for k in range(slots) if (group == k).any()]
    order = present + [k for k in range(slots) if k not in present]
    lut = np.full(slots + 1, slots, np.uint8)
    for new, old in enumerate(order):
        lut[old] = new
    return lut[group], order


def run_rollout(a, bank, model, identity, cohorts, device):
    codec, slots, size = model.codec, model.codec.slots, 128
    clips = [r['id'] for r in cohorts['showcase4']] + [r['id'] for r in cohorts['pilot32']][:a.pilot]
    seeds = [17, 29]
    records = []
    for identifier in clips:
        real_u8 = np.asarray(bank.rgb(identifier)).astype(np.uint8)
        real = torch.from_numpy(real_u8).permute(0, 3, 1, 2).float() / 255
        truth = real[4:]
        rollouts = {}
        for seed in seeds:
            rgb_p, lab_p = prefix_batch(bank, [identifier], device)
            frames, chunk_stats = [real[:4]], []
            for k in range(a.chunks):
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    context = model.prefix_context(rgb_p, lab_p)
                noise = clip_noise([identifier], seed + 1000 * k, codec.objects, codec.latent_dim, device, torch.float32)
                z = model.sample(context, noise, 16)
                video, owner, _ = render(codec, z, size)
                vid = torch.from_numpy(video).permute(0, 3, 1, 2)
                frames.append(vid)
                group = np.minimum(owner, slots)
                stat = {'chunk': k, 'noise_seed': seed + 1000 * k,
                        'boundary_jump_l1': float((vid[0] - frames[-2][-1]).abs().mean()),
                        'within_chunk_step_l1': float((vid[1:] - vid[:-1]).abs().mean()),
                        'mse_vs_truth': float((vid - truth).square().mean()) if k == 0 else None,
                        'object_slots_owning_pixels': int(len(np.unique(group[group < slots])))}
                if k == 0:
                    lab = bank.label(identifier)[4:].astype(np.int64)
                    stat['pixel_ownership_accuracy_vs_gt'] = float((group == lab).mean())
                chunk_stats.append(stat)
                new_lab, _ = relabel(group[-4:], slots)
                rgb_p = torch.from_numpy((video[-4:] * 255).round().astype(np.uint8))[None].to(device)
                lab_p = torch.from_numpy(new_lab)[None].to(device)
            seq = torch.cat(frames)
            anchor = seq[3]
            rollouts[seed] = seq
            per_frame = [{'t': t, 'mean_rgb_drift': float((seq[t].mean((1, 2)) - anchor.mean((1, 2))).abs().mean()),
                          'edge_ratio': edge_strength(seq[t]) / max(edge_strength(anchor), 1e-8)}
                         for t in range(seq.shape[0])]
            records.append({'id': identifier, 'seed': seed, 'chunks': chunk_stats, 'per_frame': per_frame})
        total = 4 + 20 * a.chunks
        gif = []
        for t in range(total):
            k = (t - 4) // 20 if t >= 4 else -1
            tag = 'observed' if t < 4 else f'chunk {k}'
            gt = real_u8[t] if t < 24 else np.full((size, size, 3), 40, np.uint8)
            panels = [labelled(gt, f'GT t={t}' if t < 24 else 'no GT')]
            for seed in seeds:
                f = (rollouts[seed][t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                panels.append(labelled(f, f's{seed} {tag}'))
            gif.append(Image.fromarray(np.concatenate(panels, 1)).convert('P', palette=Image.ADAPTIVE, colors=255))
        gif[0].save(a.out / f'rollout_{identifier}.gif', save_all=True, append_images=gif[1:],
                    duration=1000 // 12, loop=0, optimize=True)
        picks = [3] + [3 + 20 * (k + 1) for k in range(a.chunks)]
        rows = [np.concatenate([labelled((rollouts[s][t].permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                                         f's{s} t={t}') for t in picks], 1) for s in seeds]
        Image.fromarray(np.concatenate(rows, 0)).save(a.out / f'strip_{identifier}.png', optimize=True)
        atomic_json(a.out / 'progress.json', {'done': clips.index(identifier) + 1, 'total': len(clips)})
    summary = {'format': 'claude_oracle_rollout_v1', 'flow': identity, 'chunks': a.chunks,
               'frames_total': 4 + 20 * a.chunks, 'seeds': seeds, 'clips': clips, 'size': size,
               'mean_over_clips': {
                   f'chunk_{k}': {
                       'boundary_jump_l1': float(np.mean([r['chunks'][k]['boundary_jump_l1'] for r in records])),
                       'within_chunk_step_l1': float(np.mean([r['chunks'][k]['within_chunk_step_l1'] for r in records])),
                       'end_mean_rgb_drift': float(np.mean([r['per_frame'][3 + 20 * (k + 1)]['mean_rgb_drift'] for r in records])),
                       'end_edge_ratio': float(np.mean([r['per_frame'][3 + 20 * (k + 1)]['edge_ratio'] for r in records])),
                       'object_slots_owning_pixels': float(np.mean([r['chunks'][k]['object_slots_owning_pixels'] for r in records]))}
                   for k in range(a.chunks)},
               'chunk0_mse_vs_truth': float(np.mean([r['chunks'][0]['mse_vs_truth'] for r in records])),
               'chunk0_pixel_ownership_accuracy_vs_gt': float(np.mean([r['chunks'][0]['pixel_ownership_accuracy_vs_gt']
                                                                       for r in records])),
               'records': records,
               'scope': 'Only chunk 0 has ground truth (and GT masks only for the 4 observed frames as condition). '
                        'Later chunks condition on generated frames and their rendered ownership; self-consistency only.'}
    atomic_json(a.out / 'summary.json', summary)
    return {k: summary[k] for k in ['mean_over_clips', 'chunk0_mse_vs_truth', 'chunk0_pixel_ownership_accuracy_vs_gt']}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['cohort', 'views', 'rollout'])
    for name in ['data', 'cache', 'seg', 'cohorts', 'checkpoint', 'codec-checkpoint', 'out']:
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--which', choices=['ema', 'raw'], default='ema')
    ap.add_argument('--cohort', default='confirmation64')
    ap.add_argument('--count', type=int, default=0)
    ap.add_argument('--gif-pilot', type=int, default=4)
    ap.add_argument('--pilot', type=int, default=4)
    ap.add_argument('--chunks', type=int, default=4)
    ap.add_argument('--deadline-unix', type=float, default=float('inf'))
    a = ap.parse_args()
    device = torch.device('cuda')
    started = time.time()
    bank, model, identity, cohorts = load(a, device)
    if a.mode == 'cohort':
        ids = [r['id'] for r in cohorts[a.cohort]]
        ids = ids[:a.count] if a.count else ids
        result = evaluate(model, bank, ids, [], a.out, 64, identity['step'], device, True, a.deadline_unix)
        atomic_json(a.out / 'oracle_eval_meta.json', {'flow': identity, 'cohort': a.cohort, 'count': len(ids),
                                                      'elapsed_seconds': time.time() - started})
        m = result['means']
        print(json.dumps({'cohort': a.cohort, 'which': a.which, 'n': len(ids),
                          's16': m['sample_16_seed_average']['mse'], 's4': m['sample_4_seed_average']['mse'],
                          'ae': m['future_AE']['scalar']['mse'],
                          'motion_cos_s16_17': m['sample_16_seed17']['scalar']['temporal_delta_cosine']}))
        return
    a.out.mkdir(parents=True, exist_ok=False)
    with torch.inference_mode():
        out = run_views(a, bank, model, identity, cohorts, device) if a.mode == 'views' else \
            run_rollout(a, bank, model, identity, cohorts, device)
    print(json.dumps(out, indent=1)[:4000])
    print(json.dumps({'elapsed_seconds': time.time() - started}))


if __name__ == '__main__':
    main()

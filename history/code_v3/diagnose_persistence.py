"""Bounded reconstruction/continuation diagnostic; primitive IDs are NOT objects.

Run identically against baseline and later checkpoints in separate output folders.
--compare-summary binds the same clips, hashes, resolution, steps and noise seed.
No optimizer, model updates, semantic tracking or physical-validity claims.
"""
import argparse
import colorsys
import hashlib
import json
import os
from pathlib import Path
import time


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def time_metrics(pred, truth, previous):
    """T,C,H,W arrays; first delta starts at the last observed RGB frame."""
    import numpy as np
    p, y = pred.astype(np.float64), truth.astype(np.float64)
    pd = np.diff(np.concatenate([previous[None], p]), axis=0)
    yd = np.diff(np.concatenate([previous[None], y]), axis=0)
    def edge(x):
        return (np.abs(np.diff(x, axis=-1)).mean((1, 2, 3)) +
                np.abs(np.diff(x, axis=-2)).mean((1, 2, 3))) / 2
    dots = (pd * yd).sum((1, 2, 3))
    den = np.sqrt((pd * pd).sum((1, 2, 3)) * (yd * yd).sum((1, 2, 3)))
    return {'mse': ((p-y)**2).mean((1, 2, 3)).tolist(),
            'edge_l1': edge(p).tolist(), 'truth_edge_l1': edge(y).tolist(),
            'motion_l1': np.abs(pd).mean((1, 2, 3)).tolist(),
            'truth_motion_l1': np.abs(yd).mean((1, 2, 3)).tolist(),
            'motion_delta_mse': ((pd-yd)**2).mean((1, 2, 3)).tolist(),
            'motion_delta_cosine': [float(a/b) if b > 1e-12 else None for a, b in zip(dots, den)]}


def parent_maps(model, z, size, deadline):
    import numpy as np
    import torch
    t = model.future_frames
    tt, yy, xx = torch.meshgrid(torch.linspace(-1, 1, t, device=z.device),
                               torch.linspace(-1, 1, size, device=z.device),
                               torch.linspace(-1, 1, size, device=z.device), indexing='ij')
    queries = torch.stack([xx, yy, tt], -1).reshape(1, -1, 3)
    values = []
    with torch.autocast(device_type=z.device.type, enabled=False):
        params = model.decoder.decode(z.float())
        for q in queries.split(256, dim=1):
            deadline()
            local = torch.einsum('boij,bqoj->bqoi', params['inv'], q[:, :, None]-params['mean'][:, None])
            child = (local[:, :, :, None]-params['offset'][:, None])/params['scale'][:, None]
            density = ((-.5*child.square().sum(-1)).exp()*params['amplitude'][:, None]).sum(-1)
            weights = density/density.sum(-1, keepdim=True).clamp_min(1e-8)
            color = params['color'][:, None]
            if model.decoder.texture_kernels:
                color = color+model.decoder.texture(local, params)-params['dc'][:, None]
            actual = model.decoder.render(params, q)
            if not torch.allclose((weights[..., None]*color.sigmoid()).sum(2), actual, atol=1e-6, rtol=1e-6):
                raise ValueError('Parent contribution implementation differs from RGB renderer')
            values.append(weights[0].cpu().numpy())
    weights = np.concatenate(values).reshape(t, size, size, model.objects)
    if not np.isfinite(weights).all():
        raise ValueError('Nonfinite parent weights')
    mass = weights.mean((1, 2), dtype=np.float64)  # T, parents
    peak = mass.max(0)
    support = ((mass > .01*peak[None]) & (peak[None] > 1e-12)).sum(0)
    return {'mean_weight_by_frame_parent': mass.tolist(),
            'soft_support_frames_above_1pct_own_peak': support.tolist(),
            'normalization_sum_min': float(weights.sum(-1).min()),
            'normalization_sum_max': float(weights.sum(-1).max()),
            'definition': 'Spatial mean of normalized parent density per future frame; soft support threshold is 1% own temporal peak. Not semantic identity, geometric volume, visibility or object survival.'}, weights.argmax(-1).astype(np.uint16)


def save_gif(path, rows, labels, fps, start_frame=4):
    import numpy as np
    from PIL import Image, ImageDraw
    size, head = 128, 34
    frames = []
    for t in range(rows[0][1][0].shape[0]):
        canvas = Image.new('RGB', (len(labels)*size, len(rows)*(size+head)), '#141414')
        draw = ImageDraw.Draw(canvas)
        for r, (identifier, videos) in enumerate(rows):
            for c, video in enumerate(videos):
                x, y = c*size, r*(size+head)
                draw.text((x+2, y+1), labels[c], fill='white')
                draw.text((x+2, y+15), f'{identifier[-18:]} f{t+start_frame}', fill='white')
                image = Image.fromarray(np.uint8(np.clip(video[t], 0, 1)*255)).resize((size, size), Image.Resampling.NEAREST)
                canvas.paste(image, (x, y+head))
        frames.append(canvas)
    durations = [round((i+1)*100/fps)*10-round(i*100/fps)*10 for i in range(len(frames))]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=False, disposal=2)
    frames[len(frames)//2].save(path.with_suffix('.png'))


def run(a):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from eval_validation import clip_noise, frozen_fingerprint
    from wan_model import PretrainedJointObjectVideo
    start = time.monotonic()
    def deadline():
        if time.monotonic()-start > a.max_seconds:
            raise TimeoutError('Diagnostic time budget reached; completed per-clip files remain')
    if a.out.exists() and any(a.out.iterdir()):
        raise ValueError('Use a new empty output directory; refusing mixed diagnostics')
    if digest(a.checkpoint) != a.checkpoint_sha:
        raise ValueError('Checkpoint SHA mismatch')
    manifest = json.loads((a.data/'manifest.json').read_text())
    if manifest.get('test_split_used'):
        raise ValueError('Test split is not admitted')
    selected = []
    for split in ('train', 'validation'):
        candidates = sorted((r for r in manifest['samples'] if r['split'] == split), key=lambda r: r['id'])
        if len(candidates) < a.clips_per_split:
            raise ValueError(f'Insufficient {split} clips')
        # Fixed sorted leading IDs, chosen before model output, identical across checkpoints.
        selected.extend(candidates[:a.clips_per_split])
    if len({r['id'] for r in selected}) != len(selected):
        raise ValueError('Duplicate clip IDs')
    if any(Path(r['id']).name != r['id'] for r in selected):
        raise ValueError('Clip IDs must be safe filenames')
    arrays = {}
    for r in selected:
        path = a.data/r['array']
        if digest(path) != r['array_sha256']:
            raise ValueError(f'Array SHA mismatch: {r["id"]}')
        x = np.load(path, allow_pickle=False)
        if x.shape != (24, 128, 128, 3) or x.dtype != np.uint8:
            raise ValueError('Requires uint8 24x128x128x3 MOVi RGB')
        arrays[r['id']] = x
    protocol = {'version': 'persistence-diagnostic-v1', 'clips': [
        {k:r[k] for k in ('id','split','array','array_sha256')} for r in selected],
        'selection': 'First N sorted IDs per train and validation split; no output selection',
        'context_frames': 4, 'future_frames': 20, 'metric_size': a.size,
        'parent_grid_size': a.parent_size, 'sampling_steps': a.steps, 'seed': a.seed,
        'noise': 'eval_validation.clip_noise CPU FP32 clip-ID derived noise; same noise for all step counts',
        'warning': 'Small diagnostic cohort, not generalization or object-identity benchmark. Future AE sees target. Primitive IDs are not semantic masks.'}
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    old = json.loads(a.compare_summary.read_text()) if a.compare_summary else None
    if old and old['protocol_sha256'] != signature:
        raise ValueError('Comparison requires identical cohort hashes, metric settings and seeds')
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cfg = dict(checkpoint['args']); cfg['pretrained_path'] = str(a.pretrained)
    if cfg['context_frames'] != 4 or cfg['future_frames'] != 20:
        raise ValueError('Checkpoint context/future contract differs')
    frozen = frozen_fingerprint(a.pretrained)
    if frozen['fingerprint_sha256'] != cfg['pretrained_fingerprint_sha256']:
        raise ValueError('Pretrained Transformer/VAE fingerprint mismatch')
    a.out.mkdir(parents=True, exist_ok=True)
    write_json(a.out/'protocol.json', protocol)
    identity = {'checkpoint_sha256': a.checkpoint_sha, 'checkpoint_step': checkpoint['step'],
                'protocol_sha256': signature, 'manifest_sha256': digest(a.data/'manifest.json'),
                'source_sha256': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
                'frozen_pretrained_fingerprint': frozen['fingerprint_sha256'],
                'device': None, 'no_optimizer': True}
    torch.set_num_threads(4); torch.cuda.set_device(0); torch.manual_seed(20261026)
    model = PretrainedJointObjectVideo(cfg)
    missing = model.load_state_dict(checkpoint['model'], strict=False)
    required = {n for n,p in model.named_parameters() if p.requires_grad} | {
        n for n,_ in model.named_buffers() if n in model.state_dict() and not n.startswith(('vae.', 'flow.backbone.'))}
    if required.intersection(missing.missing_keys) or missing.unexpected_keys:
        raise ValueError('Checkpoint missing required trainables/buffers or has unexpected state')
    model = model.cuda().eval(); identity['device'] = torch.cuda.get_device_name(0)
    write_json(a.out/'identity.json', identity)
    rows, all_records = [], []
    names = ['future_AE']+[f'sample_{s}' for s in a.steps]
    palette = np.array([colorsys.hsv_to_rgb((i*.61803398875)%1,.75,1) for i in range(model.objects)], np.float32)
    with torch.inference_mode():
        for r in selected:
            deadline()
            full = torch.from_numpy(arrays[r['id']]).permute(0,3,1,2)[None].cuda().float()/255
            prefix, future = full[:,:4], full[:,4:]
            truth = F.interpolate(future[0], (a.size,a.size), mode='bilinear', align_corners=False).cpu().numpy()
            previous = F.interpolate(prefix[0,-1:], (a.size,a.size), mode='bilinear', align_corners=False)[0].cpu().numpy()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                condition = model.visual_features(prefix, time_start=0)
                _, mu, _, _ = model._future_posterior(future)
                latents = {'future_AE': mu}
                for steps in a.steps:
                    z = clip_noise([r['id']], a.seed, model.objects, model.latent_dim, condition.device, condition.dtype)
                    for step in range(steps):
                        deadline()
                        t = torch.full((1,), step/steps, device=z.device, dtype=z.dtype)
                        z = z+model.flow(z,t,condition)/steps
                    latents[f'sample_{steps}'] = z
            item = {'id':r['id'], 'split':r['split'], 'array_sha256':r['array_sha256'], 'methods':{}, 'parents':{}}
            videos, ids = [], []
            for name in names:
                deadline()
                pred = model.render_latent(latents[name], a.size, a.size)[0].float().cpu().numpy()
                if not np.isfinite(pred).all():
                    raise ValueError('Nonfinite RGB')
                item['methods'][name] = time_metrics(pred, truth, previous)
                item['parents'][name], dominant = parent_maps(model, latents[name], a.parent_size, deadline)
                videos.append(pred.transpose(0,2,3,1)); ids.append(palette[dominant])
            item['methods']['persistence'] = time_metrics(np.repeat(previous[None],20,axis=0), truth, previous)
            write_json(a.out/(r['id']+'.json'),item)
            save_gif(a.out/(r['id']+'_parents.gif'), [(r['id'],ids)], ['AE parent ID']+[f'{s}step parent ID' for s in a.steps], a.fps)
            rows.append((r['id'],[truth.transpose(0,2,3,1)]+videos))
            all_records.append(item)
            write_json(a.out/'progress.json', {'completed':len(all_records),'total':len(selected),'elapsed_seconds':time.monotonic()-start})
            print(json.dumps({'clip':r['id'],'completed':len(all_records),'elapsed_seconds':time.monotonic()-start}),flush=True)
    save_gif(a.out/'comparison.gif', rows, ['Ground truth','Future AE TARGET SEEN']+[f'Prefix only {s} steps' for s in a.steps],a.fps)
    groups = {}
    for split in ('train','validation'):
        subset = [r for r in all_records if r['split']==split]
        groups[split] = {}
        for name in subset[0]['methods']:
            groups[split][name] = {}
            for key in subset[0]['methods'][name]:
                values = [[r['methods'][name][key][t] for r in subset if r['methods'][name][key][t] is not None] for t in range(20)]
                groups[split][name][key] = [float(np.mean(x)) if x else None for x in values]
    result = {**identity, 'protocol':protocol, 'groups_mean_by_future_frame':groups,
              'elapsed_seconds':time.monotonic()-start,'status':'complete',
              'interpretation':'AE failure suggests representation/decoder/posterior limitation; AE success with generation failure localizes gap to conditional generation but does not isolate cause. RGB motion/edge and density contributions are not semantic identity or physical fidelity metrics.'}
    if old:
        result['comparison'] = {'reference_checkpoint_sha256':old['checkpoint_sha256'],
            'delta_mse_by_future_frame':{split:{name:[x-y for x,y in zip(groups[split][name]['mse'],old['groups_mean_by_future_frame'][split][name]['mse'])] for name in groups[split]} for split in groups}}
    write_json(a.out/'summary.json',result)
    write_json(a.out/'checksums.json',{str(p.relative_to(a.out)):digest(p) for p in sorted(a.out.iterdir()) if p.is_file() and p.name!='checksums.json'})
    print(json.dumps({'status':'complete','out':str(a.out),'elapsed_seconds':time.monotonic()-start}),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data','checkpoint','pretrained','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--checkpoint-sha',required=True)
    p.add_argument('--compare-summary',type=Path)
    p.add_argument('--clips-per-split',type=int,default=2)
    p.add_argument('--steps',type=int,nargs='+',default=[4,16])
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--size',type=int,default=64)
    p.add_argument('--parent-size',type=int,default=32)
    p.add_argument('--fps',type=int,default=12)
    p.add_argument('--max-seconds',type=int,default=900)
    a = p.parse_args()
    if min(a.clips_per_split,a.size,a.parent_size,a.fps,a.max_seconds,*a.steps)<1 or len(set(a.steps))!=len(a.steps):
        p.error('Positive bounds and distinct sampling step counts required')
    if min(a.size,a.parent_size)<2:
        p.error('Spatial grids must be >=2 for edge metrics')
    if a.clips_per_split>8 or a.size>128 or a.parent_size>64 or max(a.steps)>64:
        p.error('Bounded diagnostic limits: <=8 clips per split, size<=128, parent size<=64, steps<=64')
    run(a)


if __name__ == '__main__':
    main()

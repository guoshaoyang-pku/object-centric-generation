"""Bounded cached-feature AE experiments; no video-generation backbone is loaded.

A/xyt_t and B/xyt_xy differ only in the posterior's time-locality bias.
C/trajectory changes to persistent parents with shared affine Bezier motion.
All displayed reconstructions have seen their true future. This runner does
not measure prefix-only generation, semantic object identity, or physics.
"""
import argparse
from collections import OrderedDict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from .model import CompositeDecoder, ObjectPosterior
    from .trajectory_codec import TrajectoryDecoder
    from .wan_model import position_features
    from .diagnose_persistence import time_metrics, save_gif
except ImportError:
    from model import CompositeDecoder, ObjectPosterior
    from trajectory_codec import TrajectoryDecoder
    from wan_model import position_features
    from diagnose_persistence import time_metrics, save_gif


SHOWCASE = ['validation_s00011_r002', 'validation_s00012_r002',
            'validation_s00004_r011', 'validation_s00004_r013']
ALIASES = {'A': 'xyt_t', 'B': 'xyt_xy', 'C': 'trajectory'}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


class Codec(nn.Module):
    def __init__(self, variant='xyt_t', objects=250, latent_dim=128,
                 feature_width=256, subkernels=4, texture_kernels=None,
                 posterior_sigma=(.4, .5, .5)):
        super().__init__()
        variant = ALIASES.get(variant, variant)
        if variant not in {'xyt_t', 'xyt_xy', 'trajectory'}:
            raise ValueError('Unknown codec variant')
        if texture_kernels is None:
            texture_kernels = 109 if variant == 'trajectory' else 64
        self.config = dict(variant=variant, objects=int(objects), latent_dim=int(latent_dim),
                           feature_width=int(feature_width), subkernels=int(subkernels),
                           texture_kernels=int(texture_kernels), posterior_sigma=list(posterior_sigma))
        self.variant, self.objects, self.latent_dim = variant, objects, latent_dim
        self.future_frames, self.context_frames = 20, 4
        self.visual_adapter = nn.Sequential(nn.Linear(48, feature_width), nn.LayerNorm(feature_width),
                                            nn.GELU(), nn.Linear(feature_width, feature_width))
        self.posterior = ObjectPosterior(objects, feature_width, latent_dim)
        if variant == 'trajectory':
            self.decoder = TrajectoryDecoder(objects, latent_dim, subkernels, texture_kernels,
                                             motion_init='static')
            initial_position = F.pad(self.decoder.anchors, (0, 1))
        else:
            self.decoder = CompositeDecoder(objects, latent_dim, subkernels, texture_kernels)
            initial_position = self.decoder.anchors
        with torch.no_grad():
            self.posterior.queries.add_(.1 * position_features(initial_position, feature_width)[None])

    def encode_future(self, features, coords, localcoords):
        if features.ndim != 3 or features.shape[-1] != 48:
            raise ValueError('Expected frozen future features [B,N,48]')
        if coords.ndim == 2:
            coords = coords[None].expand(features.shape[0], -1, -1)
        if localcoords.ndim == 2:
            localcoords = localcoords[None].expand(features.shape[0], -1, -1)
        if coords.shape != features.shape[:2] + (3,) or localcoords.shape != coords.shape:
            raise ValueError('Future coordinate shape differs from features')
        # Cached clips in this protocol share the same VAE geometry. Checking
        # this makes the shared MultiheadAttention bias explicit, not accidental.
        if not torch.equal(localcoords, localcoords[:1].expand_as(localcoords)):
            raise ValueError('Mixed future token coordinate grids are not supported')
        adapted = self.visual_adapter(features)
        adapted = adapted + .1 * position_features(coords, adapted.shape[-1]).to(adapted.dtype)
        sigma = localcoords.new_tensor(self.config['posterior_sigma'])
        axes = 3 if self.variant == 'xyt_t' else 2
        anchors = self.decoder.anchors[..., :axes]
        delta = (anchors[:, None] - localcoords[0, None, :, :axes]) / sigma[:axes]
        bias = -.5 * delta.square().sum(-1)
        return self.posterior(adapted, attn_bias=bias)

    def render_queries(self, z, coords):
        with torch.autocast(device_type=z.device.type, enabled=False):
            return self.decoder.render(self.decoder.decode(z.float()), coords.float())

    def render_latent(self, z, size=64, frames=20, chunk=256):
        tt, yy, xx = torch.meshgrid(torch.linspace(-1, 1, frames, device=z.device),
                                    torch.linspace(-1, 1, size, device=z.device),
                                    torch.linspace(-1, 1, size, device=z.device), indexing='ij')
        queries = torch.stack((xx, yy, tt), -1).reshape(1, -1, 3).expand(z.shape[0], -1, -1)
        with torch.autocast(device_type=z.device.type, enabled=False):
            params = self.decoder.decode(z.float())
            rgb = torch.cat([self.decoder.render(params, q) for q in queries.split(chunk, dim=1)], dim=1)
        return rgb.reshape(z.shape[0], frames, size, size, 3).permute(0, 1, 4, 2, 3)

    def parameter_counts(self):
        result = {name: sum(p.numel() for p in module.parameters())
                  for name, module in [('visual_adapter', self.visual_adapter),
                                       ('posterior', self.posterior), ('decoder', self.decoder)]}
        result.update(total=sum(result.values()), geometry_fields_per_object=self.decoder.geometry_dim,
                      texture_fields_per_object=self.decoder.texture_dim,
                      geometry_fields_per_clip=self.objects * self.decoder.geometry_dim,
                      texture_fields_per_clip=self.objects * self.decoder.texture_dim,
                      latent_values_per_clip=self.objects * self.latent_dim)
        return result


class CacheBank:
    """Validate each immutable RGB/cache artifact before its first use."""
    def __init__(self, data, cache):
        self.data, self.cache = Path(data), Path(cache)
        self.manifest = json.loads((self.data / 'manifest.json').read_text())
        self.manifest_sha = digest(self.data / 'manifest.json')
        self.protocol = json.loads((self.cache / 'protocol.json').read_text())
        self.protocol_sha = canonical_sha(self.protocol)
        if self.manifest.get('test_split_used'):
            raise ValueError('Test split is not admitted')
        if self.protocol['source_manifest_sha256'] != self.manifest_sha:
            raise ValueError('Feature cache was produced from a different source manifest')
        if self.protocol['context_frames'] != 4 or self.protocol['future_frames'] != 20:
            raise ValueError('Requires cached 4+20 frame contract')
        if self.protocol['dtype'] != 'float32':
            raise ValueError('This matched experiment requires FP32 feature cache')
        records = self.manifest['samples']
        self.records = {r['id']: r for r in records}
        if len(self.records) != len(records):
            raise ValueError('Duplicate source clip IDs')
        self.metadata, self.feature_lru, self.verified_rgb = {}, OrderedDict(), {}

    def ids(self, split):
        return sorted(r['id'] for r in self.records.values() if r['split'] == split)

    def check_metadata(self, identifier):
        if identifier in self.metadata:
            return self.metadata[identifier]
        r = self.records[identifier]
        if r['split'] not in {'train', 'validation'} or Path(identifier).name != identifier:
            raise ValueError('Unsafe clip identity or split')
        if Path(r['array']).parts[0] != r['split'] or '..' in Path(r['array']).parts:
            raise ValueError('Unsafe RGB source path')
        meta = json.loads((self.cache / 'records' / r['split'] / (identifier + '.json')).read_text())
        if (meta['id'] != identifier or meta['split'] != r['split'] or
                meta['source_array_sha256'] != r['array_sha256'] or
                meta['protocol_sha256'] != self.protocol_sha):
            raise ValueError('Feature cache record provenance differs: ' + identifier)
        expected = str(Path('features') / r['split'] / (identifier + '.npz'))
        if meta['features'] != expected:
            raise ValueError('Unexpected feature path: ' + identifier)
        self.metadata[identifier] = meta
        return meta

    def features(self, identifier):
        if identifier in self.feature_lru:
            self.feature_lru.move_to_end(identifier)
            return self.feature_lru[identifier]
        meta = self.check_metadata(identifier)
        path = self.cache / meta['features']
        if digest(path) != meta['features_sha256']:
            raise ValueError('Feature file checksum mismatch: ' + identifier)
        with np.load(path, allow_pickle=False) as file:
            if set(file.files) != set(meta['arrays']):
                raise ValueError('Feature archive fields differ')
            arrays = {k: np.array(file[k], copy=True) for k in file.files}
        for k, value in arrays.items():
            spec = meta['arrays'][k]
            if list(value.shape) != spec['shape'] or str(value.dtype) != spec['dtype'] or not np.isfinite(value).all():
                raise ValueError('Invalid cached tensor: ' + identifier + '/' + k)
        if arrays['future'].ndim != 2 or arrays['future'].shape[-1] != 48:
            raise ValueError('Expected 48-channel future VAE features')
        self.feature_lru[identifier] = arrays
        if len(self.feature_lru) > 1024:
            self.feature_lru.popitem(last=False)
        return arrays

    def rgb(self, identifier):
        r = self.records[identifier]
        path = self.data / r['array']
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        if self.verified_rgb.get(identifier) != signature:
            if digest(path) != r['array_sha256']:
                raise ValueError('RGB checksum mismatch: ' + identifier)
            self.verified_rgb[identifier] = signature
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        if array.shape != (24, 128, 128, 3) or array.dtype != np.uint8:
            raise ValueError('Requires uint8 [24,128,128,3] source RGB')
        return array

    def batch(self, ids, device):
        arrays = [self.features(i) for i in ids]
        return tuple(torch.from_numpy(np.stack([a[key] for a in arrays])).to(device)
                     for key in ['future', 'future_coords', 'future_local_coords'])


def rng_state(device):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}


def restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if device.type == 'cuda' and state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'], device)


def pick_cohorts(a, bank):
    train = bank.ids('train')
    validation = bank.ids('validation')
    provenance = {'cohorts_sha256': None}
    if a.cohorts:
        document = json.loads(a.cohorts.read_text())
        if document['manifest_sha256'] != bank.manifest_sha:
            raise ValueError('Cohort and source manifest hashes differ')
        provenance['cohorts_sha256'] = digest(a.cohorts)
        def checked(group, split):
            result = []
            for r in document[group]:
                source = bank.records[r['id']]
                if source['split'] != split or source['array_sha256'] != r['sha256']:
                    raise ValueError('Cohort record differs from source: ' + r['id'])
                result.append(r['id'])
            if len(set(result)) != len(result):
                raise ValueError('Duplicate cohort IDs')
            return result
        validation = checked(a.eval_cohort, 'validation')
        showcase = checked('showcase4', 'validation')
        if a.train_cohort != 'all':
            train = checked(a.train_cohort, 'train')
    else:
        validation.sort(key=lambda i: hashlib.sha256(('codec-research-validation-v1|' + i).encode()).digest())
        showcase = [i for i in SHOWCASE if i in bank.records]
    if a.train_limit:
        train = train[:a.train_limit]
    validation = validation[:a.eval_count]
    if not train or not validation or len(validation) != a.eval_count:
        raise ValueError('Insufficient requested train/validation clips')
    if a.eval_count < 4:
        showcase = validation[:a.eval_count]
        provenance['showcase_note'] = 'Small smoke evaluation; designated four-case showcase replaced by eval subset.'
    elif len(showcase) != 4:
        raise ValueError('Full runs require the designated four showcase clips')
    required = (train if a.mode == 'train' else []) + validation + showcase
    for identifier in set(required):
        bank.check_metadata(identifier)
    provenance.update(train_ids=train, validation_ids=validation, showcase_ids=showcase,
                      train_cohort=a.train_cohort, eval_cohort=a.eval_cohort,
                      evaluation_scope='Official validation reconstruction, future target seen; not blind test or generation.')
    return train, validation, showcase, provenance


def sampled_queries(bank, ids, count, generator, device):
    indexes = torch.randint(20 * 128 * 128, (len(ids), count), generator=generator)
    frames = indexes // (128 * 128)
    yy = (indexes // 128) % 128
    xx = indexes % 128
    coords = torch.stack((2 * xx.float() / 127 - 1, 2 * yy.float() / 127 - 1,
                          2 * frames.float() / 19 - 1), -1).to(device)
    target = np.stack([bank.rgb(identifier)[4:].reshape(-1, 3)[index.numpy()]
                       for identifier, index in zip(ids, indexes)])
    return coords, torch.from_numpy(target).to(device=device, dtype=torch.float32) / 255


def warmstart(model, path, expected_sha=None):
    actual_sha = digest(path)
    if expected_sha and actual_sha != expected_sha:
        raise ValueError('Warmstart checkpoint checksum differs')
    if model.variant == 'trajectory':
        raise ValueError('Trajectory warmstart from old geometry is not an equivalent initialization')
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = ckpt['model']
    filtered = {k: v for k, v in state.items()
                if k.startswith(('visual_adapter.', 'posterior.', 'decoder.'))}
    model.load_state_dict(filtered, strict=True)
    return {'path': str(path.resolve()), 'sha256': actual_sha, 'source_step': ckpt.get('step'),
            'loaded_modules': ['visual_adapter', 'posterior', 'decoder'],
            'optimizer_restored': False, 'purpose': 'Matched AE-only continuation; flow is not loaded.'}


def aggregate(records):
    result = {}
    for key in records[0]['metrics']:
        rows = [r['metrics'][key] for r in records]
        frame_means = []
        for column in zip(*rows):
            valid = [x for x in column if x is not None]
            frame_means.append(float(np.mean(valid)) if valid else None)
        valid = [x for row in rows for x in row if x is not None]
        result[key] = {'mean': float(np.mean(valid)) if valid else None,
                       'by_future_frame': frame_means, 'valid_values': len(valid)}
    return result


def evaluate(model, bank, ids, showcase, out, size, step, device, amp, deadline=None):
    out.mkdir(parents=True, exist_ok=False)
    before_rng = rng_state(device)
    was_training = model.training
    start = time.perf_counter()
    model.eval()
    records, rows = [], []
    ordered = list(dict.fromkeys(ids + showcase))
    try:
        with torch.inference_mode():
            for identifier in ordered:
                if deadline is not None and time.time() >= deadline:
                    raise TimeoutError('Evaluation deadline reached')
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=amp and device.type == 'cuda'):
                    _, mu, _, _ = model.encode_future(*bank.batch([identifier], device))
                pred = model.render_latent(mu, size)[0].float().cpu().numpy()
                if not np.isfinite(pred).all():
                    raise FloatingPointError('Nonfinite reconstruction')
                video = torch.from_numpy(np.array(bank.rgb(identifier), copy=True)).permute(0, 3, 1, 2).float() / 255
                truth = F.interpolate(video[4:], (size, size), mode='bilinear', align_corners=False).numpy()
                previous = F.interpolate(video[3:4], (size, size), mode='bilinear', align_corners=False)[0].numpy()
                record = {'id': identifier, 'source_array_sha256': bank.records[identifier]['array_sha256'],
                          'metrics': time_metrics(pred, truth, previous), 'target_seen': True}
                atomic_json(out / (identifier + '.json'), record)
                if identifier in ids:
                    records.append(record)
                if identifier in showcase:
                    rows.append((identifier, [truth.transpose(0, 2, 3, 1), pred.transpose(0, 2, 3, 1)]))
        rows.sort(key=lambda r: showcase.index(r[0]))
        if rows:
            save_gif(out / 'comparison.gif', rows, ['Ground truth', 'AE FUTURE SEEN'], 12)
        summary = {'status': 'complete', 'step': step, 'variant': model.variant,
                   'evaluation_ids': ids, 'showcase_ids': showcase, 'metric_size': size,
                   'means': aggregate(records), 'records': records,
                   'elapsed_seconds': time.perf_counter() - start,
                   'scope': 'Future-seen reconstruction, not continuation or semantic/physical evaluation.'}
        atomic_json(out / 'summary.json', summary)
        atomic_json(out / 'checksums.json', {p.name: digest(p) for p in sorted(out.iterdir())
                                           if p.is_file() and p.name != 'checksums.json'})
        return summary
    except Exception as exc:
        atomic_json(out / 'error.json', {'error': type(exc).__name__, 'detail': str(exc),
                                       'completed': len(records), 'elapsed_seconds': time.perf_counter() - start})
        raise
    finally:
        model.train(was_training)
        restore_rng(before_rng, device)


def linked_alias(source, target):
    temporary = target.with_name(target.name + f'.{os.getpid()}.tmp')
    os.link(source, temporary)
    os.replace(temporary, target)


def run_train(a):
    if a.deadline_unix is not None and time.time() >= a.deadline_unix:
        raise TimeoutError('Experiment deadline already passed')
    a.out.mkdir(parents=True, exist_ok=True)
    lock = (a.out / '.train.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (a.out / 'protocol.json').exists() and not a.resume:
        raise ValueError('Output already has a protocol; use explicit --resume or a new output')
    if a.resume and a.warmstart:
        raise ValueError('Use either resume or warmstart, not both')
    bank = CacheBank(a.data, a.cache)
    train_ids, eval_ids, showcase, cohorts = pick_cohorts(a, bank)
    device = torch.device(a.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.set_num_threads(a.threads)
    random.seed(a.seed)
    np.random.seed(a.seed % 2**32)
    torch.manual_seed(a.seed)
    model = Codec(a.variant, a.objects, a.latent_dim, a.feature_width,
                  a.subkernels, a.texture_kernels).to(device)
    warm = warmstart(model, a.warmstart, a.warmstart_sha) if a.warmstart else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    data_rng = torch.Generator().manual_seed(a.seed + 101)
    query_rng = torch.Generator().manual_seed(a.seed + 202)
    # Initialization uses different numbers of random values across C vs A/B.
    # Reset stochastic posterior sampling so it is matched independently of that.
    torch.manual_seed(a.seed + 303)
    source_sha = {name: digest(Path(__file__).with_name(name)) for name in
                  ['codec_research.py', 'model.py', 'trajectory_codec.py', 'wan_model.py', 'diagnose_persistence.py']}
    fixed = {'version': 'cached-codec-research-v1', 'model_config': model.config,
             'data_manifest_sha256': bank.manifest_sha, 'cache_protocol_sha256': bank.protocol_sha,
             'cohorts': cohorts, 'seed': a.seed, 'batch': a.batch, 'points': a.points,
             'lr': a.lr, 'weight_decay': a.weight_decay, 'kl_weight': a.kl,
             'amp': a.amp, 'eval_size': a.eval_size, 'source_sha256': source_sha,
             'sampling': 'Uniform train clips with replacement; uniform discrete RGB query points; separate checkpointed CPU generators',
             'checkpoint_selection': 'best.pt minimizes MSE on the fixed quick showcase, not the final full validation cohort',
             'gpu_hours_definition': 'Single-GPU training-loop wall seconds / 3600, excluding evaluation/checkpoint writes; CPU runs report zero GPU-hours'}
    signature = canonical_sha(fixed)
    checkpoint_dir = a.out / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    step = samples = 0
    train_seconds = eval_seconds = prior_wall = 0.
    best = float('inf')
    if a.resume:
        ckpt = torch.load(a.resume, map_location='cpu', weights_only=False)
        if ckpt.get('format') != 'codec_research_v1' or ckpt['protocol_sha256'] != signature:
            raise ValueError('Resume protocol/source/config differs; use a separate explicit experiment')
        model.load_state_dict(ckpt['model'], strict=True)
        optimizer.load_state_dict(ckpt['optimizer'])
        step, samples = ckpt['step'], ckpt['samples_seen']
        train_seconds, eval_seconds, prior_wall = ckpt['train_seconds'], ckpt['eval_seconds'], ckpt['wall_seconds']
        best = ckpt['best_quick_mse']
        warm = ckpt.get('warmstart')
        data_rng.set_state(ckpt['data_rng'])
        query_rng.set_state(ckpt['query_rng'])
        restore_rng(ckpt['rng'], device)
        old = json.loads((a.out / 'protocol.json').read_text()) if (a.out / 'protocol.json').exists() else None
        if old and old['protocol_sha256'] != signature:
            raise ValueError('Output directory belongs to a different protocol')
    elif any(checkpoint_dir.iterdir()):
        raise ValueError('Nonempty checkpoint directory without resume')
    atomic_json(a.out / 'protocol.json', {**fixed, 'protocol_sha256': signature,
                'parameter_counts': model.parameter_counts(), 'warmstart': warm,
                'device': str(device), 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                'torch_version': torch.__version__})
    session_started = time.perf_counter()
    deadline = min(a.deadline_unix or float('inf'), time.time() + max(0., a.max_hours * 3600 - prior_wall))
    stop_file = a.stop_file or a.out / 'REQUEST_STOP'
    state = {'status': 'running', 'step': step, 'pid': os.getpid(), 'started_unix': time.time(),
             'deadline_unix': deadline, 'protocol_sha256': signature}
    atomic_json(a.out / 'status.json', state)

    def checkpoint(alias=None):
        payload = {'format': 'codec_research_v1', 'step': step, 'model_config': model.config,
                   'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                   'rng': rng_state(device), 'data_rng': data_rng.get_state(), 'query_rng': query_rng.get_state(),
                   'samples_seen': samples, 'train_seconds': train_seconds, 'eval_seconds': eval_seconds,
                   'wall_seconds': prior_wall + time.perf_counter() - session_started,
                   'best_quick_mse': best, 'protocol_sha256': signature, 'warmstart': warm,
                   'protocol': fixed,
                   'data_manifest_sha256': bank.manifest_sha,
                   'cache_protocol_sha256': bank.protocol_sha,
                   'source_sha256': source_sha}
        path = checkpoint_dir / f'step_{step:07d}.pt'
        temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
        torch.save(payload, temporary)
        os.replace(temporary, path)
        linked_alias(path, checkpoint_dir / 'latest.pt')
        if alias:
            linked_alias(path, checkpoint_dir / (alias + '.pt'))
        for old in sorted(checkpoint_dir.glob('step_*.pt'))[:-a.keep_checkpoints]:
            old.unlink()
        atomic_json(checkpoint_dir / 'latest.json', {'step': step, 'path': path.name,
                    'sha256': digest(path), 'samples_seen': samples, 'train_seconds': train_seconds})

    final_reason = 'step_limit'
    try:
        model.train()
        while step < a.steps:
            if stop_file.exists():
                final_reason = 'requested_stop'
                break
            if time.time() >= deadline:
                final_reason = 'time_limit'
                break
            sync(device)
            began = time.perf_counter()
            indexes = torch.randint(len(train_ids), (a.batch,), generator=data_rng).tolist()
            ids = [train_ids[i] for i in indexes]
            features = bank.batch(ids, device)
            coords, target = sampled_queries(bank, ids, a.points, query_rng, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=a.amp and device.type == 'cuda'):
                z, _, _, kl = model.encode_future(*features)
            predicted = model.render_queries(z, coords)
            recon = F.mse_loss(predicted, target)
            loss = recon + a.kl * kl
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            sync(device)
            duration = time.perf_counter() - began
            train_seconds += duration
            step += 1
            samples += a.batch
            record = {'step': step, 'samples_seen': samples, 'loss': float(loss.detach()),
                      'recon': float(recon.detach()), 'kl': float(kl.detach()), 'grad_norm': float(norm),
                      'step_seconds': duration, 'train_seconds': train_seconds,
                      'eval_seconds': eval_seconds,
                      'gpu_hours': train_seconds / 3600 if device.type == 'cuda' else 0.,
                      'clips_per_second': a.batch / duration, 'unix_time': time.time()}
            with (a.out / 'metrics.jsonl').open('a') as log:
                log.write(json.dumps(record, allow_nan=False) + '\n')
            if step % a.log_every == 0 or step == 1:
                print(json.dumps(record), flush=True)
            del features, coords, target, z, kl, predicted, recon, loss
            if step % a.eval_every == 0 and time.time() < deadline:
                eval_start = time.perf_counter()
                folder = a.out / 'eval' / f'quick_{step:07d}'
                try:
                    summary = evaluate(model, bank, showcase, showcase, folder, a.eval_size,
                                       step, device, a.amp, deadline)
                finally:
                    sync(device)
                    eval_seconds += time.perf_counter() - eval_start
                score = summary['means']['mse']['mean']
                if score < best:
                    best = score
                    checkpoint('best')
                atomic_json(a.out / 'latest_evaluation.json', {'step': step, 'directory': str(folder),
                            'mse': score, 'kind': 'fixed_quick_showcase', 'eval_seconds': eval_seconds})
            if step % a.save_every == 0:
                checkpoint()
            if step % a.log_every == 0:
                atomic_json(a.out / 'status.json', {**state, 'step': step, 'samples_seen': samples,
                            'train_seconds': train_seconds, 'eval_seconds': eval_seconds, 'updated_unix': time.time()})
        checkpoint('final')
        # Respect the absolute deadline; do not turn a completed training budget
        # into an unbounded final evaluation. The last checkpoint is already safe.
        final_evaluation = None
        if time.time() < deadline and not a.skip_final_eval:
            eval_start = time.perf_counter()
            try:
                final_evaluation = evaluate(model, bank, eval_ids, showcase,
                        a.out / 'eval' / f'final_{step:07d}', a.eval_size, step, device, a.amp, deadline)
            finally:
                sync(device)
                eval_seconds += time.perf_counter() - eval_start
            checkpoint('final')
        atomic_json(a.out / 'status.json', {**state, 'status': 'complete', 'reason': final_reason,
                    'step': step, 'samples_seen': samples, 'train_seconds': train_seconds,
                    'eval_seconds': eval_seconds, 'gpu_hours': train_seconds / 3600 if device.type == 'cuda' else 0.,
                    'final_validation_complete': final_evaluation is not None, 'finished_unix': time.time()})
    except TimeoutError as exc:
        checkpoint('final')
        atomic_json(a.out / 'status.json', {**state, 'status': 'complete', 'reason': 'evaluation_time_limit',
                    'detail': str(exc), 'step': step, 'samples_seen': samples,
                    'train_seconds': train_seconds, 'eval_seconds': eval_seconds,
                    'final_validation_complete': False, 'finished_unix': time.time()})
    except Exception as exc:
        # Do not serialize potentially corrupted/nonfinite optimizer state.
        atomic_json(a.out / 'status.json', {**state, 'status': 'failed', 'step': step,
                    'error': type(exc).__name__ + ': ' + str(exc), 'finished_unix': time.time()})
        raise


def run_eval(a):
    if a.out.exists() and any(a.out.iterdir()):
        raise ValueError('Evaluation output must be new/empty')
    bank = CacheBank(a.data, a.cache)
    _, ids, showcase, cohorts = pick_cohorts(a, bank)
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if ckpt.get('format') != 'codec_research_v1':
        raise ValueError('Requires a codec-research checkpoint')
    bound = ckpt.get('protocol')
    if not isinstance(bound, dict) or canonical_sha(bound) != ckpt['protocol_sha256']:
        raise ValueError('Missing or corrupt checkpoint protocol binding')
    if (bound['model_config'] != ckpt['model_config'] or
            bound['data_manifest_sha256'] != ckpt['data_manifest_sha256'] or
            bound['cache_protocol_sha256'] != ckpt['cache_protocol_sha256'] or
            bound['source_sha256'] != ckpt['source_sha256']):
        raise ValueError('Checkpoint metadata differs from bound training protocol')
    if (ckpt['data_manifest_sha256'] != bank.manifest_sha or
            ckpt['cache_protocol_sha256'] != bank.protocol_sha):
        raise ValueError('Evaluation data/cache provenance differs from checkpoint')
    current_sources = {name: digest(Path(__file__).with_name(name))
                       for name in bound['source_sha256']}
    if current_sources != bound['source_sha256']:
        raise ValueError('Evaluation implementation differs from checkpoint-bound sources')
    device = torch.device(a.device)
    torch.set_num_threads(a.threads)
    model = Codec(**ckpt['model_config']).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    deadline = min(a.deadline_unix or float('inf'), time.time() + a.max_hours * 3600)
    result = evaluate(model, bank, ids, showcase, a.out, a.eval_size, ckpt['step'], device, a.amp, deadline)
    atomic_json(a.out / 'identity.json', {'checkpoint_sha256': digest(a.checkpoint),
                'training_protocol_sha256': ckpt['protocol_sha256'], 'cache_protocol_sha256': bank.protocol_sha,
                'data_manifest_sha256': bank.manifest_sha, 'cohorts': cohorts,
                'source_sha256': digest(Path(__file__)), 'no_optimizer': True,
                'evaluation_complete': result['status'] == 'complete'})
    atomic_json(a.out / 'checksums.json', {p.name: digest(p) for p in sorted(a.out.iterdir())
                                         if p.is_file() and p.name != 'checksums.json'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train', 'eval'])
    for key in ['data', 'cache', 'out']:
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--variant', choices=list(ALIASES) + list(ALIASES.values()), default='xyt_t')
    parser.add_argument('--seed', type=int, default=713)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--max-hours', type=float, default=2.)
    parser.add_argument('--deadline-unix', type=float)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--points', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=.01)
    parser.add_argument('--kl', type=float, default=1e-5)
    parser.add_argument('--train-limit', type=int, default=0)
    parser.add_argument('--eval-count', type=int, default=32)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--eval-size', type=int, default=64)
    parser.add_argument('--keep-checkpoints', type=int, default=3)
    parser.add_argument('--objects', type=int, default=250)
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--feature-width', type=int, default=256)
    parser.add_argument('--subkernels', type=int, default=4)
    parser.add_argument('--texture-kernels', type=int)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--warmstart', type=Path)
    parser.add_argument('--warmstart-sha')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--stop-file', type=Path)
    parser.add_argument('--cohorts', type=Path)
    parser.add_argument('--eval-cohort', choices=['pilot32', 'confirmation64'], default='pilot32')
    parser.add_argument('--train-cohort', choices=['all', 'overfit_train4', 'pilot_train1024'], default='all')
    parser.add_argument('--skip-final-eval', action='store_true')
    a = parser.parse_args()
    if min(a.steps, a.max_hours, a.batch, a.points, a.lr, a.eval_count, a.eval_every,
           a.save_every, a.eval_size, a.keep_checkpoints, a.log_every) <= 0:
        parser.error('Training/evaluation bounds must be positive')
    if a.train_limit < 0 or a.kl < 0 or a.weight_decay < 0:
        parser.error('Limits and regularization must be nonnegative')
    if a.eval_size < 2 or a.points > 20 * 128 * 128:
        parser.error('Unsupported metric size or query count')
    if a.mode == 'eval' and not a.checkpoint:
        parser.error('--checkpoint is required for eval')
    if a.mode == 'train':
        run_train(a)
    else:
        run_eval(a)


if __name__ == '__main__':
    main()

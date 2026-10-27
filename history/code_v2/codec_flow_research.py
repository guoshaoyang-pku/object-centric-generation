"""Conditional object FM with an immutable codec and cached VAE features.

Only Wan transformer weights are loaded; no VAE module is constructed. Prefix
features alone condition generation. Future features supply detached posterior
targets during training, or separately labelled target-seen AE diagnostics.
The default objective is velocity MSE, without independent-noise paired RGB MSE.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from .codec_research import (Codec, CacheBank, digest, canonical_sha, atomic_json,
            rng_state, restore_rng, pick_cohorts, sampled_queries, linked_alias, sync)
    from .wan_model import WanObjectFlow, position_features
    from .wan_token_adapter import WanTokenAdapter, anchor_contract
    from .eval_validation import clip_noise, metrics
    from .diagnose_persistence import time_metrics, save_gif
except ImportError:
    from codec_research import (Codec, CacheBank, digest, canonical_sha, atomic_json,
            rng_state, restore_rng, pick_cohorts, sampled_queries, linked_alias, sync)
    from wan_model import WanObjectFlow, position_features
    from wan_token_adapter import WanTokenAdapter, anchor_contract
    from eval_validation import clip_noise, metrics
    from diagnose_persistence import time_metrics, save_gif


SOURCE_FILES = ['codec_flow_research.py', 'codec_research.py', 'wan_model.py',
                'wan_token_adapter.py', 'model.py', 'trajectory_codec.py', 'trajectory_codec_3dtex.py',
                'diagnose_persistence.py', 'eval_validation.py']
SAMPLING = [(4, 17), (4, 29), (16, 17), (16, 29)]


def state_sha(module):
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode())
        if isinstance(value, torch.Tensor):
            h.update(str(value.dtype).encode())
            h.update(str(tuple(value.shape)).encode())
            h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        else:
            h.update(json.dumps(value, sort_keys=True).encode())
    return h.hexdigest()


def transformer_fingerprint(root):
    files = []
    for path in sorted((root / 'transformer').rglob('*')):
        if path.is_file() and (path.suffix in {'.safetensors', '.bin'} or path.name.endswith('.json')):
            files.append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size,
                          'sha256': digest(path)})
    if not files or not any(r['path'].endswith(('.safetensors', '.bin')) for r in files):
        raise ValueError('Missing frozen transformer weights')
    return {'files': files, 'fingerprint_sha256': canonical_sha(files)}


def load_codec(path, bank, expected_sha=None):
    actual = digest(path)
    if expected_sha and actual != expected_sha:
        raise ValueError('Codec checkpoint SHA mismatch')
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if digest(path) != actual:
        raise ValueError('Codec checkpoint changed while loading')
    bound = ckpt.get('protocol')
    if ckpt.get('format') != 'codec_research_v1' or not isinstance(bound, dict):
        raise ValueError('Requires a protocol-bound codec research checkpoint')
    if canonical_sha(bound) != ckpt['protocol_sha256']:
        raise ValueError('Codec checkpoint protocol is corrupt')
    for key in ['model_config', 'data_manifest_sha256', 'cache_protocol_sha256', 'source_sha256']:
        if ckpt[key] != bound[key]:
            raise ValueError('Codec metadata differs from protocol: ' + key)
    if bound['data_manifest_sha256'] != bank.manifest_sha or bound['cache_protocol_sha256'] != bank.protocol_sha:
        raise ValueError('Codec data/cache provenance differs from this experiment')
    current = {name: digest(Path(__file__).with_name(name)) for name in bound['source_sha256']}
    if current != bound['source_sha256']:
        raise ValueError('Codec source implementation changed after training')
    codec = Codec(**ckpt['model_config'])
    codec.load_state_dict(ckpt['model'], strict=True)
    codec.requires_grad_(False).eval()
    identity = {'checkpoint_sha256': actual, 'checkpoint_step': ckpt['step'],
                'protocol_sha256': ckpt['protocol_sha256'], 'model_config': codec.config,
                'state_sha256': state_sha(codec), 'frozen': True}
    return codec, identity


class ToyFlow(nn.Module):
    """Explicit CPU-only test substitute; never an automatic Wan fallback."""
    def __init__(self, latent_dim, context_width, anchors):
        super().__init__()
        self.input = nn.Linear(latent_dim, 32)
        self.condition = nn.Linear(context_width, 32)
        self.time = nn.Linear(1, 32)
        self.output = nn.Linear(32, latent_dim)
        self.register_buffer('object_anchors', anchors.detach().clone())
        self.contract = anchor_contract(self.object_anchors, .1)

    def forward(self, z, t, context):
        h = self.input(z) + self.condition(context.mean(1))[:, None] + self.time(t[:, None])[:, None]
        return self.output(torch.tanh(h))

    def get_extra_state(self):
        return {'backend': 'cpu-test-only', 'position_mode': 'object_xy', 'anchors': self.contract}

    def set_extra_state(self, value):
        if value != self.get_extra_state() or anchor_contract(self.object_anchors, .1) != self.contract:
            raise ValueError('Toy positional policy mismatch')


class ConditionalObjectFlow(nn.Module):
    def __init__(self, codec, flow):
        super().__init__()
        self.codec, self.flow = codec, flow
        self.codec.requires_grad_(False).eval()
        self.objects, self.latent_dim = codec.objects, codec.latent_dim

    def train(self, mode=True):
        super().train(mode)
        self.codec.eval()
        return self

    @torch.no_grad()
    def prefix_context(self, prefix_features, prefix_coords):
        """Accepts only observed-prefix features and their own coordinates."""
        if prefix_features.ndim != 3 or prefix_features.shape[-1] != 48:
            raise ValueError('Expected prefix features [B,N,48]')
        if prefix_coords.shape != prefix_features.shape[:2] + (3,):
            raise ValueError('Prefix coordinate shape differs')
        adapted = self.codec.visual_adapter(prefix_features)
        return (adapted + .1 * position_features(prefix_coords, adapted.shape[-1]).to(adapted.dtype)).detach()

    @torch.no_grad()
    def target_latent(self, future_features, future_coords, future_local_coords):
        """Training target or AE diagnostic only; never called by sample()."""
        return self.codec.encode_future(future_features, future_coords, future_local_coords)

    def sample(self, context, initial_noise, steps):
        """Future-free API. Each Euler update transforms all objects jointly."""
        if steps < 1 or initial_noise.shape != (context.shape[0], self.objects, self.latent_dim):
            raise ValueError('Invalid noise shape or sampling step count')
        z = initial_noise
        for index in range(steps):
            t = torch.full((z.shape[0],), index / steps, device=z.device, dtype=torch.float32)
            z = z + self.flow(z, t, context).float() / steps
        return z


def retained_flow_keys(flow):
    state = flow.state_dict()
    names = {n for n, p in flow.named_parameters() if p.requires_grad}
    names.update(n for n, _ in flow.named_buffers() if n in state)
    names.update(n for n in state if n.endswith('_extra_state'))
    return names


def flow_state(flow):
    state = flow.state_dict()
    keep = retained_flow_keys(flow)
    return {name: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
            for name, value in state.items() if name in keep}


def load_flow_state(flow, state):
    required = retained_flow_keys(flow)
    if set(state) != required:
        raise ValueError('Flow checkpoint must contain exactly trainables, persistent buffers and extra state')
    result = flow.load_state_dict(state, strict=False)
    if result.unexpected_keys or required.intersection(result.missing_keys):
        raise ValueError('Missing flow trainables or unexpected state')


def legacy_warmstart(flow, path, bank, fingerprint, cfg, expected_sha=None):
    actual = digest(path)
    if expected_sha and actual != expected_sha:
        raise ValueError('Flow warmstart SHA mismatch')
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if digest(path) != actual or ckpt.get('model_state_format') != 'trainable_and_buffers_v1':
        raise ValueError('Requires immutable legacy trainable-and-buffers checkpoint')
    args = ckpt['args']
    checks = {'objects': cfg['objects'], 'latent_dim': cfg['latent_dim'],
              'encoder_width': cfg['feature_width'], 'lora_rank': cfg['lora_rank'],
              'lora_alpha': cfg['lora_alpha']}
    if any(args.get(k) != v for k, v in checks.items()):
        raise ValueError('Legacy flow architecture differs from requested adapters')
    # Reproduce the historical full fingerprint using current transformer hashes
    # and the already bound cache's VAE hashes, without loading/hashing a VAE.
    files = fingerprint['files'] + bank.protocol['vae_fingerprint']['files']
    historical_sha = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if args['pretrained_fingerprint_sha256'] != historical_sha:
        raise ValueError('Legacy flow frozen-base fingerprint differs')
    state = {name[len('flow.'):]: value for name, value in ckpt['model'].items() if name.startswith('flow.')}
    if '_extra_state' in state or 'object_anchors' in state:
        raise ValueError('Expected legacy grid flow, not an already converted adapter')
    state['_extra_state'] = flow.get_extra_state()
    if flow.object_anchors is not None:
        state['object_anchors'] = flow.object_anchors.detach().cpu().clone()
    load_flow_state(flow, state)
    return {'checkpoint_sha256': actual, 'source_step': ckpt['step'], 'optimizer_restored': False,
            'old_position_policy': 'legacy_grid', 'new_position_policy': flow.position_mode,
            'scope': 'Explicit flow-weight initialization only; compatibility with the new frozen codec is not assumed.'}


def position_policy(codec):
    """Do not erase the time identity of the old codec's five temporal slots."""
    if codec.variant in {'xyt_t', 'xyt_xy'}:
        if codec.objects != 250:
            raise ValueError('Legacy xyt flow requires its original 250-token grid')
        return 'legacy_grid'
    if codec.variant.startswith('trajectory'):
        anchors = codec.decoder.anchors
        if anchors.shape != (codec.objects, 2) or len(torch.unique(anchors, dim=0)) != codec.objects:
            raise ValueError('Persistent codec requires distinct explicit xy anchors')
        return 'object_xy'
    raise ValueError('No admitted positional policy for this codec variant')


def make_wan_flow(backbone, codec, lora_rank=8, lora_alpha=8.):
    """Reuse count-independent parameter modules, then remove volume plumbing.

    WanObjectFlow admits only its historical 250-token grid. That argument
    sets a Python grid attribute, not any parameter shape. Persistent codecs
    use the constructor only as a module scaffold; WanTokenAdapter performs
    direct attention over exactly codec.objects tokens and explicit anchors.
    """
    policy = position_policy(codec)
    legacy = WanObjectFlow(backbone, codec.latent_dim, codec.config['feature_width'],
                           250, lora_rank, lora_alpha)
    return WanTokenAdapter.from_legacy(legacy, position_mode=policy,
            object_anchors=codec.decoder.anchors if policy == 'object_xy' else None)


def build_model(a, bank, device):
    codec, codec_identity = load_codec(a.codec_checkpoint, bank, a.codec_sha)
    # Loading a codec constructs modules before replacing their weights. Keep
    # flow initialization independent of how many random values that consumes.
    torch.manual_seed(a.seed + 10001)
    if a.backend == 'toy':
        if device.type != 'cpu' or a.flow_warmstart:
            raise ValueError('Toy backend is CPU smoke only; no legacy warmstart')
        fingerprint = {'backend': 'cpu-test-only', 'fingerprint_sha256': 'no-pretrained-weights'}
        flow = ToyFlow(codec.latent_dim, codec.config['feature_width'], codec.decoder.anchors[..., :2])
        warm = None
    else:
        # Reject unsupported old-grid sizes before loading the large backbone.
        position_policy(codec)
        if not a.pretrained:
            raise ValueError('--pretrained is required for Wan backend')
        fingerprint = transformer_fingerprint(a.pretrained)
        import diffusers
        from diffusers import WanTransformer3DModel
        if diffusers.__version__ != '0.35.2':
            raise ValueError('Direct-token block plumbing is validated only for Diffusers 0.35.2')
        config = json.loads((a.pretrained / 'transformer' / 'config.json').read_text())
        expected = {'in_channels': 48, 'out_channels': 48, 'num_layers': 30,
                    'num_attention_heads': 24, 'attention_head_dim': 128,
                    'ffn_dim': 14336, 'text_dim': 4096, 'patch_size': [1, 2, 2]}
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError('Expected the pinned Wan2.2 TI2V5B 48-channel/30-layer/3072-width architecture')
        fingerprint.update(diffusers_version=diffusers.__version__, architecture=expected)
        backbone = WanTransformer3DModel.from_pretrained(
            a.pretrained, subfolder='transformer', torch_dtype=torch.bfloat16, local_files_only=True)
        flow = make_wan_flow(backbone, codec, a.lora_rank, a.lora_alpha)
        warm = legacy_warmstart(flow, a.flow_warmstart, bank, fingerprint,
                    {**codec.config, 'lora_rank': a.lora_rank, 'lora_alpha': a.lora_alpha},
                    a.flow_warmstart_sha) if a.flow_warmstart else None
    model = ConditionalObjectFlow(codec, flow).to(device)
    identity = {'codec': codec_identity, 'base': fingerprint, 'flow_warmstart': warm,
                'backend': a.backend, 'position_policy': flow.get_extra_state(),
                'token_contract': {'processed_object_tokens': codec.objects,
                    'legacy_constructor_scaffold_tokens': 250 if a.backend == 'wan' else None,
                    'uses_legacy_grid_positions': a.backend == 'wan' and flow.position_mode == 'legacy_grid'},
                'parameter_counts': {'total': sum(p.numel() for p in model.parameters()),
                    'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
                    'codec_frozen': sum(p.numel() for p in codec.parameters()),
                    'lora': sum(p.numel() for n, p in flow.named_parameters() if 'lora_' in n)}}
    return model, identity


def prefix_batch(bank, ids, device):
    arrays = [bank.features(i) for i in ids]
    return tuple(torch.from_numpy(np.stack([a[key] for a in arrays])).to(device)
                 for key in ['prefix', 'prefix_coords'])


def fm_loss(model, context, target, noise, t, coords=None, truth=None, render_weight=0.):
    target = target.detach().float()
    zt = (1 - t[:, None, None]) * noise.float() + t[:, None, None] * target
    velocity = model.flow(zt, t, context).float()
    flow = F.mse_loss(velocity, target - noise.float())
    auxiliary = flow.new_zeros(())
    if render_weight:
        if coords is None or truth is None:
            raise ValueError('Denoising-render supervision requires query labels')
        estimate = zt + (1 - t[:, None, None]) * velocity
        # Frozen decoder parameters; autograd through its latent input remains.
        predicted = model.codec.render_queries(estimate, coords)
        per_clip = (predicted - truth).square().mean((1, 2))
        auxiliary = (t * per_clip).mean()
    return flow + render_weight * auxiliary, flow, auxiliary


def average_records(records):
    result = {}
    for method in records[0]['methods']:
        values = [r['methods'][method] for r in records]
        scalar = {}
        for key in values[0]['scalar']:
            valid = [r['scalar'][key] for r in values if r['scalar'][key] is not None]
            scalar[key] = float(np.mean(valid)) if valid else None
        by_time = {}
        for key in values[0]['time']:
            rows = [r['time'][key] for r in values]
            by_time[key] = [float(np.mean(v)) if v else None for v in
                           ([x for x in col if x is not None] for col in zip(*rows))]
        result[method] = {'scalar': scalar, 'by_future_frame': by_time}
    # Equal arithmetic averaging of the two seeds within each clip; no oracle.
    for steps in [4, 16]:
        names = [f'sample_{steps}_seed{s}' for s in [17, 29]]
        per_clip = [(r['methods'][names[0]]['scalar']['mse'] + r['methods'][names[1]]['scalar']['mse']) / 2
                    for r in records]
        result[f'sample_{steps}_seed_average'] = {
            'mse': float(np.mean(per_clip)), 'clips': len(per_clip),
            'paired_mse_minus_persistence': float(np.mean([x - r['methods']['persistence']['scalar']['mse']
                                                         for x, r in zip(per_clip, records)]))}
    return result


def evaluate(model, bank, ids, showcase, out, size, step, device, amp, deadline):
    out.mkdir(parents=True, exist_ok=False)
    saved_rng = rng_state(device)
    was_training = model.training
    model.eval()
    start = time.perf_counter()
    records, rows = [], []
    try:
        with torch.inference_mode():
            for identifier in list(dict.fromkeys(ids + showcase)):
                if time.time() >= deadline:
                    raise TimeoutError('Flow evaluation deadline reached')
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=amp and device.type == 'cuda'):
                    context = model.prefix_context(*prefix_batch(bank, [identifier], device))
                generated = {}
                # All prefix-only samples are complete before the future
                # posterior is called for the separate AE diagnostic below.
                for steps, seed in SAMPLING:
                    if time.time() >= deadline:
                        raise TimeoutError('Flow sampling deadline reached')
                    noise = clip_noise([identifier], seed, model.objects, model.latent_dim,
                                       device, torch.float32)
                    z = model.sample(context, noise, steps)
                    generated[f'sample_{steps}_seed{seed}'] = model.codec.render_latent(z, size)[0].cpu().numpy()
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=amp and device.type == 'cuda'):
                    _, mu, _, _ = model.target_latent(*bank.batch([identifier], device))
                ae = model.codec.render_latent(mu, size)[0].cpu().numpy()
                video = torch.from_numpy(np.array(bank.rgb(identifier), copy=True)).permute(0, 3, 1, 2).float() / 255
                past = F.interpolate(video[:4], (size, size), mode='bilinear', align_corners=False).numpy()
                truth = F.interpolate(video[4:], (size, size), mode='bilinear', align_corners=False).numpy()
                outputs = {'persistence': np.repeat(past[-1:], 20, 0), 'future_AE': ae, **generated}
                methods = {}
                for name, pred in outputs.items():
                    if not np.isfinite(pred).all():
                        raise FloatingPointError('Nonfinite rendered video: ' + name)
                    methods[name] = {'scalar': metrics(pred, truth), 'time': time_metrics(pred, truth, past[-1])}
                record = {'id': identifier, 'source_array_sha256': bank.records[identifier]['array_sha256'],
                          'methods': methods, 'target_seen_method': 'future_AE only'}
                atomic_json(out / (identifier + '.json'), record)
                if identifier in ids:
                    records.append(record)
                if identifier in showcase:
                    videos = [truth] + list(outputs.values())
                    full = [np.concatenate([past, v]).transpose(0, 2, 3, 1) for v in videos]
                    labels = ['GT', 'Repeat last', 'AE FUTURE SEEN', '4 steps s17', '4 steps s29',
                              '16 steps s17', '16 steps s29']
                    save_gif(out / (identifier + '.gif'), [(identifier, full)], labels, 12, start_frame=0)
                    rows.append((identifier, full))
                atomic_json(out / 'progress.json', {'completed': len(records), 'total': len(ids), 'step': step})
        rows.sort(key=lambda row: showcase.index(row[0]))
        if rows:
            save_gif(out / 'comparison.gif', rows, labels, 12, start_frame=0)
        result = {'status': 'complete', 'step': step, 'evaluation_ids': ids, 'showcase_ids': showcase,
                  'sampling': SAMPLING, 'seed_policy': 'All fixed seeds reported; no best-of selection',
                  'metric_size': size, 'means': average_records(records), 'records': records,
                  'elapsed_seconds': time.perf_counter() - start,
                  'scope': 'Prefix-only generation and separately labelled future-seen AE; RGB metrics do not establish physics or object identity.'}
        atomic_json(out / 'summary.json', result)
        atomic_json(out / 'checksums.json', {p.name: digest(p) for p in sorted(out.iterdir())
                                           if p.is_file() and p.name != 'checksums.json'})
        return result
    except Exception as exc:
        atomic_json(out / 'error.json', {'error': type(exc).__name__, 'detail': str(exc),
                                       'completed': len(records), 'elapsed_seconds': time.perf_counter() - start})
        raise
    finally:
        model.train(was_training)
        restore_rng(saved_rng, device)


def validate_bound_checkpoint(ckpt, fixed=None):
    if ckpt.get('format') != 'codec_flow_research_v1' or not isinstance(ckpt.get('protocol'), dict):
        raise ValueError('Requires protocol-bound codec-flow checkpoint')
    bound = ckpt['protocol']
    if canonical_sha(bound) != ckpt['protocol_sha256']:
        raise ValueError('Flow checkpoint protocol binding is corrupt')
    current = {name: digest(Path(__file__).with_name(name)) for name in SOURCE_FILES}
    if bound['source_sha256'] != current:
        raise ValueError('Flow checkpoint implementation differs from current sources')
    if fixed is not None and canonical_sha(fixed) != ckpt['protocol_sha256']:
        raise ValueError('Resume configuration/data/base/codec differs from checkpoint')
    return bound


def run_train(a):
    startup_start = time.perf_counter()
    if a.deadline_unix is not None and time.time() >= a.deadline_unix:
        raise TimeoutError('Flow experiment deadline already passed')
    a.out.mkdir(parents=True, exist_ok=True)
    lock = (a.out / '.train.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (a.out / 'protocol.json').exists() and not a.resume:
        raise ValueError('Use explicit resume or a fresh output directory')
    if a.resume and a.flow_warmstart:
        raise ValueError('Resume and legacy flow warmstart are mutually exclusive')
    bank = CacheBank(a.data, a.cache)
    train_ids, eval_ids, showcase, cohorts = pick_cohorts(a, bank)
    device = torch.device(a.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.set_num_threads(a.threads)
    random.seed(a.seed)
    np.random.seed(a.seed % 2**32)
    torch.manual_seed(a.seed)
    model, identity = build_model(a, bank, device)
    trainables = [p for p in model.flow.parameters() if p.requires_grad]
    if any(p.requires_grad for p in model.codec.parameters()) or not trainables:
        raise ValueError('Invalid codec/flow gradient routing')
    optimizer = torch.optim.AdamW(trainables, lr=a.lr, weight_decay=a.weight_decay)
    data_rng = torch.Generator().manual_seed(a.seed + 101)
    query_rng = torch.Generator().manual_seed(a.seed + 202)
    noise_rng = torch.Generator().manual_seed(a.seed + 303)
    torch.manual_seed(a.seed + 404)
    source_sha = {name: digest(Path(__file__).with_name(name)) for name in SOURCE_FILES}
    fixed = {'version': 'cached-object-flow-v1', 'identity': identity,
             'data_manifest_sha256': bank.manifest_sha, 'cache_protocol_sha256': bank.protocol_sha,
             'cohorts': cohorts, 'seed': a.seed, 'batch': a.batch, 'lr': a.lr,
             'weight_decay': a.weight_decay, 'lora_rank': a.lora_rank, 'lora_alpha': a.lora_alpha,
             'denoise_render_weight': a.denoise_render_weight, 'points': a.points,
             'amp': a.amp, 'eval_size': a.eval_size, 'sampling': SAMPLING,
             'source_sha256': source_sha,
             'objective': 'q(z|future) sampled and detached; z_t=(1-t)epsilon+t*z; velocity MSE target z-epsilon; optional t-weighted corresponding denoised-render auxiliary',
             'checkpoint_selection': 'best quick 16-step RGB MSE averaged over both fixed seeds, on fixed showcase; final pilot evaluated separately',
             'gpu_hours_definition': 'Single-GPU training-loop wall time; evaluation and initialization are reported separately; no H200/A100 conversion'}
    signature = canonical_sha(fixed)
    step = samples = 0
    train_seconds = eval_seconds = prior_wall = previous_setup = 0.
    best = float('inf')
    if a.resume:
        ckpt = torch.load(a.resume, map_location='cpu', weights_only=False)
        # A resume inherits its already documented legacy warmstart provenance.
        fixed['identity']['flow_warmstart'] = ckpt.get('protocol', {}).get('identity', {}).get('flow_warmstart')
        signature = canonical_sha(fixed)
        validate_bound_checkpoint(ckpt, fixed)
        load_flow_state(model.flow, ckpt['flow'])
        optimizer.load_state_dict(ckpt['optimizer'])
        step, samples = ckpt['step'], ckpt['samples_seen']
        train_seconds, eval_seconds, prior_wall = ckpt['train_seconds'], ckpt['eval_seconds'], ckpt['wall_seconds']
        previous_setup = ckpt['setup_seconds']
        best = ckpt['best_quick_mse']
        for generator, name in [(data_rng, 'data_rng'), (query_rng, 'query_rng'), (noise_rng, 'noise_rng')]:
            generator.set_state(ckpt[name])
        restore_rng(ckpt['rng'], device)
    atomic_json(a.out / 'protocol.json', {**fixed, 'protocol_sha256': signature,
                'torch_version': torch.__version__, 'device': str(device),
                'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})
    folder = a.out / 'checkpoints'
    folder.mkdir(exist_ok=True)
    if any(folder.iterdir()) and not a.resume:
        raise ValueError('Nonempty checkpoint directory without resume')
    session_started = time.perf_counter()
    setup_seconds = previous_setup + session_started - startup_start
    deadline = min(a.deadline_unix or float('inf'), time.time() + max(0., a.max_hours * 3600 - prior_wall - setup_seconds))
    stop_file = a.stop_file or a.out / 'REQUEST_STOP'
    status = {'status': 'running', 'pid': os.getpid(), 'step': step,
              'started_unix': time.time(), 'deadline_unix': deadline, 'protocol_sha256': signature,
              'setup_seconds': setup_seconds}
    atomic_json(a.out / 'status.json', status)

    def checkpoint(alias=None):
        payload = {'format': 'codec_flow_research_v1', 'step': step,
                   'flow': flow_state(model.flow), 'optimizer': optimizer.state_dict(),
                   'protocol': fixed, 'protocol_sha256': signature,
                   'rng': rng_state(device), 'data_rng': data_rng.get_state(),
                   'query_rng': query_rng.get_state(), 'noise_rng': noise_rng.get_state(),
                   'samples_seen': samples, 'train_seconds': train_seconds, 'eval_seconds': eval_seconds,
                   'setup_seconds': setup_seconds,
                   'wall_seconds': prior_wall + time.perf_counter() - session_started, 'best_quick_mse': best}
        path = folder / f'step_{step:07d}.pt'
        tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
        torch.save(payload, tmp)
        os.replace(tmp, path)
        linked_alias(path, folder / 'latest.pt')
        if alias:
            linked_alias(path, folder / (alias + '.pt'))
        for old in sorted(folder.glob('step_*.pt'))[:-a.keep_checkpoints]:
            old.unlink()
        atomic_json(folder / 'latest.json', {'step': step, 'path': path.name, 'sha256': digest(path),
                    'samples_seen': samples, 'train_seconds': train_seconds})

    reason = 'step_limit'
    try:
        model.train()
        while step < a.steps:
            if stop_file.exists():
                reason = 'requested_stop'
                break
            if time.time() >= deadline:
                reason = 'time_limit'
                break
            sync(device)
            start = time.perf_counter()
            ids = [train_ids[i] for i in torch.randint(len(train_ids), (a.batch,), generator=data_rng).tolist()]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=a.amp and device.type == 'cuda'):
                context = model.prefix_context(*prefix_batch(bank, ids, device))
                target, _, _, _ = model.target_latent(*bank.batch(ids, device))
            noise = torch.randn(a.batch, model.objects, model.latent_dim, generator=noise_rng).to(device)
            t = torch.rand(a.batch, generator=noise_rng).to(device)
            coords = truth = None
            if a.denoise_render_weight:
                coords, truth = sampled_queries(bank, ids, a.points, query_rng, device)
            optimizer.zero_grad(set_to_none=True)
            loss, flow, auxiliary = fm_loss(model, context, target, noise, t, coords, truth, a.denoise_render_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite flow loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainables, 1., error_if_nonfinite=True)
            optimizer.step()
            sync(device)
            duration = time.perf_counter() - start
            step += 1
            samples += a.batch
            train_seconds += duration
            record = {'step': step, 'samples_seen': samples, 'loss': float(loss.detach()),
                      'flow': float(flow.detach()), 'denoise_render': float(auxiliary.detach()),
                      'grad_norm': float(norm), 'step_seconds': duration, 'train_seconds': train_seconds,
                      'eval_seconds': eval_seconds, 'gpu_hours': train_seconds / 3600 if device.type == 'cuda' else 0.,
                      'eval_gpu_hours': eval_seconds / 3600 if device.type == 'cuda' else 0.,
                      'clips_per_second': a.batch / duration, 'unix_time': time.time()}
            with (a.out / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            if step == 1 or step % a.log_every == 0:
                print(json.dumps(record), flush=True)
            del context, target, noise, t, coords, truth, loss, flow, auxiliary
            if step % a.eval_every == 0 and time.time() < deadline:
                began = time.perf_counter()
                try:
                    result = evaluate(model, bank, showcase, showcase,
                            a.out / 'eval' / f'quick_{step:07d}', a.eval_size, step, device, a.amp, deadline)
                finally:
                    sync(device)
                    eval_seconds += time.perf_counter() - began
                score = result['means']['sample_16_seed_average']['mse']
                if score < best:
                    best = score
                    checkpoint('best')
                atomic_json(a.out / 'latest_evaluation.json', {'step': step, 'mse': score,
                            'kind': 'fixed_showcase_16step_two_seed_mean', 'eval_seconds': eval_seconds})
            if step % a.save_every == 0:
                checkpoint()
            if step % a.log_every == 0:
                atomic_json(a.out / 'status.json', {**status, 'step': step, 'samples_seen': samples,
                            'train_seconds': train_seconds, 'eval_seconds': eval_seconds, 'updated_unix': time.time()})
        checkpoint('final')
        final_done = False
        if time.time() < deadline and not a.skip_final_eval:
            began = time.perf_counter()
            try:
                evaluate(model, bank, eval_ids, showcase, a.out / 'eval' / f'final_{step:07d}',
                         a.eval_size, step, device, a.amp, deadline)
                final_done = True
            finally:
                sync(device)
                eval_seconds += time.perf_counter() - began
            checkpoint('final')
        if state_sha(model.codec) != identity['codec']['state_sha256']:
            raise RuntimeError('Frozen codec changed during flow training')
        atomic_json(a.out / 'status.json', {**status, 'status': 'complete', 'reason': reason,
                    'step': step, 'samples_seen': samples, 'train_seconds': train_seconds,
                    'eval_seconds': eval_seconds, 'gpu_hours': train_seconds / 3600 if device.type == 'cuda' else 0.,
                    'eval_gpu_hours': eval_seconds / 3600 if device.type == 'cuda' else 0.,
                    'allocated_gpu_hours': (prior_wall + time.perf_counter() - session_started + setup_seconds) / 3600 if device.type == 'cuda' else 0.,
                    'codec_unchanged': True, 'final_validation_complete': final_done, 'finished_unix': time.time()})
    except TimeoutError as exc:
        checkpoint('final')
        atomic_json(a.out / 'status.json', {**status, 'status': 'complete', 'reason': 'evaluation_time_limit',
                    'detail': str(exc), 'step': step, 'samples_seen': samples,
                    'train_seconds': train_seconds, 'eval_seconds': eval_seconds,
                    'final_validation_complete': False, 'finished_unix': time.time()})
    except Exception as exc:
        atomic_json(a.out / 'status.json', {**status, 'status': 'failed', 'step': step,
                    'error': type(exc).__name__ + ': ' + str(exc), 'finished_unix': time.time()})
        raise


def run_eval(a):
    if a.out.exists() and any(a.out.iterdir()):
        raise ValueError('Evaluation output must be fresh')
    checkpoint_sha = digest(a.checkpoint)
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if digest(a.checkpoint) != checkpoint_sha:
        raise ValueError('Checkpoint changed while loading')
    bound = validate_bound_checkpoint(ckpt)
    bank = CacheBank(a.data, a.cache)
    if bound['data_manifest_sha256'] != bank.manifest_sha or bound['cache_protocol_sha256'] != bank.protocol_sha:
        raise ValueError('Flow evaluation data/cache differs from training')
    _, ids, showcase, cohorts = pick_cohorts(a, bank)
    device = torch.device(a.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.set_num_threads(a.threads)
    a.backend = bound['identity']['backend']
    a.lora_rank, a.lora_alpha = bound['lora_rank'], bound['lora_alpha']
    a.codec_sha = bound['identity']['codec']['checkpoint_sha256']
    if a.flow_warmstart:
        raise ValueError('Evaluation restores its checkpoint; no additional warmstart')
    model, identity = build_model(a, bank, device)
    for key in ['codec', 'base', 'backend', 'position_policy', 'token_contract', 'parameter_counts']:
        if identity[key] != bound['identity'][key]:
            raise ValueError('Evaluation model identity differs: ' + key)
    load_flow_state(model.flow, ckpt['flow'])
    deadline = min(a.deadline_unix or float('inf'), time.time() + a.max_hours * 3600)
    evaluate(model, bank, ids, showcase, a.out, a.eval_size, ckpt['step'], device, a.amp, deadline)
    atomic_json(a.out / 'identity.json', {'checkpoint_sha256': checkpoint_sha,
                'training_protocol_sha256': ckpt['protocol_sha256'], 'model_identity': identity,
                'cohorts': cohorts, 'no_optimizer': True})
    atomic_json(a.out / 'checksums.json', {p.name: digest(p) for p in sorted(a.out.iterdir())
                                         if p.is_file() and p.name != 'checksums.json'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['train', 'eval'])
    for name in ['data', 'cache', 'out', 'codec-checkpoint']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--pretrained', type=Path)
    p.add_argument('--codec-sha')
    p.add_argument('--flow-warmstart', type=Path)
    p.add_argument('--flow-warmstart-sha')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--resume', type=Path)
    p.add_argument('--backend', choices=['wan', 'toy'], default='wan')
    p.add_argument('--seed', type=int, default=713)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--max-hours', type=float, default=2.)
    p.add_argument('--deadline-unix', type=float)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--lora-rank', type=int, default=8)
    p.add_argument('--lora-alpha', type=float, default=8.)
    p.add_argument('--denoise-render-weight', type=float, default=0.)
    p.add_argument('--points', type=int, default=1024)
    p.add_argument('--train-limit', type=int, default=0)
    p.add_argument('--eval-count', type=int, default=32)
    p.add_argument('--eval-size', type=int, default=64)
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--save-every', type=int, default=250)
    p.add_argument('--keep-checkpoints', type=int, default=3)
    p.add_argument('--log-every', type=int, default=25)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--cohorts', type=Path)
    p.add_argument('--train-cohort', choices=['all', 'overfit_train4', 'pilot_train1024'], default='all')
    p.add_argument('--eval-cohort', choices=['pilot32', 'confirmation64'], default='pilot32')
    p.add_argument('--stop-file', type=Path)
    p.add_argument('--skip-final-eval', action='store_true')
    a = p.parse_args()
    if min(a.batch, a.steps, a.max_hours, a.lr, a.points, a.eval_count, a.eval_size,
           a.eval_every, a.save_every, a.keep_checkpoints, a.log_every) <= 0:
        p.error('Bounds must be positive')
    if a.train_limit < 0 or a.weight_decay < 0 or a.denoise_render_weight < 0:
        p.error('Limits/weights must be nonnegative')
    if a.mode == 'eval' and not a.checkpoint:
        p.error('--checkpoint is required for eval')
    run_train(a) if a.mode == 'train' else run_eval(a)


if __name__ == '__main__':
    main()

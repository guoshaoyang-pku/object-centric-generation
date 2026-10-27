"""Regular spacetime VAE-patch FM baseline; not raw RGB diffusion.

Same frozen prefix processing and ObjectFlow backbone family as the object
pilot. Cached future posterior-mode targets pack to 96x192; no VAE decode is
performed by the training loss. Evaluation uses the frozen original VAE.
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

from codec_research import (CacheBank, digest, canonical_sha, atomic_json, rng_state, restore_rng,
                            pick_cohorts, linked_alias, sync)
from codec_flow_research import (load_codec, state_sha, flow_state, load_flow_state, prefix_batch,
                                 average_records, memory_peaks, training_budget)
from native_patch_codec import (TARGET_CONTRACT, NativePatchFlow, pack_features, unpack_tokens,
                                check_future_grid, load_vae)
from patch_video_metrics import annotation_binding, load_fg_mask, add_fg_bg
from eval_validation import clip_noise as object_clip_noise, metrics
from diagnose_persistence import time_metrics, save_gif

SOURCE_FILES = ['patch_flow_research.py', 'native_patch_codec.py', 'patch_video_metrics.py',
                'codec_flow_research.py', 'codec_research.py', 'model.py', 'small_object_flow.py',
                'trajectory_codec.py', 'trajectory_codec_3dtex.py', 'wan_model.py',
                'wan_token_adapter.py', 'diagnose_persistence.py', 'eval_validation.py',
                'cache_vae_features.py', 'evaluate_vae_floor.py']
SAMPLING = [(4, 17), (4, 29), (16, 17), (16, 29)]


def clip_noise(ids, master, objects, latent_dim, device, dtype):
    if (objects, latent_dim) != (96, 192):
        raise ValueError('Native patch noise shape differs')
    shared = object_clip_noise(ids, master, 250, 128, device, dtype)
    return shared.flatten(1)[:, :18432].reshape(-1, 96, 192)


def frozen_state_sha(module):
    """Streaming state digest also supporting scalar VAE buffers."""
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode())
        if isinstance(value, torch.Tensor):
            h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
            h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            h.update(json.dumps(value, sort_keys=True).encode())
    return h.hexdigest()


class NativePatchModel(nn.Module):
    def __init__(self, codec, flow, decoder, annotations=None, wrong_prefix_map=None):
        super().__init__()
        self.codec, self.flow, self.decoder = codec, flow, decoder
        self.objects, self.latent_dim = 96, 192
        self.annotations = annotations
        self.wrong_prefix_map = wrong_prefix_map or {}
        self.codec.requires_grad_(False).eval()
        self.decoder.requires_grad_(False).eval()
        self._codec_sha = state_sha(codec)
        self._decoder_sha = frozen_state_sha(decoder)

    def train(self, mode=True):
        super().train(mode)
        self.codec.eval(); self.decoder.eval()
        return self

    def assert_frozen(self):
        if state_sha(self.codec) != self._codec_sha or frozen_state_sha(self.decoder) != self._decoder_sha:
            raise RuntimeError('Frozen prefix codec or VAE changed')
        if any(p.requires_grad or p.grad is not None for m in [self.codec, self.decoder] for p in m.parameters()):
            raise RuntimeError('Frozen module received gradients')

    @torch.no_grad()
    def prefix_context(self, features, coords):
        if features.ndim != 3 or features.shape[1:] != (128, 48) or coords.shape != features.shape[:2] + (3,):
            raise ValueError('Prefix must contain only cached four-frame 2x8x8 tokens')
        return self.codec.adapt_features(features, coords).detach()

    @torch.no_grad()
    def target_latent(self, features, coords, localcoords):
        check_future_grid(coords, localcoords)
        packed = pack_features(features.float()).detach()
        return packed, None, None, packed.new_zeros(())

    def sample(self, context, initial_noise, steps):
        if steps < 1 or initial_noise.shape != (context.shape[0], 96, 192):
            raise ValueError('Invalid native-patch sampling shape or steps')
        z = initial_noise
        for index in range(steps):
            t = torch.full((z.shape[0],), index / steps, device=z.device, dtype=torch.float32)
            z = z + self.flow(z, t, context).float() / steps
        return z


def build_model(a, bank, device):
    if a.flow_warmstart or a.denoise_render_weight:
        raise ValueError('Patch baseline admits fresh native flow and pure FM only')
    if not a.codec_sha:
        raise ValueError('Bind the exact frozen prefix-codec checkpoint SHA')
    codec, codec_identity = load_codec(a.codec_checkpoint, bank, a.codec_sha)
    cfg = codec.config
    if codec_identity['checkpoint_step'] != 0 or cfg['variant'] != 'trajectory' or cfg['encoder_depth'] != 0:
        raise ValueError('Prefix must be the designated v3 C2 baseline initial checkpoint')
    if a.decoder_backend == 'wan_vae' and (cfg['objects'], cfg['latent_dim'], cfg['feature_width']) != (250, 128, 256):
        raise ValueError('Production prefix architecture differs from the matched object run')
    decoder, vae_identity = load_vae(a.pretrained, bank, device, a.decoder_backend)
    mapping = {}
    if getattr(a, 'cohorts', None):
        document = json.loads(a.cohorts.read_text())
        if document['manifest_sha256'] != bank.manifest_sha:
            raise ValueError('Wrong-prefix cohort manifest differs')
        for group in ['pilot32', 'showcase4']:
            members = sorted(r['id'] for r in document[group])
            if len(members) < 2 or set(members).intersection(mapping):
                raise ValueError('Wrong-prefix cohorts must be disjoint and nontrivial')
            for r in document[group]:
                if bank.records[r['id']]['array_sha256'] != r['sha256'] or bank.records[r['id']]['split'] != 'validation':
                    raise ValueError('Wrong-prefix cohort source mismatch')
            mapping.update({v: members[(i + 1) % len(members)] for i, v in enumerate(members)})
    elif a.decoder_backend == 'cpu_mock':
        members = bank.ids('validation')
        mapping = {v: members[(i + 1) % len(members)] for i, v in enumerate(members)}
    else:
        raise ValueError('Production wrong-prefix control requires the fixed pilot32/showcase4 cohorts')
    torch.manual_seed(a.seed + 10001)
    flow = NativePatchFlow(cfg['feature_width'], a.small_width, a.small_depth, a.small_heads)
    model = NativePatchModel(codec, flow, decoder, a.annotations, mapping).to(device)
    base = {'backend': 'regular_vae_patch', 'initialization': 'fresh ObjectFlow; no backbone pretraining',
            'architecture': flow.model_config, 'target_contract': TARGET_CONTRACT}
    base['fingerprint_sha256'] = canonical_sha(base)
    identity = {'codec': codec_identity, 'base': base, 'flow_warmstart': None,
                'backend': 'regular_vae_patch', 'position_policy': flow.get_extra_state(),
                'flow_model_config': flow.model_config,
                'token_contract': TARGET_CONTRACT,
                'vae': {**vae_identity, 'state_sha256': model._decoder_sha},
                'annotation_binding': annotation_binding(a.annotations, bank),
                'evaluation_contract': {'wrong_prefix_map': mapping,
                    'policy': 'pilot32 sorted cyclic shift1; disjoint showcase4 sorted cyclic shift1; target-ID noise'},
                'parameter_counts': {'total': sum(p.numel() for p in model.parameters()),
                    'trainable': sum(p.numel() for p in flow.parameters()),
                    'codec_frozen': sum(p.numel() for p in codec.parameters()),
                    'vae_frozen': sum(p.numel() for p in decoder.parameters()), 'lora': 0}}
    return model, identity


def fm_loss(model, context, target, noise, t, coords=None, truth=None, render_weight=0.):
    if render_weight or coords is not None or truth is not None:
        raise ValueError('VAE patch decoder does not enter the FM training loss')
    target = target.detach().float()
    zt = (1 - t[:, None, None]) * noise.float() + t[:, None, None] * target
    velocity = model.flow(zt, t, context).float()
    loss = F.mse_loss(velocity, target - noise.float())
    return loss, loss, loss.new_zeros(())


def evaluate(model, bank, ids, showcase, out, size, step, device, amp, deadline):
    out.mkdir(parents=True, exist_ok=False)
    saved_rng, was_training = rng_state(device), model.training
    model.eval(); start = time.perf_counter()
    records, rows = [], []
    if not set(ids + showcase).issubset(model.wrong_prefix_map):
        raise ValueError('Evaluation IDs lack a preregistered wrong-prefix mapping')
    wrong = model.wrong_prefix_map
    try:
        with torch.inference_mode():
            for identifier in list(dict.fromkeys(ids + showcase)):
                if time.time() >= deadline: raise TimeoutError('Patch evaluation deadline')
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == 'cuda'):
                    context = model.prefix_context(*prefix_batch(bank, [identifier], device))
                    wrong_context = model.prefix_context(*prefix_batch(bank, [wrong[identifier]], device))
                generated = {}
                for steps, seed in SAMPLING:
                    if time.time() >= deadline: raise TimeoutError('Patch sampling deadline')
                    noise = clip_noise([identifier], seed, 96, 192, device, torch.float32)
                    z = model.sample(context, noise, steps)
                    generated[f'sample_{steps}_seed{seed}'] = model.decoder.decode_tokens(z, size)[0].cpu().numpy()
                for seed in [17, 29]:
                    if time.time() >= deadline: raise TimeoutError('Wrong-prefix sampling deadline')
                    noise = clip_noise([identifier], seed, 96, 192, device, torch.float32)
                    z = model.sample(wrong_context, noise, 16)
                    generated[f'wrong_prefix_16_seed{seed}'] = model.decoder.decode_tokens(z, size)[0].cpu().numpy()
                # All future-free generation is finished before target decoding or RGB metrics.
                target, _, _, _ = model.target_latent(*bank.batch([identifier], device))
                vae_reference = model.decoder.decode_tokens(target, size)[0].cpu().numpy()
                video = torch.from_numpy(np.array(bank.rgb(identifier), copy=True)).permute(0, 3, 1, 2).float() / 255
                other = torch.from_numpy(np.array(bank.rgb(wrong[identifier]), copy=True)).permute(0, 3, 1, 2).float() / 255
                past = F.interpolate(video[:4], (size, size), mode='bilinear', align_corners=False).numpy()
                other_past = F.interpolate(other[:4], (size, size), mode='bilinear', align_corners=False).numpy()
                truth = F.interpolate(video[4:], (size, size), mode='bilinear', align_corners=False).numpy()
                fg = load_fg_mask(model.annotations, bank, identifier, size, required=identifier in showcase and model.annotations is not None)
                outputs = {'persistence': np.repeat(past[-1:], 20, 0), 'future_VAE': vae_reference, **generated}
                methods = {}
                for name, pred in outputs.items():
                    if not np.isfinite(pred).all(): raise FloatingPointError('Nonfinite VAE-decoded video')
                    foreground = add_fg_bg(pred, truth, fg)
                    methods[name] = {'scalar': {**metrics(pred, truth), **foreground['scalar']},
                                     'time': {**time_metrics(pred, truth, past[-1]), **foreground['time']}}
                item = {'id': identifier, 'source_array_sha256': bank.records[identifier]['array_sha256'],
                        'wrong_prefix_id': wrong[identifier], 'wrong_prefix_source_sha256': bank.records[wrong[identifier]]['array_sha256'],
                        'methods': methods, 'target_seen_method': 'future_VAE only', 'foreground_annotation_available': fg is not None}
                atomic_json(out / (identifier + '.json'), item)
                if identifier in ids: records.append(item)
                if identifier in showcase:
                    videos = [np.concatenate([past, truth]).transpose(0, 2, 3, 1)]
                    for name, pred in outputs.items():
                        prefix = other_past if name.startswith('wrong_prefix') else past
                        videos.append(np.concatenate([prefix, pred]).transpose(0, 2, 3, 1))
                    labels = ['GT', 'Repeat last', 'VAE FUTURE SEEN', '4 steps s17', '4 steps s29',
                              '16 steps s17', '16 steps s29', 'Wrong16 s17', 'Wrong16 s29']
                    save_gif(out / (identifier + '.gif'), [(identifier, videos)], labels, 12, start_frame=0)
                    rows.append((identifier, videos))
                atomic_json(out / 'progress.json', {'completed': len(records), 'total': len(ids), 'step': step})
        rows.sort(key=lambda row: showcase.index(row[0]))
        if rows: save_gif(out / 'comparison.gif', rows, labels, 12, start_frame=0)
        result = {'status': 'complete', 'step': step, 'evaluation_ids': ids, 'showcase_ids': showcase,
                  'sampling': SAMPLING, 'seed_policy': 'All seeds; no best-of selection', 'metric_size': size,
                  'wrong_prefix_policy': 'pilot32 sorted shift1; separate showcase4 sorted shift1; same target-ID noise',
                  'foreground_annotation_clips': sum(r['foreground_annotation_available'] for r in records),
                  'means': average_records(records), 'records': records, 'elapsed_seconds': time.perf_counter() - start,
                  'scope': 'Regular spacetime VAE-patch generation; target-seen VAE reference; RGB/motion/FG-BG are not physical proof.'}
        atomic_json(out / 'summary.json', result)
        atomic_json(out / 'checksums.json', {p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != 'checksums.json'})
        return result
    finally:
        model.train(was_training); restore_rng(saved_rng, device)


def validate_bound_checkpoint(ckpt, fixed=None):
    if ckpt.get('format') != 'native_patch_flow_v5' or not isinstance(ckpt.get('protocol'), dict):
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
        torch.cuda.reset_peak_memory_stats(device)
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
    fixed = {'version': 'native-patch-flow-v5', 'identity': identity,
             'data_manifest_sha256': bank.manifest_sha, 'cache_protocol_sha256': bank.protocol_sha,
             'cohorts': cohorts, 'seed': a.seed, 'batch': a.batch, 'lr': a.lr,
             'weight_decay': a.weight_decay, 'lora_rank': a.lora_rank, 'lora_alpha': a.lora_alpha,
             'denoise_render_weight': a.denoise_render_weight, 'points': a.points,
             'amp': a.amp, 'eval_size': a.eval_size, 'sampling': SAMPLING,
             'training_gpu_hour_budget': a.train_hours,
             'evaluation_schedule': {'every_updates': a.eval_every, 'save_every_updates': a.save_every},
             'noise_schedule': 'Draw common object-space250x128 epsilon; use first18432 values reshaped96x192; then draw t. Same seed/draw consumption as matched object run.',
             'source_sha256': source_sha,
             'objective': 'Fixed cached VAE posterior-mode target packed96x192; z_t=(1-t)epsilon+t*z; velocity MSE target z-epsilon; decoder absent from training graph',
             'checkpoint_selection': 'best quick16-step two-seed RGB metric is preview only; use final900-training-second endpoint for budget comparison',
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
        payload = {'format': 'native_patch_flow_v5', 'step': step,
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
            if a.train_hours is not None and train_seconds >= a.train_hours * 3600:
                reason = 'training_compute_limit'
                break
            sync(device)
            start = time.perf_counter()
            ids = [train_ids[i] for i in torch.randint(len(train_ids), (a.batch,), generator=data_rng).tolist()]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=a.amp and device.type == 'cuda'):
                context = model.prefix_context(*prefix_batch(bank, ids, device))
                target, _, _, _ = model.target_latent(*bank.batch(ids, device))
            shared_noise = torch.randn(a.batch, 250, 128, generator=noise_rng)
            noise = shared_noise.flatten(1)[:, :18432].reshape(a.batch, 96, 192).to(device)
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
        model.assert_frozen()
        if state_sha(model.codec) != identity['codec']['state_sha256']:
            raise RuntimeError('Frozen codec changed during flow training')
        atomic_json(a.out / 'status.json', {**status, 'status': 'complete', 'reason': reason,
                    'step': step, 'samples_seen': samples, 'train_seconds': train_seconds,
                    'eval_seconds': eval_seconds, 'gpu_hours': train_seconds / 3600 if device.type == 'cuda' else 0.,
                    'eval_gpu_hours': eval_seconds / 3600 if device.type == 'cuda' else 0.,
                    'allocated_gpu_hours': (prior_wall + time.perf_counter() - session_started + setup_seconds) / 3600 if device.type == 'cuda' else 0.,
                    **memory_peaks(device),
                    **training_budget(a, train_seconds),
                    'codec_unchanged': True, 'final_validation_complete': final_done, 'finished_unix': time.time()})
    except TimeoutError as exc:
        checkpoint('final')
        atomic_json(a.out / 'status.json', {**status, 'status': 'complete', 'reason': 'evaluation_time_limit',
                    'detail': str(exc), 'step': step, 'samples_seen': samples,
                    'train_seconds': train_seconds, 'eval_seconds': eval_seconds,
                    **memory_peaks(device),
                    **training_budget(a, train_seconds),
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
    a.backend = 'regular_vae_patch'
    a.decoder_backend = bound['identity']['vae']['backend']
    if a.decoder_backend == 'cpu_mock_fixture_only': a.decoder_backend = 'cpu_mock'
    cfg = bound['identity']['flow_model_config']
    a.small_width, a.small_depth, a.small_heads = cfg['width'], cfg['depth'], cfg['heads']
    if a.backend == 'small':
        cfg = bound['identity']['flow_model_config']
        a.small_width, a.small_depth, a.small_heads = cfg['width'], cfg['depth'], cfg['heads']
    a.lora_rank, a.lora_alpha = bound['lora_rank'], bound['lora_alpha']
    a.codec_sha = bound['identity']['codec']['checkpoint_sha256']
    if a.flow_warmstart:
        raise ValueError('Evaluation restores its checkpoint; no additional warmstart')
    model, identity = build_model(a, bank, device)
    for key in ['codec', 'base', 'backend', 'position_policy', 'flow_model_config', 'token_contract', 'parameter_counts', 'vae', 'annotation_binding', 'evaluation_contract']:
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
    p.add_argument('--codec-sha', required=True)
    p.add_argument('--pretrained', type=Path)
    p.add_argument('--annotations', type=Path)
    p.add_argument('--decoder-backend', choices=['wan_vae', 'cpu_mock'], default='wan_vae')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--resume', type=Path)
    p.add_argument('--small-width', type=int, default=384)
    p.add_argument('--small-depth', type=int, default=8)
    p.add_argument('--small-heads', type=int, default=6)
    p.add_argument('--seed', type=int, default=713)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--steps', type=int, default=100000)
    p.add_argument('--max-hours', type=float, default=.5)
    p.add_argument('--train-hours', type=float, default=.25)
    p.add_argument('--deadline-unix', type=float)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--train-limit', type=int, default=0)
    p.add_argument('--eval-count', type=int, default=32)
    p.add_argument('--eval-size', type=int, default=64)
    p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--save-every', type=int, default=250)
    p.add_argument('--keep-checkpoints', type=int, default=12)
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
    # Fixed pure-FM contract. No object/warmstart/RGB-auxiliary compatibility path.
    a.backend = 'regular_vae_patch'
    a.flow_warmstart = a.flow_warmstart_sha = None
    a.lora_rank, a.lora_alpha, a.denoise_render_weight, a.points = 0, 0., 0., 0
    if min(a.batch, a.steps, a.max_hours, a.train_hours, a.lr, a.eval_count, a.eval_size,
           a.eval_every, a.save_every, a.keep_checkpoints, a.log_every,
           a.small_width, a.small_depth, a.small_heads) <= 0:
        p.error('Bounds must be positive')
    if a.train_limit < 0 or a.weight_decay < 0:
        p.error('Limits must be nonnegative')
    if a.mode == 'eval' and not a.checkpoint:
        p.error('--checkpoint is required for eval')
    run_train(a) if a.mode == 'train' else run_eval(a)


if __name__ == '__main__':
    main()

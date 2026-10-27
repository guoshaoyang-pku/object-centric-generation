"""Raw RGB flow pilot at64px with fixed train-only normalization.

Observed prefix uses the same frozen cached-VAE adapter as object pilots.
The prediction target is raw future RGB1280x192, with no learned target decoder.
This supplementary common64 probe is not part of the registered native128 trio.
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
from raw_pixel_codec import (TARGET_CONTRACT, RawPixelFlow, FixedPixelDecoder, load_statistics, preprocess_rgb)
from patch_video_metrics import annotation_binding, load_fg_mask, add_fg_bg
from eval_validation import clip_noise as object_clip_noise, metrics
from diagnose_persistence import time_metrics, save_gif

SOURCE_FILES = ['raw_pixel_flow_research.py', 'raw_pixel_codec.py', 'patch_video_metrics.py',
                'codec_flow_research.py', 'codec_research.py', 'model.py', 'small_object_flow.py',
                'trajectory_codec.py', 'trajectory_codec_3dtex.py', 'wan_model.py',
                'wan_token_adapter.py', 'diagnose_persistence.py', 'eval_validation.py',
                'cache_vae_features.py']
SAMPLING = [(4, 17), (4, 29), (16, 17), (16, 29)]


def clip_noise(ids, master, objects, latent_dim, device, dtype):
    if (objects, latent_dim) != (1280, 192):
        raise ValueError('Raw-pixel noise shape differs')
    common = object_clip_noise(ids, master, 250, 128, device, dtype).flatten(1)
    extra = object_clip_noise(ids, master + 1000003, 1, 213760, device, dtype).flatten(1)
    return torch.cat((common, extra), dim=1).reshape(-1,1280,192)


def training_noise(batch, primary_rng, extra_rng, device):
    common=torch.randn(batch,250,128,generator=primary_rng).flatten(1)
    extra=torch.randn(batch,213760,generator=extra_rng)
    noise=torch.cat((common,extra),1).reshape(batch,1280,192).to(device)
    t=torch.rand(batch,generator=primary_rng).to(device)
    return noise,t


def target_rgb_batch(bank, ids, device):
    # No future VAE feature/encoder is queried by this target path.
    return torch.stack([preprocess_rgb(bank.rgb(i)) for i in ids]).to(device)


def select_cohorts(a, bank):
    train, ids, showcase, provenance = pick_cohorts(a, bank)
    if a.fixture_only and a.cohorts:
        showcase = [r['id'] for r in json.loads(a.cohorts.read_text())['showcase4']]
        for identifier in showcase: bank.check_metadata(identifier)
        provenance['showcase_ids'] = showcase
        provenance.pop('showcase_note', None)
    provenance['evaluation_scope'] = 'Observed-prefix-only raw RGB generation; future RGB used separately for scoring'
    return train, ids, showcase, provenance


def frozen_state_sha(module):
    """Streaming state digest also supporting scalar buffers."""
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode())
        if isinstance(value, torch.Tensor):
            h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
            h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            h.update(json.dumps(value, sort_keys=True).encode())
    return h.hexdigest()


class RawPixelModel(nn.Module):
    def __init__(self, codec, flow, decoder, annotations=None, wrong_prefix_map=None):
        super().__init__()
        self.codec, self.flow, self.decoder = codec, flow, decoder
        self.objects, self.latent_dim = 1280, 192
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
            raise RuntimeError('Frozen prefix codec or RGB normalization changed')
        if any(p.requires_grad or p.grad is not None for m in [self.codec, self.decoder] for p in m.parameters()):
            raise RuntimeError('Frozen module received gradients')

    @torch.no_grad()
    def prefix_context(self, features, coords):
        if features.ndim != 3 or features.shape[1:] != (128, 48) or coords.shape != features.shape[:2] + (3,):
            raise ValueError('Prefix must contain only cached four-frame 2x8x8 tokens')
        return self.codec.adapt_features(features, coords).detach()

    @torch.no_grad()
    def target_latent(self, rgb):
        packed = self.decoder.encode_rgb(rgb).detach()
        return packed, None, None, packed.new_zeros(())

    def sample(self, context, initial_noise, steps):
        if steps < 1 or initial_noise.shape != (context.shape[0], 1280, 192):
            raise ValueError('Invalid native-patch sampling shape or steps')
        z = initial_noise
        for index in range(steps):
            t = torch.full((z.shape[0],), index / steps, device=z.device, dtype=torch.float32)
            z = z + self.flow(z, t, context).float() / steps
        return z


def build_model(a, bank, device):
    if a.fixture_only and device.type != 'cpu':
        raise ValueError('Fixture mode requires CPU')
    if a.flow_warmstart or a.denoise_render_weight:
        raise ValueError('Patch baseline admits fresh native flow and pure FM only')
    if not a.fixture_only and a.codec_sha != '9a4c2b7afaa4fc9e41d298ef680d0f4dd5b181bdf61068090c3a10a5cd536d26':
        raise ValueError('Production prefix codec must be the fixed C2 initial checkpoint')
    if not a.codec_sha:
        raise ValueError('Bind the exact frozen prefix-codec checkpoint SHA')
    codec, codec_identity = load_codec(a.codec_checkpoint, bank, a.codec_sha)
    cfg = codec.config
    if codec_identity['checkpoint_step'] != 0 or cfg['variant'] != 'trajectory' or cfg['encoder_depth'] != 0:
        raise ValueError('Prefix must be the designated v3 C2 baseline initial checkpoint')
    if not a.fixture_only and (cfg['objects'], cfg['latent_dim'], cfg['feature_width']) != (250, 128, 256):
        raise ValueError('Production prefix architecture differs from the matched object run')
    statistics, statistics_identity = load_statistics(a.statistics, a.statistics_sha, bank)
    decoder = FixedPixelDecoder(statistics['channel_mean'], statistics['channel_std']).to(device)
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
    elif a.fixture_only:
        members = bank.ids('validation')
        mapping = {v: members[(i + 1) % len(members)] for i, v in enumerate(members)}
    else:
        raise ValueError('Production wrong-prefix control requires the fixed pilot32/showcase4 cohorts')
    torch.manual_seed(a.seed + 10001)
    flow = RawPixelFlow(cfg['feature_width'], a.small_width, a.small_depth, a.small_heads)
    model = RawPixelModel(codec, flow, decoder, a.annotations, mapping).to(device)
    base = {'backend': 'raw_pixel64', 'initialization': 'fresh ObjectFlow; no backbone pretraining',
            'architecture': flow.model_config, 'target_contract': TARGET_CONTRACT}
    base['fingerprint_sha256'] = canonical_sha(base)
    identity = {'codec': codec_identity, 'base': base, 'flow_warmstart': None,
                'backend': 'raw_pixel64', 'position_policy': flow.get_extra_state(),
                'flow_model_config': flow.model_config,
                'token_contract': TARGET_CONTRACT,
                'pixel_normalization': {**statistics_identity, 'state_sha256': model._decoder_sha},
                'annotation_binding': annotation_binding(a.annotations, bank),
                'evaluation_contract': {'wrong_prefix_map': mapping,
                    'policy': 'pilot32 sorted cyclic shift1; disjoint showcase4 sorted cyclic shift1; target-ID noise'},
                'parameter_counts': {'total': sum(p.numel() for p in model.parameters()),
                    'trainable': sum(p.numel() for p in flow.parameters()),
                    'codec_frozen': sum(p.numel() for p in codec.parameters()),
                    'target_decoder_parameters': sum(p.numel() for p in decoder.parameters()), 'lora': 0}}
    return model, identity


def fm_loss(model, context, target, noise, t, coords=None, truth=None, render_weight=0.):
    if render_weight or coords is not None or truth is not None:
        raise ValueError('Raw image decoder does not enter the FM training loss')
    target = target.detach().float()
    zt = (1 - t[:, None, None]) * noise.float() + t[:, None, None] * target
    velocity = model.flow(zt, t, context).float()
    loss = F.mse_loss(velocity, target - noise.float())
    return loss, loss, loss.new_zeros(())


def evaluate(model, bank, ids, showcase, out, size, step, device, amp, deadline):
    if size != 64: raise ValueError('Raw-pixel metrics are native64 only')
    out.mkdir(parents=True, exist_ok=False)
    saved_rng, was_training = rng_state(device), model.training
    model.eval(); start = time.perf_counter(); before = frozen_state_sha(model)
    records, rows, artifacts = {}, [], []
    if not set(ids + showcase).issubset(model.wrong_prefix_map):
        raise ValueError('Evaluation IDs lack preregistered wrong-prefix mapping')
    wrong = model.wrong_prefix_map
    def check():
        if time.time() >= deadline: raise TimeoutError('Raw-pixel evaluation deadline')
    def save_array(path, **values):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary=path.with_name(path.name+'.tmp')
        with temporary.open('wb') as stream: np.savez(stream, **values)
        os.replace(temporary,path)
    try:
        with torch.inference_mode():
            for identifier in list(dict.fromkeys(ids + showcase)):
                check();generated={}
                for condition, prefix_id in [('real',identifier),('wrong',wrong[identifier])]:
                    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp and device.type=='cuda'):
                        context=model.prefix_context(*prefix_batch(bank,[prefix_id],device))
                    for steps, seed in SAMPLING:
                        check();noise=clip_noise([identifier],seed,1280,192,device,torch.float32)
                        latent=model.sample(context,noise,steps)
                        prediction=model.decoder.decode_tokens(latent,64)[0].float().cpu().numpy()
                        if not np.isfinite(prediction).all():raise FloatingPointError('Nonfinite raw generated pixels')
                        method=f'sample_{steps}_seed{seed}' if condition=='real' else f'wrong_prefix_{steps}_seed{seed}'
                        generated[method]=prediction
                        stem=f'{condition}_seed{seed}' if steps==16 else f'{condition}4_seed{seed}'
                        path=out/'samples'/identifier/(stem+'.npz')
                        save_array(path,latent=latent.float().cpu().numpy(),rgb=prediction)
                        artifacts.append({'target_id':identifier,'prefix_id':prefix_id,'condition':condition,
                            'seed':seed,'steps':steps,'relative_path':str(path.relative_to(out)),'sha256':digest(path),
                            'target_rgb_sha256':bank.records[identifier]['array_sha256'],
                            'prefix_rgb_sha256':bank.records[prefix_id]['array_sha256']})
                        atomic_json(out/'samples_manifest.json',artifacts)
                # No target RGB is accessed until all eight future-free samples exist.
                rgb=target_rgb_batch(bank,[identifier],device)
                target,_,_,_=model.target_latent(rgb)
                reconstruction=model.decoder.decode_tokens(target,64)[0].cpu().numpy()
                truth=rgb[0].cpu().numpy()
                max_error=float(np.max(np.abs(reconstruction-truth)))
                if max_error>2e-7:raise AssertionError('Fixed RGB pack/normalization reference changed pixels')
                save_array(out/'samples'/identifier/'future_RGB.npz',latent_mean=target.float().cpu().numpy(),
                           rgb=reconstruction,target_kind=np.asarray('raw_RGB_identity_reference'))
                video=torch.from_numpy(np.array(bank.rgb(identifier)[:4],copy=True)).permute(0,3,1,2).float()/255
                other=torch.from_numpy(np.array(bank.rgb(wrong[identifier])[:4],copy=True)).permute(0,3,1,2).float()/255
                past=F.interpolate(video,(64,64),mode='bilinear',align_corners=False,antialias=False).numpy()
                other_past=F.interpolate(other,(64,64),mode='bilinear',align_corners=False,antialias=False).numpy()
                fg=load_fg_mask(model.annotations,bank,identifier,64,required=model.annotations is not None)
                outputs={'persistence':np.repeat(past[-1:],20,0),'future_RGB_identity':reconstruction,**generated}
                methods={}
                for name,pred in outputs.items():
                    region=add_fg_bg(pred,truth,fg);temporal=time_metrics(pred,truth,past[-1]);scalar=metrics(pred,truth)
                    scalar['motion_delta_mse_after_first']=float(np.mean(temporal['motion_delta_mse'][1:]))
                    methods[name]={'scalar':{**scalar,**region['scalar']},'time':{**temporal,**region['time']}}
                item={'id':identifier,'source_array_sha256':bank.records[identifier]['array_sha256'],
                    'wrong_prefix_id':wrong[identifier],'wrong_prefix_source_sha256':bank.records[wrong[identifier]]['array_sha256'],
                    'methods':methods,'target_seen_method':'future_RGB_identity only; arithmetic roundtrip, not learned AE',
                    'reference_roundtrip_max_abs':max_error,'foreground_annotation_available':fg is not None}
                records[identifier]=item;atomic_json(out/'cases'/(identifier+'.json'),item)
                if identifier in showcase:
                    videos=[np.concatenate([past,truth]).transpose(0,2,3,1)]
                    display=['persistence','sample_4_seed17','sample_4_seed29','sample_16_seed17',
                             'wrong_prefix_16_seed17','sample_16_seed29','wrong_prefix_16_seed29']
                    for name in display:
                        prefix=other_past if name.startswith('wrong_prefix') else past
                        videos.append(np.concatenate([prefix,outputs[name]]).transpose(0,2,3,1))
                    labels=['GT64','Repeat last','Real4 /17','Real4 /29','Real16 /17','Wrong16 /17','Real16 /29','Wrong16 /29']
                    rows.append((identifier,videos))
                atomic_json(out/'progress.json',{'completed':len(records),'total':len(set(ids+showcase)),'step':step})
        rows.sort(key=lambda row:showcase.index(row[0]))
        if rows:save_gif(out/'comparison.gif',rows,labels,12,start_frame=0)
        model.assert_frozen();after=frozen_state_sha(model)
        if before!=after:raise RuntimeError('Model state changed during evaluation')
        primary=[records[i] for i in ids];preview=[records[i] for i in showcase]
        result={'status':'complete','step':step,'evaluation_ids':ids,'showcase_ids':showcase,
            'sampling':SAMPLING,'conditions':['real','wrong'],'seed_policy':'All seeds and both step counts retained; no best-of selection',
            'metric_size':64,'wrong_prefix_policy':'pilot32 sorted shift1; separate showcase4 sorted shift1; target-ID noise',
            'means':average_records(primary),'records':primary,
            'groups':{('pilot32' if len(ids)==32 else 'evaluation'):{'ids':ids,'mean':average_records(primary)},
                      'showcase4':{'ids':showcase,'mean':average_records(preview)}},
            'elapsed_seconds':time.perf_counter()-start,'model_state_unchanged':True,
            'state_before_sha256':before,'state_after_sha256':after,'saved_samples':len(artifacts),
            'scope':'Supplemental raw-pixel generation at64px, no target VAE/learned decoder; target agreement is not physical proof.'}
        atomic_json(out/'summary.json',result)
        atomic_json(out/'checksums.json',{str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p.name!='checksums.json'})
        return result
    finally:
        model.assert_frozen()
        if frozen_state_sha(model)!=before:raise RuntimeError('State changed on evaluation exit')
        model.train(was_training);restore_rng(saved_rng,device)


def validate_bound_checkpoint(ckpt, fixed=None):
    if ckpt.get('format') != 'raw_pixel_flow_v6' or not isinstance(ckpt.get('protocol'), dict):
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
    train_ids, eval_ids, showcase, cohorts = select_cohorts(a, bank)
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
    extra_noise_rng = torch.Generator().manual_seed(a.seed + 305)
    torch.manual_seed(a.seed + 404)
    source_sha = {name: digest(Path(__file__).with_name(name)) for name in SOURCE_FILES}
    fixed = {'version': 'raw-pixel-flow-v6', 'identity': identity,
             'data_manifest_sha256': bank.manifest_sha, 'cache_protocol_sha256': bank.protocol_sha,
             'cohorts': cohorts, 'seed': a.seed, 'batch': a.batch, 'lr': a.lr,
             'weight_decay': a.weight_decay, 'lora_rank': a.lora_rank, 'lora_alpha': a.lora_alpha,
             'denoise_render_weight': a.denoise_render_weight, 'points': a.points,
             'amp': a.amp, 'eval_size': a.eval_size, 'sampling': SAMPLING,
             'training_gpu_hour_budget': a.train_hours,
             'evaluation_schedule': {'every_updates': a.eval_every, 'save_every_updates': a.save_every},
             'noise_schedule': 'Common first32000 values from object250x128 primary generator; extra213760 from independent seed+305 generator; draw t from primary afterward. Preserve object primary-stream/time draw consumption at matched updates.',
             'source_sha256': source_sha,
             'objective': 'Raw future RGB64 standardized by train-only global channel moments, packed1280x192; z_t=(1-t)epsilon+t*z; velocity MSE target z-epsilon; no learned target codec',
             'comparison_scope': 'Supplemental raw pixel common64 probe; token count1280 and target scalar count245760 differ from object250x128 and native VAE96x192',
             'fixture_only': a.fixture_only,
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
        for generator, name in [(data_rng, 'data_rng'), (query_rng, 'query_rng'), (noise_rng, 'noise_rng'), (extra_noise_rng, 'extra_noise_rng')]:
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
        payload = {'format': 'raw_pixel_flow_v6', 'step': step,
                   'flow': flow_state(model.flow), 'optimizer': optimizer.state_dict(),
                   'protocol': fixed, 'protocol_sha256': signature,
                   'rng': rng_state(device), 'data_rng': data_rng.get_state(),
                   'query_rng': query_rng.get_state(), 'noise_rng': noise_rng.get_state(),
                   'extra_noise_rng': extra_noise_rng.get_state(),
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
                target, _, _, _ = model.target_latent(target_rgb_batch(bank, ids, device))
            noise, t = training_noise(a.batch, noise_rng, extra_noise_rng, device)
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
    _, ids, showcase, cohorts = select_cohorts(a, bank)
    device = torch.device(a.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.set_num_threads(a.threads)
    a.backend = 'raw_pixel64'
    a.fixture_only = bound.get('fixture_only', False)
    a.statistics_sha = bound['identity']['pixel_normalization']['file_sha256']
    cfg = bound['identity']['flow_model_config']
    a.small_width, a.small_depth, a.small_heads = cfg['width'], cfg['depth'], cfg['heads']
    a.lora_rank, a.lora_alpha = bound['lora_rank'], bound['lora_alpha']
    a.codec_sha = bound['identity']['codec']['checkpoint_sha256']
    if a.flow_warmstart:
        raise ValueError('Evaluation restores its checkpoint; no additional warmstart')
    model, identity = build_model(a, bank, device)
    for key in ['codec', 'base', 'backend', 'position_policy', 'flow_model_config', 'token_contract', 'parameter_counts', 'pixel_normalization', 'annotation_binding', 'evaluation_contract']:
        if identity[key] != bound['identity'][key]:
            raise ValueError('Evaluation model identity differs: ' + key)
    load_flow_state(model.flow, ckpt['flow'])
    deadline = min(a.deadline_unix or float('inf'), time.time() + a.max_hours * 3600)
    evaluate(model, bank, ids, showcase, a.out, a.eval_size, ckpt['step'], device, a.amp, deadline)
    atomic_json(a.out / 'identity.json', {'checkpoint_sha256': checkpoint_sha,
                'training_protocol_sha256': ckpt['protocol_sha256'], 'model_identity': identity,
                'cohorts': cohorts, 'no_optimizer': True})
    atomic_json(a.out / 'checksums.json', {str(p.relative_to(a.out)): digest(p) for p in sorted(a.out.rglob('*'))
                                         if p.is_file() and p.name != 'checksums.json'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['train', 'eval'])
    for name in ['data', 'cache', 'out', 'codec-checkpoint']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--codec-sha', required=True)
    p.add_argument('--statistics', type=Path, required=True)
    p.add_argument('--statistics-sha', required=True)
    p.add_argument('--fixture-only', action='store_true')
    p.add_argument('--annotations', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--resume', type=Path)
    p.add_argument('--small-width', type=int, default=384)
    p.add_argument('--small-depth', type=int, default=8)
    p.add_argument('--small-heads', type=int, default=6)
    p.add_argument('--seed', type=int, default=713)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--steps', type=int, default=100000)
    p.add_argument('--max-hours', type=float, default=.75)
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
    if a.eval_size != 64:
        p.error('Raw-pixel pilot has one common native metric size:64')
    if a.fixture_only and a.device != 'cpu':
        p.error('Fixture mode is CPU engineering only')
    # Fixed pure-FM contract. No object/warmstart/RGB-auxiliary compatibility path.
    a.backend = 'raw_pixel64'
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

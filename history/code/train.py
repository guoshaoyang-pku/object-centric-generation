"""Distributed joint video autoencoder and conditional object flow training.

No future frame is passed to the conditioning encoder during generation.
Dataset directories must contain independently split RGB uint8 NPY clips.
"""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def pretrained_provenance(root, revision=None):
    """Fingerprint the local backbone/VAE weights actually available for loading."""
    root = Path(root).resolve()
    files = []
    for subfolder in ('transformer','vae'):
        directory = root/subfolder
        if not directory.is_dir():
            raise FileNotFoundError(f'Missing pretrained submodel directory: {directory}')
        for path in sorted(directory.rglob('*')):
            if path.is_file() and (path.suffix in {'.safetensors','.bin'} or path.name.endswith('.json')):
                files.append({'path':str(path.relative_to(root)), 'bytes':path.stat().st_size,
                              'sha256':digest(path)})
    inferred_revision = root.name if re.fullmatch(r'[0-9a-f]{40}',root.name) else None
    return {'path':str(root),'revision':revision or inferred_revision,
            'revision_source':'explicit CLI' if revision else 'snapshot directory name' if inferred_revision else 'unknown',
            'files':files,
            'fingerprint_sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}


def optimizer_groups(model,args):
    scales={'ae':args.ae_lr_scale,'decoder':args.decoder_lr_scale,
            'lora':args.lora_lr_scale,'adapter':args.adapter_lr_scale}
    named=[(name,value) for name,value in model.named_parameters() if value.requires_grad]
    if all(value==1. for value in scales.values()):
        return [{'params':[value for _,value in named], 'group_name':'all_trainable',
                 'param_names':[name for name,_ in named], 'lr_scale':1., 'lr':args.lr}]
    buckets={name:[] for name in scales}
    for name,value in named:
        if 'lora_A' in name or 'lora_B' in name:
            group='lora'
        elif name.startswith(('vision.','visual_adapter.','posterior.')):
            group='ae'
        elif name.startswith('decoder.'):
            group='decoder'
        else:
            group='adapter'
        buckets[group].append((name,value))
    return [{'params':[value for _,value in values], 'group_name':group,
             'param_names':[name for name,_ in values], 'lr_scale':scales[group],
             'lr':args.lr*scales[group]} for group,values in buckets.items() if values]


class VideoClips(Dataset):
    def __init__(self, root, frames, image_size):
        self.root = Path(root)
        self.files = sorted(self.root.glob('*.npy'))
        if not self.files:
            raise FileNotFoundError(f'No NPY clips: {root}')
        self.frames, self.image_size = frames, image_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        a = np.array(np.load(path, allow_pickle=False), copy=True)
        if a.dtype != np.uint8 or a.ndim != 4 or a.shape[-1] != 3 or len(a) < self.frames:
            raise ValueError(f'Expected uint8 [T,H,W,3], T>={self.frames}: {path}: {a.shape}/{a.dtype}')
        x = torch.from_numpy(a[:self.frames]).permute(0, 3, 1, 2).float().div_(255)
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        return x


class BatchStream:
    def __init__(self, loader, sampler, epoch=0, offset=0):
        self.loader, self.sampler = loader, sampler
        self.epoch, self.offset = epoch, 0
        self.sampler.set_epoch(epoch)
        self.iterator = iter(loader)
        for _ in range(offset):
            next(self.iterator)
            self.offset += 1

    def next(self):
        try:
            value = next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self.offset = 0
            self.sampler.set_epoch(self.epoch)
            self.iterator = iter(self.loader)
            value = next(self.iterator)
        self.offset += 1
        return value


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


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed % (2**63-1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed % (2**63-1))


def barrier(world):
    if world > 1:
        dist.barrier()


def json_write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    os.replace(tmp, path)


def save_checkpoint(model, optimizer, stream, args, step, device, rank, world, run_state):
    states = [None] * world if rank == 0 else None
    state = rng_state(device)
    if world > 1:
        dist.gather_object(state, states, dst=0)
    else:
        states = [state]
    if rank == 0:
        folder = args.out / 'checkpoints'
        folder.mkdir(exist_ok=True)
        path = folder / f'step_{step:07d}.pt'
        tmp = path.with_suffix('.pt.tmp')
        keep = {name for name,parameter in model.named_parameters() if parameter.requires_grad}
        keep.update(name for name,_ in model.named_buffers())
        # Frozen multi-billion-parameter weights remain in the fingerprinted
        # pretrained snapshot; adapters and all persistent buffers are saved.
        model_state = {name:value.detach().cpu() for name,value in model.state_dict().items() if name in keep}
        torch.save({'model': model_state, 'model_state_format':'trainable_and_buffers_v1',
                    'optimizer': optimizer.state_dict(),
                    'step': step, 'args': vars(args), 'rng_by_rank': states,
                    'stream_epoch': stream.epoch, 'stream_offset': stream.offset,
                    'world_size': world, **run_state}, tmp)
        os.replace(tmp, path)
        latest = folder / 'latest.pt'
        latest_tmp = folder / 'latest.pt.tmp'
        if latest_tmp.exists():
            latest_tmp.unlink()
        try:
            os.link(path, latest_tmp)
        except OSError:
            shutil.copyfile(path, latest_tmp)
        os.replace(latest_tmp, latest)
        for old in sorted(folder.glob('step_*.pt'))[:-args.keep_checkpoints]:
            old.unlink()
        json_write(args.out / 'status.json', {'status': 'running', 'step': step,
                   'checkpoint': str(path), 'unix_time': time.time(),
                   'samples_seen':run_state['samples_seen'],
                   'accumulated_training_seconds':run_state['accumulated_training_seconds']})


@torch.no_grad()
def evaluate(model, validation, ids, args, step, device):
    model.eval()
    folder = args.out / 'previews' / f'step_{step:07d}'
    folder.mkdir(parents=True, exist_ok=True)
    examples, records = [], []
    # Evaluation must not alter the subsequent training RNG sequence.
    devices = [device.index] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        for index in ids:
            full = validation[index].unsqueeze(0).to(device)
            context = full[:, :args.context_frames]
            gt = full[:, args.context_frames:]
            truth = F.interpolate(gt.flatten(0, 1), (args.preview_size, args.preview_size), mode='bilinear', align_corners=False).reshape(1, args.future_frames, 3, args.preview_size, args.preview_size)
            past = F.interpolate(context.flatten(0, 1), (args.preview_size, args.preview_size), mode='bilinear', align_corners=False).reshape(1, args.context_frames, 3, args.preview_size, args.preview_size)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda' and args.bf16):
                reconstruction = model.reconstruct(full, height=args.preview_size, width=args.preview_size).float()
                generated = model.sample(context, steps=args.sample_steps, height=args.preview_size,
                                         width=args.preview_size, future_frames=args.future_frames,
                                         seed=args.seed + 100000 + index).float()
            if not torch.isfinite(reconstruction).all() or not torch.isfinite(generated).all():
                raise FloatingPointError('Nonfinite evaluation output')
            reconstruction, generated = reconstruction.clamp(0, 1), generated.clamp(0, 1)
            rec_mse = float((reconstruction-truth).square().mean())
            gen_mse = float((generated-truth).square().mean())
            persistence = float((past[:, -1:] - truth).square().mean())
            record = {'validation_id': validation.files[index].stem,
                      'reconstruction_mse': rec_mse, 'reconstruction_psnr': -10*math.log10(max(rec_mse, 1e-12)),
                      'generated_mse_single_sample': gen_mse,
                      'persistence_mse': persistence, 'resolution': args.preview_size,
                      'generation_condition': f'only first {args.context_frames} RGB frames',
                      'note': 'Single-sample pixel error is not a distributional or physics metric.'}
            # These diagnostics distinguish content/motion from a constant mean
            # image even when a flat prediction obtains a tolerable RGB MSE.
            def content_metrics(prediction):
                spatial = float(prediction.std(dim=(-2,-1),unbiased=False).mean())
                motion = float(prediction.diff(dim=1).abs().mean()) if prediction.shape[1]>1 else 0.
                errors = [((prediction.diff(dim=axis)-truth.diff(dim=axis))**2).mean() for axis in (-1,-2)]
                a=prediction-prediction.mean(dim=(-2,-1),keepdim=True)
                b=truth-truth.mean(dim=(-2,-1),keepdim=True)
                correlation=float((a*b).mean()/((a.square().mean()*b.square().mean()).sqrt()+1e-8))
                return {'spatial_std':spatial,'frame_change_mae':motion,
                        'spatial_difference_mse':float(torch.stack(errors).mean()),
                        'centered_rgb_correlation':correlation}
            record['target_content'] = content_metrics(truth)
            record['reconstruction_content'] = content_metrics(reconstruction)
            record['generated_content'] = content_metrics(generated)
            records.append(record)
            arrays = [v[0].detach().cpu().numpy() for v in (past, truth, reconstruction, generated)]
            np.savez_compressed(folder / (validation.files[index].stem + '.npz'),
                                context=arrays[0], target=arrays[1], reconstruction=arrays[2], generated=arrays[3])
            examples.append(arrays)
    frames = []
    display = max(128, args.preview_size)
    for frame_index in range(args.context_frames + args.future_frames):
        canvas = Image.new('RGB', (display*3, len(examples)*(display+44)), '#181818')
        draw = ImageDraw.Draw(canvas)
        for row, arrays in enumerate(examples):
            past, truth, reconstruction, generated = arrays
            values = [past[frame_index]]*3 if frame_index < args.context_frames else [v[frame_index-args.context_frames] for v in (truth, reconstruction, generated)]
            phase = 'PAST / observed' if frame_index < args.context_frames else 'FUTURE / held out'
            for col, (title, value) in enumerate(zip(['Ground truth', 'AE reconstruction', 'Conditional generation'], values)):
                x, y = col*display, row*(display+44)
                image = Image.fromarray((value.transpose(1, 2, 0).clip(0, 1)*255).round().astype(np.uint8))
                canvas.paste(image.resize((display, display), Image.Resampling.NEAREST), (x, y+44))
                draw.text((x+4, y+3), title, fill='white')
                draw.text((x+4, y+18), f'{phase} | frame {frame_index+1}', fill='#efbb66' if frame_index < args.context_frames else '#8ed0ab')
                draw.text((x+4, y+31), records[row]['validation_id'][:25], fill='white')
        frames.append(canvas)
    durations = [round((i+1)*1000/args.fps)-round(i*1000/args.fps) for i in range(len(frames))]
    # GIF quantizes to centiseconds; cumulative rounding preserves clip duration.
    durations = [round((i+1)*100/args.fps)*10-round(i*100/args.fps)*10 for i in range(len(frames))]
    frames[0].save(folder / 'comparison.gif', save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, disposal=2)
    frames[len(frames)//2].save(folder / 'midpoint.png')
    summary = {'step': step, 'records': records,
               'reconstruction_mse_mean': float(np.mean([x['reconstruction_mse'] for x in records])),
               'generated_mse_single_sample_mean': float(np.mean([x['generated_mse_single_sample'] for x in records])),
               'persistence_mse_mean': float(np.mean([x['persistence_mse'] for x in records]))}
    json_write(folder / 'metrics.json', summary)
    with (args.out / 'evaluation.jsonl').open('a') as stream:
        stream.write(json.dumps(summary) + '\n')
    model.train()
    return summary


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model-kind', choices=['scratch','wan'], default='wan')
    p.add_argument('--pretrained-path', type=Path)
    p.add_argument('--pretrained-revision', help='Verified snapshot commit, when not encoded in local path')
    p.add_argument('--lora-rank', type=int, default=8)
    p.add_argument('--lora-alpha', type=float, default=8.)
    p.add_argument('--posterior-locality', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--posterior-sigma', type=float, nargs=3, default=[.4,.5,.5])
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--max-hours', type=float, default=10.,
                   help='Cumulative known run wall-time budget across resumed stages')
    p.add_argument('--microbatch', type=int, default=1)
    p.add_argument('--grad-accum', type=int, default=2)
    p.add_argument('--num-points', type=int, default=512)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--lr-end', type=float, default=2e-5)
    p.add_argument('--ae-lr-scale', type=float, default=1.)
    p.add_argument('--decoder-lr-scale', type=float, default=1.)
    p.add_argument('--lora-lr-scale', type=float, default=1.)
    p.add_argument('--adapter-lr-scale', type=float, default=1.)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--flow-ramp', type=int, default=2000)
    p.add_argument('--flow-start-weight', type=float, default=.05)
    p.add_argument('--kl-weight', type=float, default=1e-5)
    p.add_argument('--generated-rgb-weight', type=float, default=0.,
                   help='Wan only: t-weighted RGB loss on teacher-noised one-step endpoint, not prefix-only sampling')
    p.add_argument('--rollout-rgb-weight', type=float, default=0.,
                   help='Wan only: RGB supervision through all steps of a pure-noise prefix-only rollout')
    p.add_argument('--rollout-steps', type=int, default=4,
                   help='Differentiable rollout steps; default 4 differs from default 16-step evaluation')
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--checkpoint-every', type=int, default=500)
    p.add_argument('--keep-checkpoints', type=int, default=3)
    p.add_argument('--eval-clips', type=int, default=2)
    p.add_argument('--preview-size', type=int, default=64)
    p.add_argument('--sample-steps', type=int, default=16)
    p.add_argument('--fps', type=float, default=12.)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--resume', type=Path)
    p.add_argument('--stop-file', type=Path, help='Rank zero checks this sentinel and checkpoints/evaluates before stopping')
    p.add_argument('--allow-world-size-change', action='store_true',
                   help='Explicitly allow changed distributed/batch layout; reset RNG and data cursor')
    p.add_argument('--allow-dataset-change', action='store_true',
                   help='Explicitly allow a different dataset snapshot; preserve cross-stage heldout separation')
    p.add_argument('--resume-provenance', type=Path,
                   help='Old run provenance for legacy checkpoints without split hashes')
    p.add_argument('--bf16', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--find-unused-parameters', action='store_true')
    for name, value in [('image-size',128), ('context-frames',4), ('future-frames',20),
                        ('objects',250), ('subkernels',4), ('latent-dim',128), ('width',512),
                        ('depth',12), ('heads',8), ('encoder-width',256), ('encoder-depth',4),
                        ('texture-kernels',64)]:
        p.add_argument('--'+name, type=int, default=value)
    a = p.parse_args()
    if a.model_kind == 'wan' and a.pretrained_path is None:
        p.error('--pretrained-path is required for --model-kind wan')
    if a.lora_rank < 1 or a.lora_alpha<=0:
        p.error('--lora-rank and --lora-alpha must be positive')
    if min(a.posterior_sigma)<=0 or min(a.ae_lr_scale,a.decoder_lr_scale,a.lora_lr_scale,a.adapter_lr_scale)<=0:
        p.error('Posterior sigmas and learning-rate scales must be positive')
    if not 0<=a.flow_start_weight<=1:
        p.error('--flow-start-weight must be in [0,1]')
    if not math.isfinite(a.generated_rgb_weight) or a.generated_rgb_weight<0:
        p.error('--generated-rgb-weight must be finite and nonnegative')
    if a.generated_rgb_weight and a.model_kind!='wan':
        p.error('--generated-rgb-weight requires --model-kind wan')
    if not math.isfinite(a.rollout_rgb_weight) or a.rollout_rgb_weight<0 or a.rollout_steps<1:
        p.error('--rollout-rgb-weight must be finite/nonnegative and --rollout-steps positive')
    if a.rollout_rgb_weight and a.model_kind!='wan':
        p.error('--rollout-rgb-weight requires --model-kind wan')
    if min(a.steps, a.microbatch, a.grad_accum, a.num_points, a.eval_every, a.checkpoint_every,
           a.eval_clips, a.keep_checkpoints, a.preview_size, a.sample_steps) < 1 or a.max_hours <= 0:
        p.error('Training and evaluation sizes/intervals and max-hours must be positive')
    return a


def main():
    args = arguments()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    rank, local_rank, world = int(os.environ.get('RANK',0)), int(os.environ.get('LOCAL_RANK',0)), int(os.environ.get('WORLD_SIZE',1))
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        init_kwargs = {'backend': 'nccl' if device.type == 'cuda' else 'gloo',
                       'timeout': timedelta(minutes=60)}
        if device.type == 'cuda':
            # Global rank modulo local GPU count is wrong for heterogeneous
            # node allocations; bind collectives to the explicit local device.
            init_kwargs['device_id'] = device
        dist.init_process_group(**init_kwargs)
    seed_all(args.seed+rank)
    if rank == 0:
        if args.out.exists() and any(args.out.iterdir()) and not args.resume:
            raise FileExistsError(f'Refusing to overwrite {args.out}; provide --resume')
        args.out.mkdir(parents=True, exist_ok=True)
    barrier(world)
    train = VideoClips(args.data/'train', args.context_frames+args.future_frames, args.image_size)
    validation = VideoClips(args.data/'validation', args.context_frames+args.future_frames, args.image_size)
    sampler = DistributedSampler(train, num_replicas=world, rank=rank, seed=args.seed, shuffle=True, drop_last=True)
    loader = DataLoader(train, batch_size=args.microbatch, sampler=sampler, num_workers=args.workers,
                        pin_memory=device.type == 'cuda', drop_last=True, persistent_workers=args.workers > 0)
    if not len(loader):
        raise ValueError('Too few train clips for world size and microbatch')
    cfg = {key: getattr(args, key) for key in ['image_size','context_frames','future_frames','objects','subkernels',
           'latent_dim','width','depth','heads','encoder_width','encoder_depth','texture_kernels']}
    cfg.update(pretrained_path=str(args.pretrained_path) if args.pretrained_path else None,
               lora_rank=args.lora_rank,lora_alpha=args.lora_alpha,
               posterior_locality=args.posterior_locality,posterior_sigma=args.posterior_sigma,
               model_kind=args.model_kind)
    if args.model_kind == 'wan':
        from wan_model import PretrainedJointObjectVideo
        model = PretrainedJointObjectVideo(cfg).to(device)
    else:
        from model import JointObjectVideo
        model = JointObjectVideo(cfg).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError('The model has no trainable parameters')
    groups=optimizer_groups(model,args)
    group_settings=[{key:value for key,value in group.items() if key!='params'} for group in groups]
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    architecture={'configuration':{key:value for key,value in cfg.items() if key!='pretrained_path'},
                  'parameters':[(name,list(value.shape),str(value.dtype),value.requires_grad)
                                for name,value in model.named_parameters()],
                  'buffers':[(name,list(value.shape),str(value.dtype)) for name,value in model.named_buffers()]}
    args.architecture_fingerprint_sha256=hashlib.sha256(json.dumps(architecture,sort_keys=True).encode()).hexdigest()
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    start_step = 0
    if checkpoint:
        if checkpoint['args'].get('model_kind','scratch') != args.model_kind:
            raise ValueError('Resume model kind differs')
        old_architecture=checkpoint['args'].get('architecture_fingerprint_sha256')
        if old_architecture and old_architecture!=args.architecture_fingerprint_sha256:
            raise ValueError('Resume architecture/parameter schema fingerprint differs')
        for key, value in cfg.items():
            if key == 'model_kind':
                continue
            if key == 'pretrained_path':
                # Cache roots can move between nodes. The full weight/config
                # fingerprint below remains strict.
                continue
            legacy_defaults={'lora_alpha':8.,'posterior_locality':False,'posterior_sigma':None}
            if checkpoint['args'].get(key,legacy_defaults.get(key)) != value:
                raise ValueError(f'Resume architecture differs: {key}')
        if checkpoint.get('model_state_format') == 'trainable_and_buffers_v1':
            missing,unexpected = model.load_state_dict(checkpoint['model'],strict=False)
            required = {name for name,parameter in model.named_parameters() if parameter.requires_grad}
            if unexpected or required.intersection(missing):
                raise ValueError(f'Invalid adapter checkpoint: missing trainable={required.intersection(missing)}, unexpected={unexpected}')
        else:
            model.load_state_dict(checkpoint['model'])
        if len(checkpoint['optimizer']['param_groups'])!=len(group_settings):
            raise ValueError('Resume optimizer group layout differs; pass the previous learning-rate scale configuration')
        optimizer.load_state_dict(checkpoint['optimizer'])
        for group,settings in zip(optimizer.param_groups,group_settings):
            group.update(settings)
            group['weight_decay']=args.weight_decay
        start_step = checkpoint['step']
    ids = list(range(min(args.eval_clips,len(validation))))
    if rank == 0:
        train_records = [{'id':p.stem,'path':str(p.resolve()),'sha256':digest(p)} for p in train.files]
        validation_records = [{'id':p.stem,'path':str(p.resolve()),'sha256':digest(p)} for p in validation.files]
        if {x['sha256'] for x in train_records} & {x['sha256'] for x in validation_records}:
            raise ValueError('Exact RGB duplicate found across train and validation')
        try:
            revision = subprocess.check_output(['git','rev-parse','HEAD'], cwd=Path(__file__).parent, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            revision = None
        snapshot = hashlib.sha256(json.dumps({'train':[(x['id'],x['sha256']) for x in train_records],
                    'validation':[(x['id'],x['sha256']) for x in validation_records]},sort_keys=True).encode()).hexdigest()
        current_train_hashes={x['sha256'] for x in train_records}
        current_validation_hashes={x['sha256'] for x in validation_records}
        old_split_hashes=checkpoint.get('data_split_hashes') if checkpoint else None
        dataset_changed=bool(checkpoint and checkpoint['args'].get('dataset_snapshot_sha256')!=snapshot)
        if checkpoint and not old_split_hashes and dataset_changed:
            candidates=[args.resume_provenance] if args.resume_provenance else [
                args.resume.parent.parent/'resume_provenance.json',args.resume.parent.parent/'provenance.json']
            for candidate in candidates:
                if candidate and candidate.exists():
                    previous=json.loads(candidate.read_text())
                    if previous.get('dataset_snapshot_sha256')==checkpoint['args'].get('dataset_snapshot_sha256'):
                        old_split_hashes={'train':[x['sha256'] for x in previous['training_clips']],
                                          'validation':[x['sha256'] for x in previous['validation_clips']]}
                        break
            if not old_split_hashes:
                raise ValueError('Dataset migration requires old split hashes; supply --resume-provenance for this legacy checkpoint')
        old_train_hashes=set((old_split_hashes or {}).get('train',[]))
        old_validation_hashes=set((old_split_hashes or {}).get('validation',[]))
        if old_validation_hashes & current_train_hashes:
            raise ValueError('A previously held-out validation clip would enter training')
        if old_train_hashes & current_validation_hashes:
            raise ValueError('A previously trained clip would enter validation')
        split_history={'train':sorted(old_train_hashes|current_train_hashes),
                       'validation':sorted(old_validation_hashes|current_validation_hashes)}
        pretrained = pretrained_provenance(args.pretrained_path,args.pretrained_revision) if args.model_kind=='wan' else None
        provenance = {'args':vars(args), 'model_config':cfg, 'world_size':world,
                      'architecture_fingerprint_sha256':args.architecture_fingerprint_sha256,
                      'dataset_snapshot_sha256':snapshot,
                      'global_batch':world*args.microbatch*args.grad_accum,
                      'training_clips':train_records,'validation_clips':validation_records,
                      'fixed_validation_ids':[validation.files[i].stem for i in ids],
                      'source_sha256':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
                      'data_manifest_sha256':{p.name:digest(p) for p in args.data.glob('*.json')},
                      'git_revision':revision, 'torch':torch.__version__, 'cuda':torch.version.cuda,
                      'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
                      'parameter_counts': model.parameter_counts() if hasattr(model,'parameter_counts') else {'total':sum(p.numel() for p in model.parameters())},
                      'parameter_ownership':{'trainable':sum(p.numel() for p in trainable),
                          'frozen':sum(p.numel() for p in model.parameters() if not p.requires_grad),
                          'optimizer_parameters':sum(p.numel() for p in trainable)},
                      'optimizer_groups':[{**settings,'parameters':sum(p.numel() for p in group['params'])}
                                          for settings,group in zip(group_settings,groups)],
                      'pretrained':pretrained,
                      'model_pretrained_info':getattr(model,'pretrained_info',None),
                      'rgb_supervision':{
                          'teacher_noised_endpoint_weight':args.generated_rgb_weight,
                          'prefix_only_rollout_weight':args.rollout_rgb_weight,
                          'rollout_steps':args.rollout_steps,'evaluation_steps':args.sample_steps,
                          'step_mismatch':bool(args.rollout_rgb_weight and args.rollout_steps!=args.sample_steps),
                          'rollout_gradient':'all Euler steps, nonreentrant checkpoint; prefix shared adapter detached',
                          'limitations':'Paired RGB MSE may average stochastic futures; small matched pilot, not evidence of physical fidelity'},
                      'condition':'past frames only; future encoder is training target branch',
                      'claim':'new joint-training pilot; not an established or validated physics model'}
    snapshot_container = [snapshot if rank == 0 else None,
                          pretrained['fingerprint_sha256'] if rank == 0 and pretrained else None,
                          split_history if rank == 0 else None]
    if world > 1:
        dist.broadcast_object_list(snapshot_container,src=0)
    args.dataset_snapshot_sha256 = snapshot_container[0]
    args.pretrained_fingerprint_sha256 = snapshot_container[1]
    world_changed=bool(checkpoint and checkpoint['world_size']!=world)
    dataset_changed=bool(checkpoint and checkpoint['args'].get('dataset_snapshot_sha256')!=args.dataset_snapshot_sha256)
    batch_changed=bool(checkpoint and (checkpoint['args']['microbatch']!=args.microbatch or checkpoint['args']['grad_accum']!=args.grad_accum))
    if (world_changed or batch_changed) and not args.allow_world_size_change:
        raise ValueError('Resume distributed/batch layout differs; explicitly pass --allow-world-size-change')
    if dataset_changed and not args.allow_dataset_change:
        raise ValueError('Resume dataset differs; explicitly pass --allow-dataset-change')
    if checkpoint and checkpoint['args'].get('pretrained_fingerprint_sha256') != args.pretrained_fingerprint_sha256:
        raise ValueError('Resume requires identical frozen pretrained weights and configs')
    reset_stream=world_changed or dataset_changed or batch_changed
    if checkpoint and not reset_stream and args.seed!=checkpoint['args']['seed']:
        raise ValueError('An exact resume requires the original seed')
    old_global_batch=checkpoint['world_size']*checkpoint['args']['microbatch']*checkpoint['args']['grad_accum'] if checkpoint else 0
    prior_samples=int(checkpoint.get('samples_seen',start_step*old_global_batch)) if checkpoint else 0
    prior_seconds=float(checkpoint.get('accumulated_training_seconds',0.)) if checkpoint else 0.
    prior_time_known=not checkpoint or bool(checkpoint.get('accumulated_training_seconds_known',
                                                         'accumulated_training_seconds' in checkpoint))
    sampler_seed=args.seed+start_step if reset_stream else int(checkpoint.get('sampler_seed',args.seed)) if checkpoint else args.seed
    sampler.seed=sampler_seed
    config_defaults={'generated_rgb_weight':0.,'rollout_rgb_weight':0.,'rollout_steps':4}
    training_config_changes={key:{'old':checkpoint['args'].get(key,config_defaults.get(key)),'new':getattr(args,key)}
        for key in ['steps','lr','lr_end','warmup','flow_ramp','flow_start_weight','kl_weight','weight_decay','num_points','bf16',
                    'ae_lr_scale','decoder_lr_scale','lora_lr_scale','adapter_lr_scale','generated_rgb_weight',
                    'rollout_rgb_weight','rollout_steps']
        if checkpoint and checkpoint['args'].get(key,config_defaults.get(key))!=getattr(args,key)}
    transition={'resume_source':str(args.resume.resolve()) if args.resume else None,
                'old_world_size':checkpoint['world_size'] if checkpoint else None,'new_world_size':world,
                'world_size_changed':world_changed,'dataset_changed':dataset_changed,'batch_layout_changed':batch_changed,
                'rng_and_data_cursor_reset':reset_stream,'sampler_seed':sampler_seed,
                'bitwise_reproducible_resume':False if reset_stream or training_config_changes else None,
                'rng_and_cursor_restored':bool(checkpoint and not reset_stream),
                'training_config_changes':training_config_changes,
                'samples_seen_from_legacy_formula':bool(checkpoint and 'samples_seen' not in checkpoint),
                'prior_runtime_known':prior_time_known,
                'note':'Explicit migration preserves model and optimizer; reset changes sample order and is not bitwise reproducible.' if reset_stream
                       else 'RNG/data cursor restored; bitwise reproducibility has not been verified.' if checkpoint else None}
    run_state={'samples_seen':prior_samples,'accumulated_training_seconds':prior_seconds,
               'accumulated_training_seconds_known':prior_time_known,'sampler_seed':sampler_seed,
               'data_split_hashes':snapshot_container[2],'resume_transition':transition}
    if rank==0:
        provenance.update(resume_transition=transition,prior_samples_seen=prior_samples,
                          prior_accumulated_training_seconds=prior_seconds,
                          runtime_budget='max-hours applies cumulatively to known run wall time across resumes')
        json_write(args.out/('resume_provenance.json' if checkpoint else 'provenance.json'),provenance)
    barrier(world)
    wrapped = DistributedDataParallel(model, device_ids=[local_rank] if device.type=='cuda' else None,
                                      broadcast_buffers=False,
                                      find_unused_parameters=args.find_unused_parameters) if world>1 else model
    stream = BatchStream(loader, sampler, checkpoint.get('stream_epoch',0) if checkpoint and not reset_stream else 0,
                         checkpoint.get('stream_offset',0) if checkpoint and not reset_stream else 0)
    if checkpoint and not reset_stream:
        restore_rng(checkpoint['rng_by_rank'][rank],device)
    elif reset_stream:
        seed_all(args.seed+start_step*1000003+rank)
    model.train()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    if prior_seconds>=args.max_hours*3600:
        if rank==0:
            json_write(args.out/'status.json',{'status':'time_limit_already_reached','step':start_step,
                       'samples_seen':prior_samples,'accumulated_training_seconds':prior_seconds,
                       'resume_source':str(args.resume),'unix_time':time.time()})
        barrier(world)
        if world>1:dist.destroy_process_group()
        return
    began = time.monotonic(); last_report = began
    final_step = start_step
    for step in range(start_step+1,args.steps+1):
        fraction = (step-1)/max(args.steps-1,1)
        rate = args.lr_end + .5*(args.lr-args.lr_end)*(1+math.cos(math.pi*fraction))
        rate *= min(1., step/max(args.warmup,1))
        for group in optimizer.param_groups: group['lr']=rate*group.get('lr_scale',1.)
        flow_weight = args.flow_start_weight + (1-args.flow_start_weight)*min(1.,step/max(args.flow_ramp,1))
        optimizer.zero_grad(set_to_none=True)
        metrics = torch.zeros(8,device=device)
        for micro in range(args.grad_accum):
            video = stream.next().to(device,non_blocking=True)
            sync = wrapped.no_sync() if world>1 and micro<args.grad_accum-1 else nullcontext()
            with sync:
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda' and args.bf16):
                    extra={'generated_rgb_aux':True} if args.generated_rgb_weight else {}
                    if args.rollout_rgb_weight:
                        extra.update(rollout_rgb_aux=True,rollout_steps=args.rollout_steps)
                    values = wrapped(video,num_points=args.num_points,**extra)
                    loss = values['recon'] + flow_weight*values['flow'] + args.kl_weight*values['kl']
                    if args.generated_rgb_weight:
                        loss=loss+args.generated_rgb_weight*values['generated_rgb']
                    if args.rollout_rgb_weight:
                        loss=loss+args.rollout_rgb_weight*values['rollout_rgb']
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss rank={rank} step={step}')
                (loss/args.grad_accum).backward()
            metrics += torch.stack([loss.detach(),values['recon'].detach(),values['flow'].detach(),
                                    values['kl'].detach(),values['latent_std'].detach(),
                                    values.get('generated_rgb',loss.new_zeros(())).detach(),
                                    values.get('generated_rgb_unweighted',loss.new_zeros(())).detach(),
                                    values.get('rollout_rgb',loss.new_zeros(())).detach()]).float()/args.grad_accum
        grad = torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
        optimizer.step()
        final_step = step
        run_state['samples_seen']+=world*args.microbatch*args.grad_accum
        if world>1:
            dist.all_reduce(metrics);metrics/=world
        now = time.monotonic()
        run_state['accumulated_training_seconds']=prior_seconds+now-began
        if rank==0:
            elapsed_step = now-last_report
            record={'step':step,'elapsed_seconds':now-began,
                    'session_gpu_hours':world*(now-began)/3600,
                    'session_gpu_count':world,
                    'accumulated_training_seconds':run_state['accumulated_training_seconds'],
                    'samples_seen':run_state['samples_seen'],
                    'session_samples_seen':(step-start_step)*world*args.microbatch*args.grad_accum,
                    'lr':rate,'flow_weight':flow_weight,'loss':float(metrics[0]),'recon':float(metrics[1]),
                    'group_lrs':{group.get('group_name',str(i)):group['lr'] for i,group in enumerate(optimizer.param_groups)},
                    'flow':float(metrics[2]),'kl':float(metrics[3]),'latent_std':float(metrics[4]),
                    'generated_rgb_teacher_noised_t_weighted':float(metrics[5]),
                    'generated_rgb_teacher_noised_unweighted':float(metrics[6]),
                    'generated_rgb_weight':args.generated_rgb_weight,
                    'rollout_rgb_prefix_only':float(metrics[7]),'rollout_rgb_weight':args.rollout_rgb_weight,
                    'rollout_steps':args.rollout_steps,
                    'grad_norm_rank0':float(grad),'step_seconds':elapsed_step,
                    'clips_per_second':world*args.microbatch*args.grad_accum/max(elapsed_step,1e-9),
                    'peak_memory_bytes_rank0':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0}
            with (args.out/'metrics.jsonl').open('a') as output:output.write(json.dumps(record)+'\n')
            if step<=5 or step%10==0: print(json.dumps(record),flush=True)
        requested=bool(rank==0 and args.stop_file and args.stop_file.exists())
        stop = torch.tensor(2 if requested else int(run_state['accumulated_training_seconds'] >= args.max_hours*3600),device=device)
        if world>1:dist.all_reduce(stop,op=dist.ReduceOp.MAX)
        final = step==args.steps or bool(stop)
        stop_reason='requested_stop' if int(stop)==2 else 'time_limit' if int(stop)==1 else 'completed'
        run_state['stop_reason']=stop_reason if final else None
        if step%args.checkpoint_every==0 or final:
            save_checkpoint(model,optimizer,stream,args,step,device,rank,world,run_state)
        if step%args.eval_every==0 or final:
            barrier(world)
            if rank==0:
                summary=evaluate(model,validation,ids,args,step,device)
                print(json.dumps({'evaluation':summary}),flush=True)
            barrier(world)
        last_report = time.monotonic()
        if final:break
    barrier(world)
    if rank==0:
        json_write(args.out/'status.json',{'status':stop_reason if final_step>start_step else 'completed',
                   'step':final_step,'elapsed_seconds':time.monotonic()-began,
                   'accumulated_training_seconds':prior_seconds+time.monotonic()-began,
                   'samples_seen':run_state['samples_seen'],'unix_time':time.time()})
    if world>1:dist.destroy_process_group()


if __name__=='__main__':
    main()

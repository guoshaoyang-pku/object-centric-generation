"""DDP runner for the oracle codec (stage codec) and the flow on it (stage flow); amendment F.

Stage codec: OracleCodec from scratch. Loss = mixture RGB MSE + GT-routed RGB
MSE + lambda_own * ownership cross-entropy + kl * KL, on future query points
(half uniform, half on object pixels). Optional horizontal mirroring of whole
clips (--hflip), which is physics-symmetric in MOVi.

Stage flow: frozen oracle codec; posterior mean/log-variance of every train
clip is computed once per rank (no mirroring) and the target is a fresh
posterior sample each step, as in stable_flow_ddp.py. The flow is
OracleObjectFlow (QK-norm StableObjectFlow plus per-slot prefix glimpse
tokens); same width/depth/heads, LR schedule, EMA, global batch and
evaluation protocol (codec_flow_research.evaluate on pilot32/showcase4, fixed
seeds 17/29, 4/16 Euler steps, 64 px) as the stable patch-grid run.

GPU-hours: train = step wall x world; allocated = session wall x world.
Control files in --out: REQUEST_PAUSE saves and exits; REQUEST_STOP saves,
runs the final eval and exits. Heartbeat: status.json.
"""
import argparse
import collections
import copy
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle as orc  # noqa: E402  (puts the code_v4 base on sys.path)
from codec_research import digest, canonical_sha, atomic_json, sync, pick_cohorts  # noqa: E402
from codec_flow_research import evaluate, flow_state, load_flow_state, memory_peaks  # noqa: E402

FORMAT = 'claude_oracle_v1'
SOURCE_FILES = ['oracle.py', 'oracle_train.py', 'stable_flow.py']
CONTINUE, STOP, PAUSE, TARGET, CAP, DEADLINE, DIVERGED = range(7)
REASONS = {STOP: 'requested_stop', PAUSE: 'paused', TARGET: 'target_train_gpu_hours_reached',
           CAP: 'allocated_gpu_hour_cap', DEADLINE: 'wall_deadline', DIVERGED: 'diverged'}


def lr_at(a, step, train_seconds):
    if step < a.warmup_steps:
        return a.lr * (step + 1) / a.warmup_steps
    frac = min(max(train_seconds / max(a.decay_end_hours * 3600., 1.), 0.), 1.)
    return a.lr_min + .5 * (a.lr - a.lr_min) * (1 + math.cos(math.pi * frac))


@torch.no_grad()
def ema_update(ema, model, decay):
    for e, p in zip(ema.parameters(), model.parameters()):
        e.mul_(decay).add_(p.detach(), alpha=1 - decay)


def full_state(module):
    return {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
            for k, v in module.state_dict().items()}


@torch.no_grad()
def codec_eval(codec, bank, ids, out, size, device, deadline):
    """Future-seen AE only: RGB MSE at `size` (bilinear GT, as evaluate()), ownership accuracy (nearest GT)."""
    out.mkdir(parents=True, exist_ok=False)
    was = codec.training
    codec.eval()
    rows = []
    try:
        for identifier in ids:
            if time.time() >= deadline:
                raise TimeoutError('Codec evaluation deadline reached')
            rgb, lab, _ = bank.batch([identifier], device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                _, mu, _, _ = codec.encode_future(rgb, lab)
            video, owner = codec.render_latent(mu, size, owners=True)
            truth = F.interpolate(rgb[0].permute(0, 3, 1, 2).float() / 255, (size, size), mode='bilinear',
                                  align_corners=False)
            pick = torch.linspace(0, lab.shape[-1] - 1, size, device=device).round().long()
            gt = lab[0].long()[:, pick][:, :, pick]
            group = owner[0].clamp(max=codec.slots)
            err = (video[0] - truth).square().mean(1)
            obj = gt < codec.slots
            rows.append({'id': identifier, 'mse': float(err.mean()),
                         'object_pixel_mse': float(err[obj].mean()) if obj.any() else None,
                         'background_pixel_mse': float(err[~obj].mean()),
                         'ownership_accuracy': float((group == gt).float().mean()),
                         'object_ownership_recall': float((group == gt)[obj].float().mean()) if obj.any() else None,
                         'slot_info': bank.slot_info[identifier]})
        keys = ['mse', 'object_pixel_mse', 'background_pixel_mse', 'ownership_accuracy', 'object_ownership_recall']
        means = {k: float(np.mean([r[k] for r in rows if r[k] is not None])) for k in keys}
        result = {'status': 'complete', 'ids': ids, 'size': size, 'means': means, 'records': rows,
                  'scope': 'Future-seen reconstruction (AE upper bound); not generation.'}
        atomic_json(out / 'summary.json', result)
        return result
    finally:
        codec.train(was)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stage', choices=['codec', 'flow'], required=True)
    for name in ['data', 'cache', 'seg', 'cohorts', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--codec-checkpoint', type=Path)
    p.add_argument('--codec-sha')
    p.add_argument('--seed', type=int, default=713)
    p.add_argument('--batch-per-rank', type=int, default=8)
    p.add_argument('--points', type=int, default=1024)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--lr-min', type=float, default=3e-5)
    p.add_argument('--warmup-steps', type=int, default=1000)
    p.add_argument('--decay-end-hours', type=float, required=True)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--lambda-own', type=float, default=.01)
    p.add_argument('--kl', type=float, default=1e-5)
    p.add_argument('--hflip', type=float, default=0.)
    p.add_argument('--ema-decay', type=float, default=.9999)
    p.add_argument('--small-width', type=int, default=384)
    p.add_argument('--small-depth', type=int, default=8)
    p.add_argument('--small-heads', type=int, default=6)
    p.add_argument('--target-train-gpu-hours', type=float, required=True)
    p.add_argument('--allocated-gpu-hour-cap', type=float, required=True)
    p.add_argument('--wall-deadline-unix', type=float, required=True)
    p.add_argument('--milestones', default='')
    p.add_argument('--memory-cohort', default='pilot_train1024')
    p.add_argument('--memory-count', type=int, default=32)
    p.add_argument('--quick-eval-minutes', type=float, default=60.)
    p.add_argument('--save-minutes', type=float, default=30.)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--eval-size', type=int, default=64)
    p.add_argument('--eval-cohort', default='pilot32')
    p.add_argument('--eval-count', type=int, default=32)
    p.add_argument('--diverge-loss', type=float, default=.45)
    p.add_argument('--diverge-grad', type=float, default=5.)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--train-limit', type=int, default=0)
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args()
    a.milestone_hours = tuple(float(x) for x in a.milestones.split(',') if x.strip())

    dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo')
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device('cuda', local) if torch.cuda.is_available() else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    lead = rank == 0
    out, folder = a.out, a.out / 'checkpoints'
    if lead:
        if not a.resume and folder.exists() and any(folder.iterdir()):
            raise SystemExit('Nonempty checkpoint directory without --resume')
        folder.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    started = time.time()

    bank = orc.OracleBank(a.data, a.cache, a.seg)
    cohort_args = argparse.Namespace(mode='train', cohorts=a.cohorts, eval_cohort=a.eval_cohort,
                                     train_cohort='all', train_limit=a.train_limit, eval_count=a.eval_count)
    train_ids, eval_ids, showcase, cohorts = pick_cohorts(cohort_args, bank)
    document = json.loads(a.cohorts.read_text())
    memory_ids = [r['id'] for r in document[a.memory_cohort][:a.memory_count]] if a.memory_count else []
    if any(bank.records[i]['split'] != 'train' for i in memory_ids):
        raise ValueError('Memory cohort must be train clips')
    bank.preload(sorted(set(train_ids + eval_ids + showcase + memory_ids)))
    overflow = sum(bank.slot_info[i]['overflow_instances'] for i in train_ids)

    source_sha = {name: digest(Path(__file__).with_name(name)) for name in SOURCE_FILES}
    fixed = {'format': FORMAT, 'stage': a.stage, 'seed': a.seed, 'world_size': world,
             'batch_per_rank': a.batch_per_rank, 'global_batch': a.batch_per_rank * world,
             'lr': a.lr, 'lr_min': a.lr_min, 'warmup_steps': a.warmup_steps,
             'decay_end_train_gpu_hours': a.decay_end_hours, 'weight_decay': a.weight_decay,
             'grad_clip': 1.0, 'source_sha256': source_sha, 'cohorts_sha256': cohorts['cohorts_sha256'],
             'segmentation_log_sha256': digest(a.seg / 'clips.jsonl'), 'train_clips': len(train_ids),
             'data_manifest_sha256': bank.manifest_sha}
    torch.manual_seed(a.seed)
    if a.stage == 'codec':
        fixed.update(points=a.points, lambda_own=a.lambda_own, kl=a.kl, hflip=a.hflip,
                     query_sampling='half uniform future pixels, half uniform object pixels')
        codec = orc.OracleCodec().to(device)
        fixed['model_config'] = codec.config
        trainable = codec
        model_for_eval = None
    else:
        if digest(a.codec_checkpoint) != a.codec_sha:
            raise ValueError('Codec checkpoint SHA mismatch')
        ck = torch.load(a.codec_checkpoint, map_location='cpu', weights_only=False)
        if ck.get('format') != FORMAT or ck['protocol']['stage'] != 'codec':
            raise ValueError('Expected an oracle codec checkpoint')
        codec = orc.OracleCodec(**{k: ck['protocol']['model_config'][k] for k in
                                   ['slots', 'bg_side', 'latent_dim', 'subkernels', 'texture_kernels']}).to(device)
        codec.load_state_dict(ck['model'], strict=True)
        codec.requires_grad_(False).eval()
        torch.manual_seed(a.seed + 10001)
        flow = orc.OracleObjectFlow(codec.objects, codec.latent_dim, codec.config['feature_width'],
                                    codec.decoder.anchors, a.small_width, a.small_depth, a.small_heads).to(device)
        fixed.update(codec_sha256=a.codec_sha, small_architecture=[a.small_width, a.small_depth, a.small_heads],
                     ema_decay=a.ema_decay, flow='OracleObjectFlow (QK-norm StableObjectFlow + per-slot prefix tokens)',
                     eval_cohort=a.eval_cohort, eval_count=a.eval_count,
                     divergence_rule={'mean_loss_1000_steps_above': a.diverge_loss,
                                      'median_grad_500_steps_above': a.diverge_grad})
        trainable = flow
        model = orc.OracleFlowModel(codec, flow)
        ema_flow = copy.deepcopy(flow).requires_grad_(False)
        ema_model = orc.OracleFlowModel(codec, ema_flow)
    signature = canonical_sha(fixed)
    params = [q for q in trainable.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    state = {'step': 0, 'samples_seen': 0, 'train_gpu_seconds': 0., 'eval_seconds': 0.,
             'allocated_gpu_seconds': 0., 'milestones_done': []}
    data_rng = torch.Generator().manual_seed(a.seed + 101 + 1000 * rank)
    noise_rng = torch.Generator().manual_seed(a.seed + 202 + 1000 * rank)
    if a.resume:
        ckpt = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
        if ckpt.get('format') != FORMAT or ckpt.get('signature_sha256') != signature:
            raise ValueError('Resume checkpoint does not match this protocol signature')
        if a.stage == 'codec':
            codec.load_state_dict(ckpt['model'], strict=True)
        else:
            load_flow_state(flow, ckpt['flow'])
            ema_flow.load_state_dict(ckpt['ema_flow'], strict=True)
        opt.load_state_dict(ckpt['optimizer'])
        data_rng.set_state(ckpt['rank_rng'][rank]['data'])
        noise_rng.set_state(ckpt['rank_rng'][rank]['noise'])
        for key in state:
            state[key] = ckpt[key]
        del ckpt
    elif lead:
        atomic_json(out / 'protocol.json', {**fixed, 'signature_sha256': signature,
                    'trainable_parameters': sum(q.numel() for q in params), 'train_overflow_instances': overflow,
                    'gpu_hours_definition': 'train: step wall x world; allocated: session wall x world',
                    'torch_version': torch.__version__,
                    'device_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
    ddp = DDP(trainable, device_ids=[local] if device.type == 'cuda' else None)

    if a.stage == 'flow':
        # Posterior statistics of every train clip, unmirrored, computed identically on each rank.
        began = time.perf_counter()
        mus, logvars = [], []
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=a.amp):
            for i in range(0, len(train_ids), 32):
                rgb, lab, _ = bank.batch(train_ids[i:i + 32], device)
                _, mu, logvar, _ = codec.encode_future(rgb, lab)
                mus.append(mu.half())
                logvars.append(logvar.half())
        post_mu, post_logvar = torch.cat(mus), torch.cat(logvars)
        latent_cache_seconds = time.perf_counter() - began
        index_of = {c: i for i, c in enumerate(train_ids)}

    setup_seconds = time.time() - started
    session_start = time.perf_counter()
    prior_alloc = state['allocated_gpu_seconds']
    allocated = lambda: prior_alloc + (time.perf_counter() - session_start) * world  # noqa: E731
    target, cap = a.target_train_gpu_hours * 3600., a.allocated_gpu_hour_cap * 3600.
    last_quick = last_save = time.perf_counter()
    recent_loss = collections.deque(maxlen=1000)
    recent_grad = collections.deque(maxlen=500)

    def ledger(event):
        if lead:
            with (out / 'budget_ledger.jsonl').open('a') as stream:
                stream.write(json.dumps({'event': event, 'unix': time.time(), 'step': state['step'],
                                         'train_gpu_hours_cumulative': state['train_gpu_seconds'] / 3600,
                                         'allocated_gpu_hours_this_run': allocated() / 3600,
                                         'world_size': world}) + '\n')

    def status(extra):
        if lead:
            atomic_json(out / 'status.json', {
                'pid': os.getpid(), 'mode': 'oracle_' + a.stage, 'world_size': world,
                'signature_sha256': signature, 'step': state['step'], 'samples_seen': state['samples_seen'],
                'train_hours_cumulative': state['train_gpu_seconds'] / 3600,
                'allocated_hours_this_run': allocated() / 3600,
                'eval_hours_cumulative': state['eval_seconds'] / 3600,
                'milestones_done': state['milestones_done'], 'lr': opt.param_groups[0]['lr'],
                'updated_unix': time.time(), 'setup_seconds': setup_seconds, **memory_peaks(device), **extra})

    def run_eval(which, ids, tag):
        result = None
        if lead:
            began = time.perf_counter()
            try:
                path = out / 'eval' / f'{tag}_{which}_{state["step"]:07d}'
                if a.stage == 'codec':
                    result = codec_eval(codec, bank, ids, path, a.eval_size, device, a.wall_deadline_unix)
                else:
                    result = evaluate(ema_model if which == 'ema' else model, bank, ids,
                                      showcase if ids is eval_ids else [], path, a.eval_size,
                                      state['step'], device, a.amp, a.wall_deadline_unix)
            finally:
                sync(device)
                state['eval_seconds'] += time.perf_counter() - began
        dist.barrier()
        return result

    def quick_record(tag, result):
        if lead:
            with (out / 'quick_history.jsonl').open('a') as stream:
                stream.write(json.dumps({'step': state['step'], 'which': tag, 'means': result['means'],
                                         'train_gpu_hours': state['train_gpu_seconds'] / 3600,
                                         'unix_time': time.time()}, allow_nan=False) + '\n')

    def save(name):
        states = [None] * world
        dist.all_gather_object(states, {'data': data_rng.get_state(), 'noise': noise_rng.get_state()})
        if lead:
            state['allocated_gpu_seconds'] = allocated()
            payload = {'format': FORMAT, 'signature_sha256': signature, 'protocol': fixed,
                       'optimizer': opt.state_dict(), 'rank_rng': states, **state}
            if a.stage == 'codec':
                payload['model'] = full_state(codec)
            else:
                payload.update(flow=flow_state(flow), ema_flow=full_state(ema_flow))
            path = folder / f'{name}.pt'
            tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
            torch.save(payload, tmp)
            os.replace(tmp, path)
            link = folder / f'latest.{os.getpid()}.tmp'
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(path.name, link)
            os.replace(link, folder / 'latest.pt')
            atomic_json(folder / 'latest.json', {'name': name, 'step': state['step'], 'sha256': digest(path),
                                                 'train_seconds': state['train_gpu_seconds'],
                                                 'allocated_seconds': state['allocated_gpu_seconds']})
            for old in sorted(folder.glob('step_*.pt'))[:-2]:
                old.unlink()
        dist.barrier()

    ledger('resume' if a.resume else 'start')
    status({'phase': 'training', **({'latent_cache_seconds': latent_cache_seconds} if a.stage == 'flow' else {})})
    trainable.train()
    reason = None
    try:
        while True:
            decision = torch.zeros(3, dtype=torch.long, device=device)
            if lead:
                code = CONTINUE
                if (out / 'REQUEST_STOP').exists():
                    code = STOP
                elif (out / 'REQUEST_PAUSE').exists():
                    code = PAUSE
                elif state['train_gpu_seconds'] >= target:
                    code = TARGET
                elif allocated() >= cap:
                    code = CAP
                elif time.time() >= a.wall_deadline_unix:
                    code = DEADLINE
                elif a.stage == 'flow' and state['step'] > 20000 and (
                        (len(recent_loss) == recent_loss.maxlen and statistics.fmean(recent_loss) > a.diverge_loss) or
                        (len(recent_grad) == recent_grad.maxlen and statistics.median(recent_grad) > a.diverge_grad)):
                    code = DIVERGED
                due = [i for i, h in enumerate(a.milestone_hours)
                       if h * 3600 <= state['train_gpu_seconds'] and h not in state['milestones_done']]
                decision[0] = code
                decision[1] = due[0] + 1 if due else 0
                decision[2] = (1 if time.perf_counter() - last_quick >= a.quick_eval_minutes * 60 else 0) \
                    + (2 if time.perf_counter() - last_save >= a.save_minutes * 60 else 0)
            dist.broadcast(decision, 0)
            code, milestone, periodic = (int(x) for x in decision.tolist())
            if code != CONTINUE:
                reason = REASONS[code]
                break
            if milestone:
                h = a.milestone_hours[milestone - 1]
                for which in (['codec'] if a.stage == 'codec' else ['ema', 'raw']):
                    run_eval(which, eval_ids, f'milestone_{h:g}h')
                if a.stage == 'flow' and memory_ids:
                    run_eval('ema', memory_ids, f'memory_{h:g}h')
                state['milestones_done'].append(h)
                save(f'milestone_{h:g}h')
                ledger(f'milestone_{h:g}h')
            if periodic & 1:
                result = run_eval('codec' if a.stage == 'codec' else 'ema', showcase, 'quick')
                if lead:
                    quick_record('codec' if a.stage == 'codec' else 'ema', result)
                ledger('quick_eval')
                last_quick = time.perf_counter()
            if periodic & 2:
                save(f'step_{state["step"]:07d}')
                ledger('checkpoint')
                last_save = time.perf_counter()
            lr = lr_at(a, state['step'], state['train_gpu_seconds'])
            lr_t = torch.tensor([lr], dtype=torch.float64, device=device)
            dist.broadcast(lr_t, 0)
            for group in opt.param_groups:
                group['lr'] = float(lr_t)
            ids = [train_ids[i] for i in torch.randint(len(train_ids), (a.batch_per_rank,),
                                                        generator=data_rng).tolist()]
            sync(device)
            began = time.perf_counter()
            extra = {}
            if a.stage == 'codec':
                flips = torch.rand(a.batch_per_rank, generator=data_rng) < a.hflip
                rgb, lab = bank.clip_tensors(ids, device, flips)
                coords, target_rgb, labels = orc.query_batch(rgb[:, 4:], lab[:, 4:], a.points, data_rng)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=a.amp):
                    parts = ddp(rgb[:, 4:], lab[:, 4:], coords, target_rgb, labels)
                loss = parts['recon'] + parts['routed'] + a.lambda_own * parts['ownership'] + a.kl * parts['kl']
                extra = {k: float(v.detach()) for k, v in parts.items()}
            else:
                rgb, lab = bank.clip_tensors(ids, device)
                rows = torch.tensor([index_of[i] for i in ids], device=device)
                eps = torch.randn(a.batch_per_rank, codec.objects, codec.latent_dim, generator=noise_rng).to(device)
                target_z = post_mu[rows].float() + (.5 * post_logvar[rows].float()).exp() * eps
                noise = torch.randn(a.batch_per_rank, codec.objects, codec.latent_dim, generator=noise_rng).to(device)
                t = torch.rand(a.batch_per_rank, generator=noise_rng).to(device)
                zt = (1 - t[:, None, None]) * noise + t[:, None, None] * target_z
                velocity = ddp(zt, t, (rgb[:, :4], lab[:, :4])).float()
                loss = F.mse_loss(velocity, target_z - noise)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            opt.step()
            if a.stage == 'flow':
                ema_update(ema_flow, flow, a.ema_decay)
            sync(device)
            duration = time.perf_counter() - began
            state['step'] += 1
            state['samples_seen'] += a.batch_per_rank * world
            state['train_gpu_seconds'] += duration * world
            if lead:
                recent_loss.append(loss.item())
                recent_grad.append(grad.item())
                if state['step'] % a.log_every == 0:
                    row = {'step': state['step'], 'samples_seen': state['samples_seen'], 'loss': recent_loss[-1],
                           'grad_norm': recent_grad[-1], 'lr': float(lr_t), 'step_seconds': duration,
                           'world_size': world, 'train_gpu_hours_cumulative': state['train_gpu_seconds'] / 3600,
                           'allocated_gpu_hours_this_run': allocated() / 3600, 'unix_time': time.time(), **extra}
                    with (out / 'metrics.jsonl').open('a') as stream:
                        stream.write(json.dumps(row, allow_nan=False) + '\n')
                    status({'phase': 'training'})
    except Exception as exc:
        status({'phase': 'failed', 'error': type(exc).__name__ + ': ' + str(exc)})
        ledger('failed:' + type(exc).__name__)
        raise

    save({'target_train_gpu_hours_reached': 'final', 'requested_stop': 'final',
          'diverged': 'diverged'}.get(reason, 'interrupted'))
    final_done = False
    if reason in {'target_train_gpu_hours_reached', 'requested_stop', 'allocated_gpu_hour_cap'} \
            and time.time() < a.wall_deadline_unix:
        for which in (['codec'] if a.stage == 'codec' else ['ema', 'raw']):
            run_eval(which, eval_ids, 'final')
        if a.stage == 'flow' and memory_ids:
            run_eval('ema', memory_ids, 'memory_final')
        final_done = True
    status({'phase': 'stopped', 'reason': reason,
            'status': 'complete' if reason in {'target_train_gpu_hours_reached', 'requested_stop'} else reason,
            'final_eval_done': final_done})
    ledger('exit:' + reason)
    if lead:
        print(json.dumps({'reason': reason, 'step': state['step'],
                          'train_gpu_hours': state['train_gpu_seconds'] / 3600,
                          'allocated_gpu_hours_this_run': allocated() / 3600}), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

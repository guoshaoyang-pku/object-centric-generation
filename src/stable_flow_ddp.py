"""Concentrated frozen-codec flow training, stabilized after the 10-27 divergence.

Same objective, data stream and evaluation as frozen_flow_ddp.py (flow matching
only; frozen C2 codec; detached posterior-sample target; detached prefix
context; single-node DDP; EMA flow as the reported model). The changes:

- The flow is StableObjectFlow (stable_flow.py): QK-normalized attention. A
  branch from a plain SmallObjectFlow checkpoint loads every old weight, then
  calibrates the new gains on one fixed batch so typical logits keep their
  scale. The optimizer is a fresh AdamW, because the parameter set changed.
- Learning rate: linear warmup, then cosine decay from --lr to --lr-min over
  the training GPU-hours between the branch point and --decay-end-hours. The
  stop target is not part of the protocol, so it can be extended on resume.
- The EMA flow is saved in full. (frozen_flow_ddp.py saved only its buffers,
  because the EMA copy had requires_grad off; its EMA was never checkpointed.)
- A divergence watchdog stops the run instead of training a collapsed model:
  mean flow loss over the last 1000 steps above --diverge-loss, or median
  pre-clip gradient norm over the last 500 steps above --diverge-grad.
- Pinned checkpoints every --pin-every-hours are never rotated away.

Control files in the output directory: REQUEST_PAUSE saves and exits;
REQUEST_STOP saves, runs the final pilot32 eval and exits.

Launch: python -m torch.distributed.run --standalone --nproc_per_node=K stable_flow_ddp.py ...
"""
import argparse
import collections
import copy
import datetime as dt
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joint_flow as jf  # noqa: E402  (also puts the code_v4 base on sys.path)
from codec_research import CacheBank, digest, canonical_sha, atomic_json, sync, pick_cohorts  # noqa: E402
from codec_flow_research import evaluate, flow_state, load_flow_state, prefix_batch, memory_peaks  # noqa: E402
from stable_flow import StableObjectFlow, calibrate_gains, warm_start_from_small  # noqa: E402

FORMAT = 'claude_stable_flow_ddp_v1'
SOURCE_FILES = ['stable_flow_ddp.py', 'stable_flow.py', 'joint_flow.py']
# Rank decisions broadcast each step.
CONTINUE, STOP, PAUSE, TARGET, CAP, DEADLINE, DIVERGED = range(7)
REASONS = {STOP: 'requested_stop', PAUSE: 'paused', TARGET: 'target_train_gpu_hours_reached',
           CAP: 'allocated_gpu_hour_cap', DEADLINE: 'wall_deadline', DIVERGED: 'diverged'}


def protocol_fields(a, world, source_sha, cohort_sha, branch_sha):
    """Everything that defines the experiment; budget caps are excluded."""
    return {'format': FORMAT, 'mode': 'frozen', 'seed': a.seed, 'world_size': world,
            'batch_per_rank': a.batch_per_rank, 'global_batch': a.batch_per_rank * world,
            'lr': a.lr, 'lr_min': a.lr_min, 'warmup_steps': a.warmup_steps,
            'lr_schedule': 'linear warmup, then cosine over training GPU-hours from branch to decay end, then lr_min',
            'decay_end_train_gpu_hours': a.decay_end_hours,
            'weight_decay': a.weight_decay, 'ema_decay': a.ema_decay, 'grad_clip': 1.0,
            'flow': 'StableObjectFlow (QK-normalized attention)',
            'small_architecture': [a.small_width, a.small_depth, a.small_heads],
            'divergence_rule': {'mean_loss_1000_steps_above': a.diverge_loss,
                                'median_grad_500_steps_above': a.diverge_grad},
            'codec_sha256': a.codec_sha, 'branch_checkpoint_sha256': branch_sha,
            'cohorts_sha256': cohort_sha, 'eval_cohort': a.eval_cohort, 'eval_count': a.eval_count,
            'source_sha256': source_sha}


def full_state(module):
    return {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
            for k, v in module.state_dict().items()}


@torch.no_grad()
def ema_update(ema, model, decay):
    for e, p in zip(ema.parameters(), model.parameters()):
        e.mul_(decay).add_(p.detach(), alpha=1 - decay)


def lr_at(a, state):
    if state['stable_steps'] < a.warmup_steps:
        return a.lr * (state['stable_steps'] + 1) / a.warmup_steps
    start = state['start_train_gpu_seconds']
    span = max(a.decay_end_hours * 3600. - start, 1.)
    frac = min(max((state['train_gpu_seconds'] - start) / span, 0.), 1.)
    return a.lr_min + .5 * (a.lr - a.lr_min) * (1 + math.cos(math.pi * frac))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ['data', 'cache', 'out', 'codec-checkpoint', 'cohorts', 'branch-from']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--codec-sha', required=True)
    p.add_argument('--branch-from-sha', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--seed', type=int, default=713)
    p.add_argument('--batch-per-rank', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lr-min', type=float, default=1e-5)
    p.add_argument('--warmup-steps', type=int, default=2000)
    p.add_argument('--decay-end-hours', type=float, required=True,
                   help='Cumulative training GPU-hours where the cosine reaches --lr-min; a later target trains on at lr-min')
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--ema-decay', type=float, default=.9999)
    p.add_argument('--small-width', type=int, default=384)
    p.add_argument('--small-depth', type=int, default=8)
    p.add_argument('--small-heads', type=int, default=6)
    p.add_argument('--eval-cohort', choices=['pilot32', 'confirmation64'], default='pilot32')
    p.add_argument('--eval-count', type=int, default=32)
    p.add_argument('--eval-size', type=int, default=64)
    p.add_argument('--target-train-gpu-hours', type=float, required=True,
                   help='Cumulative training GPU-hours, including those inherited at the branch point')
    p.add_argument('--allocated-gpu-hour-cap', type=float, required=True,
                   help='Cap on this run: session wall time x world size, summed over sessions')
    p.add_argument('--wall-deadline-unix', type=float, required=True)
    p.add_argument('--quick-eval-minutes', type=float, default=60.)
    p.add_argument('--save-minutes', type=float, default=30.)
    p.add_argument('--keep-checkpoints', type=int, default=3)
    p.add_argument('--pin-every-hours', type=float, default=2.)
    p.add_argument('--diverge-loss', type=float, default=.45)
    p.add_argument('--diverge-grad', type=float, default=5.)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--milestones', default='4,10,20,30,40,50')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args()
    a.milestone_hours = tuple(float(x) for x in a.milestones.split(',') if x.strip())

    dist.init_process_group('nccl', timeout=dt.timedelta(minutes=60))
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ['LOCAL_RANK'])
    device = torch.device('cuda', local)
    torch.cuda.set_device(device)
    torch.set_num_threads(a.threads)
    lead = rank == 0
    out = a.out
    folder = out / 'checkpoints'
    if lead:
        if not a.resume and folder.exists() and any(folder.iterdir()):
            raise SystemExit('Nonempty checkpoint directory without --resume')
        out.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    started = time.time()
    source_sha = {name: digest(Path(__file__).with_name(name)) for name in SOURCE_FILES}
    bank = CacheBank(a.data, a.cache)
    cohort_args = argparse.Namespace(mode='train', cohorts=a.cohorts, eval_cohort=a.eval_cohort,
                                     train_cohort='all', train_limit=0, eval_count=a.eval_count)
    train_ids, eval_ids, showcase, cohorts = pick_cohorts(cohort_args, bank)
    fixed = protocol_fields(a, world, source_sha, cohorts['cohorts_sha256'], a.branch_from_sha)
    signature = canonical_sha(fixed)
    build_args = argparse.Namespace(codec_checkpoint=a.codec_checkpoint, codec_sha=a.codec_sha, seed=a.seed,
                                    small_width=a.small_width, small_depth=a.small_depth,
                                    small_heads=a.small_heads, mode='frozen')
    model, codec_identity = jf.build(build_args, bank, device)
    codec = model.codec
    flow = StableObjectFlow(codec.objects, codec.latent_dim, codec.config['feature_width'],
                            codec.decoder.anchors, a.small_width, a.small_depth, a.small_heads).to(device)
    model.flow = flow
    state = {'step': 0, 'samples_seen': 0, 'train_gpu_seconds': 0., 'eval_seconds': 0.,
             'allocated_gpu_seconds': 0., 'milestones_done': [], 'pins_done': [],
             'stable_steps': 0, 'start_train_gpu_seconds': 0.}
    params = [q for q in flow.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    branch_meta = None
    if a.resume:
        ckpt = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
        if ckpt.get('format') != FORMAT or ckpt.get('signature_sha256') != signature:
            raise ValueError('Resume checkpoint does not match this protocol signature')
        load_flow_state(flow, ckpt['flow'])
        opt.load_state_dict(ckpt['optimizer'])
        ema_state = ckpt['ema_flow']
        rank_rng = ckpt['rank_rng'][rank]
        for key in state:
            state[key] = ckpt[key]
    else:
        if digest(a.branch_from) != a.branch_from_sha:
            raise ValueError('Branch checkpoint SHA mismatch')
        ckpt = torch.load(a.branch_from, map_location='cpu', weights_only=False)
        if ckpt.get('format') != jf.FORMAT or ckpt['protocol']['mode'] != 'frozen':
            raise ValueError('Branch point must be a frozen joint_flow.py checkpoint')
        new_params = warm_start_from_small(flow, ckpt['flow'])
        # One fixed batch, identical on every rank, so every rank calibrates the same gains.
        calib_ids = train_ids[:8]
        g = torch.Generator().manual_seed(a.seed + 4242)
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=a.amp):
            context = model.prefix_context(*prefix_batch(bank, calib_ids, device))
            z1, _, _, _ = codec.encode_future(*bank.batch(calib_ids, device))
        noise = torch.randn(len(calib_ids), codec.objects, codec.latent_dim, generator=g).to(device)
        t = torch.rand(len(calib_ids), generator=g).to(device)
        zt = (1 - t[:, None, None]) * noise + t[:, None, None] * z1.float()
        reference = jf.SmallObjectFlow(codec.objects, codec.latent_dim, codec.config['feature_width'],
                                       codec.decoder.anchors, a.small_width, a.small_depth, a.small_heads)
        load_flow_state(reference, ckpt['flow'])
        with torch.no_grad():
            expected = reference.to(device).eval()(zt, t, context).float()
        measured, gains = calibrate_gains(flow.eval(), zt, t, context)
        identity_error = float((measured.float() - expected).abs().max())
        if identity_error > 5e-2 * float(expected.abs().max()) + 1e-3:
            raise ValueError(f'QK attention does not reproduce the warm-start flow: max error {identity_error}')
        del reference
        ema_state = None
        rank_rng = None
        state.update(step=int(ckpt['step']), samples_seen=int(ckpt['samples_seen']),
                     train_gpu_seconds=float(ckpt['train_seconds']),
                     start_train_gpu_seconds=float(ckpt['train_seconds']))
        branch_meta = {'branch_step': int(ckpt['step']), 'branch_train_seconds': float(ckpt['train_seconds']),
                       'branch_signature': ckpt['signature_sha256'], 'new_parameters': new_params,
                       'optimizer': 'fresh AdamW (parameter set changed)',
                       'calibration': {'ids': calib_ids, 'gains': gains,
                                       'identity_pass_max_abs_error': identity_error}}
    del ckpt
    ema_flow = copy.deepcopy(flow).requires_grad_(False)
    if ema_state is not None:
        ema_flow.load_state_dict(ema_state, strict=True)
    ema_model = jf.JointFlow(codec, ema_flow, False)
    ddp_flow = DDP(flow, device_ids=[local], broadcast_buffers=False)
    data_rng = torch.Generator().manual_seed(a.seed + 101 + 7919 * (rank + 1))
    noise_rng = torch.Generator().manual_seed(a.seed + 303 + 7919 * (rank + 1))
    if rank_rng is not None:
        data_rng.set_state(rank_rng['data'])
        noise_rng.set_state(rank_rng['noise'])
    if lead and not a.resume:
        atomic_json(out / 'protocol.json', {**fixed, 'signature_sha256': signature,
                    'codec_identity': codec_identity, 'branch': {**branch_meta, 'path': str(a.branch_from)},
                    'flow_trainable_parameters': sum(q.numel() for q in params),
                    'gpu_hours_definition': 'train: step wall x world; allocated: session wall x world',
                    'torch_version': torch.__version__,
                    'device_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})

    setup_seconds = time.time() - started
    session_start = time.perf_counter()
    prior_alloc = state['allocated_gpu_seconds']
    allocated = lambda: prior_alloc + (time.perf_counter() - session_start) * world  # noqa: E731
    target = a.target_train_gpu_hours * 3600.
    cap = a.allocated_gpu_hour_cap * 3600.
    last_quick = last_save = time.perf_counter()
    pid_started = time.ctime()
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
                'pid': os.getpid(), 'pid_started': pid_started, 'mode': 'stable_ddp', 'world_size': world,
                'signature_sha256': signature, 'step': state['step'], 'samples_seen': state['samples_seen'],
                'train_hours_cumulative': state['train_gpu_seconds'] / 3600,
                'allocated_hours_this_run': allocated() / 3600,
                'eval_hours_cumulative': state['eval_seconds'] / 3600,
                'milestones_done': state['milestones_done'], 'pins_done': state['pins_done'],
                'lr': opt.param_groups[0]['lr'], 'updated_unix': time.time(),
                **memory_peaks(device), **extra})

    def run_eval(which, ids, tag):
        """Rank 0 evaluates; the others wait at the barrier."""
        result = None
        if lead:
            began = time.perf_counter()
            try:
                result = evaluate(ema_model if which == 'ema' else model, bank, ids, showcase,
                                  out / 'eval' / f'{tag}_{which}_{state["step"]:07d}', a.eval_size,
                                  state['step'], device, a.amp, a.wall_deadline_unix)
            finally:
                sync(device)
                state['eval_seconds'] += time.perf_counter() - began
        dist.barrier()
        return result

    def save(name):
        states = [None] * world
        dist.all_gather_object(states, {'data': data_rng.get_state(), 'noise': noise_rng.get_state()})
        checksum = torch.stack([q.detach().float().sum() for q in params]).sum().reshape(1)
        sums = [torch.zeros_like(checksum) for _ in range(world)]
        dist.all_gather(sums, checksum)
        if lead:
            spread = float(torch.cat(sums).max() - torch.cat(sums).min())
            if spread != 0.:
                raise RuntimeError(f'Ranks diverged: parameter checksum spread {spread}')
            state['allocated_gpu_seconds'] = allocated()
            payload = {'format': FORMAT, 'signature_sha256': signature, 'protocol': fixed,
                       'flow': flow_state(flow), 'ema_flow': full_state(ema_flow),
                       'optimizer': opt.state_dict(), 'rank_rng': states, **state,
                       'setup_seconds': setup_seconds}
            jf.save_checkpoint(folder, name, {**payload, 'train_seconds': state['train_gpu_seconds'],
                                              'allocated_seconds': state['allocated_gpu_seconds']},
                               a.keep_checkpoints)
        dist.barrier()

    def quick_record(tag, result):
        if lead:
            with (out / 'quick_history.jsonl').open('a') as stream:
                stream.write(json.dumps({'step': state['step'], 'which': tag, 'means': result['means'],
                                         'train_gpu_hours': state['train_gpu_seconds'] / 3600,
                                         'unix_time': time.time()}, allow_nan=False) + '\n')

    ledger('resume' if a.resume else 'start_from_branch')
    if not a.resume:
        quick_record('raw_step0', run_eval('raw', showcase, 'quick_step0'))
    flow.train()
    reason = None
    try:
        while True:
            decision = torch.zeros(4, dtype=torch.long, device=device)
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
                elif (len(recent_loss) == recent_loss.maxlen and statistics.fmean(recent_loss) > a.diverge_loss) or \
                        (len(recent_grad) == recent_grad.maxlen and statistics.median(recent_grad) > a.diverge_grad):
                    code = DIVERGED
                due = [i for i, h in enumerate(a.milestone_hours)
                       if h * 3600 <= state['train_gpu_seconds'] and h not in state['milestones_done']]
                pin = int(state['train_gpu_seconds'] // (a.pin_every_hours * 3600))
                decision[0] = code
                decision[1] = due[0] + 1 if due else 0
                decision[2] = (1 if time.perf_counter() - last_quick >= a.quick_eval_minutes * 60 else 0) \
                    + (2 if time.perf_counter() - last_save >= a.save_minutes * 60 else 0)
                decision[3] = pin if pin > 0 and pin * a.pin_every_hours not in state['pins_done'] else 0
            dist.broadcast(decision, 0)
            code, milestone, periodic, pin = (int(x) for x in decision.tolist())
            if code != CONTINUE:
                reason = REASONS[code]
                break
            if milestone:
                h = a.milestone_hours[milestone - 1]
                for which in ['ema', 'raw']:
                    run_eval(which, eval_ids, f'milestone_{h:g}h')
                state['milestones_done'].append(h)
                save(f'milestone_{h:g}h')
                ledger(f'milestone_{h:g}h')
            if pin:
                h = pin * a.pin_every_hours
                state['pins_done'].append(h)
                save(f'pin_{h:g}h')
                ledger(f'pin_{h:g}h')
            if periodic & 1:
                quick_record('ema', run_eval('ema', showcase, 'quick'))
                ledger('quick_eval')
                last_quick = time.perf_counter()
            if periodic & 2:
                save(f'step_{state["step"]:07d}')
                ledger('checkpoint')
                last_save = time.perf_counter()
            # Each rank times its own steps, so train_gpu_seconds differs slightly between ranks;
            # rank 0's schedule value is the one every rank applies.
            lr_t = torch.tensor([lr_at(a, state)], dtype=torch.float64, device=device)
            dist.broadcast(lr_t, 0)
            lr = float(lr_t)
            for group in opt.param_groups:
                group['lr'] = lr
            ids = [train_ids[i] for i in torch.randint(len(train_ids), (a.batch_per_rank,),
                                                        generator=data_rng).tolist()]
            sync(device)
            began = time.perf_counter()
            prefix = prefix_batch(bank, ids, device)
            future = bank.batch(ids, device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=a.amp):
                context = model.prefix_context(*prefix)
                with torch.no_grad():
                    z1, _, _, _ = codec.encode_future(*future)
            target_z = z1.detach().float()
            noise = torch.randn(a.batch_per_rank, model.objects, model.latent_dim, generator=noise_rng).to(device)
            t = torch.rand(a.batch_per_rank, generator=noise_rng).to(device)
            zt = (1 - t[:, None, None]) * noise + t[:, None, None] * target_z
            velocity = ddp_flow(zt, t, context).float()
            loss = F.mse_loss(velocity, target_z - noise)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite flow loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            opt.step()
            ema_update(ema_flow, flow, a.ema_decay)
            sync(device)
            duration = time.perf_counter() - began
            state['step'] += 1
            state['stable_steps'] += 1
            state['samples_seen'] += a.batch_per_rank * world
            state['train_gpu_seconds'] += duration * world
            if lead:
                recent_loss.append(loss.item())
                recent_grad.append(grad.item())
                if state['step'] % a.log_every == 0:
                    row = {'step': state['step'], 'samples_seen': state['samples_seen'], 'flow': recent_loss[-1],
                           'grad_norm': recent_grad[-1], 'lr': lr, 'step_seconds': duration, 'world_size': world,
                           'train_gpu_hours_cumulative': state['train_gpu_seconds'] / 3600,
                           'allocated_gpu_hours_this_run': allocated() / 3600, 'unix_time': time.time()}
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
        for which in ['ema', 'raw']:
            run_eval(which, eval_ids, 'final')
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

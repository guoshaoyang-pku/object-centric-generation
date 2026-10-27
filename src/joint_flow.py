"""Continue the 20261027 small object flow: frozen-codec control or joint codec.

Lineage: the cached VAE features, cohorts and RNG streams of
flow_small_c2_initial_s713_900s, continued from its final.pt (flow weights,
AdamW state and RNG streams). Both modes share the flow objective exactly.

- frozen: codec frozen; flow matching loss only. This is the control.
- joint:  codec encoder and decoder trainable. Flow target is the detached
          posterior sample. The codec gets L_AE on sampled future RGB queries
          plus KL. An optional matched-denoise term renders the flow's denoised
          estimate; its gradient reaches the flow and the decoder, not the encoder.

Prefix leakage: the flow conditions only on prefix_context(prefix features),
which is detached. Future features feed only the posterior target path, and
future RGB feeds only the training labels. Sampling never calls the encoder.

Budget: train_seconds is cumulative training-loop time and includes the
inherited 900.009 s. Eval, setup and checkpoint time count only toward
allocated time. Caps: cumulative training hours, allocated hours and a wall
deadline. Control files in the output directory: REQUEST_PAUSE saves and exits;
REQUEST_STOP saves, runs the pilot32 eval, and exits.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

CODE_BASE = Path(os.environ.get('PCV_CODE_BASE', Path(__file__).resolve().parent / 'code_base_v4'))
sys.path.insert(0, str(CODE_BASE))
from codec_research import (CacheBank, digest, canonical_sha, atomic_json, sync, rng_state,  # noqa: E402
                            restore_rng, sampled_queries, pick_cohorts)
from codec_flow_research import (load_codec, SmallObjectFlow, ConditionalObjectFlow, evaluate,  # noqa: E402
                                 flow_state, load_flow_state, prefix_batch, memory_peaks)

FORMAT = 'claude_joint_flow_v1'
INIT_FORMAT = 'codec_flow_research_v4'
INIT_PROTOCOL_SHA = '8a595f4b206b926a543be9a50fdf4bd9b8d73f5a290d985e1b1badb68261a36d'
INHERITED_TRAIN_SECONDS = 900.009142
MILESTONE_HOURS = (1., 4., 10., 25., 50.)
SOURCE_FILES = ['joint_flow.py']


class JointFlow(ConditionalObjectFlow):
    """Same interface as ConditionalObjectFlow; the codec may be trainable."""

    def __init__(self, codec, flow, trainable_codec):
        super().__init__(codec, flow)
        self.trainable_codec = bool(trainable_codec)
        codec.requires_grad_(self.trainable_codec)
        if self.trainable_codec:
            codec.config['freeze_decoder'] = False
            codec.decoder.requires_grad_(True)
        self.codec.train(self.trainable_codec)

    def train(self, mode=True):
        nn.Module.train(self, mode)
        self.codec.train(bool(mode) and self.trainable_codec)
        return self


def build(args, bank, device):
    codec, codec_identity = load_codec(args.codec_checkpoint, bank, args.codec_sha)
    torch.manual_seed(args.seed + 10001)
    flow = SmallObjectFlow(codec.objects, codec.latent_dim, codec.config['feature_width'],
                           codec.decoder.anchors, args.small_width, args.small_depth, args.small_heads)
    return JointFlow(codec, flow, args.mode == 'joint').to(device), codec_identity


def codec_groups(model, lr_encoder, lr_decoder):
    """Encoder moves the flow target, so it gets the slower LR; decoder adapts rendering."""
    encoder = [q for n, q in model.codec.named_parameters() if q.requires_grad and not n.startswith('decoder.')]
    decoder = [q for n, q in model.codec.named_parameters() if q.requires_grad and n.startswith('decoder.')]
    return [{'params': encoder, 'lr': lr_encoder}, {'params': decoder, 'lr': lr_decoder}], encoder + decoder


def protocol_fields(args, source_sha, cohort_sha):
    """Everything that defines the experiment. Budget caps are excluded on purpose."""
    return {
        'format': FORMAT,
        'mode': args.mode,
        'seed': args.seed,
        'batch': args.batch,
        'points': args.points,
        'lr_flow': args.lr_flow,
        'lr_encoder': args.lr_encoder,
        'lr_decoder': args.lr_decoder,
        'weight_decay': args.weight_decay,
        'kl': args.kl,
        'lambda_ae': args.lambda_ae,
        'lambda_match': args.lambda_match,
        'small_architecture': [args.small_width, args.small_depth, args.small_heads],
        'init_flow_sha256': args.init_flow_sha,
        'codec_sha256': args.codec_sha,
        'cohorts_sha256': cohort_sha,
        'train_cohort': args.train_cohort,
        'train_limit': args.train_limit,
        'eval_cohort': args.eval_cohort,
        'eval_count': args.eval_count,
        'source_sha256': source_sha,
    }


def training_step(model, bank, ids, args, device, gens, opt_flow, opt_codec, flow_params, codec_params):
    data_rng, query_rng, noise_rng = gens
    prefix = prefix_batch(bank, ids, device)
    future = bank.batch(ids, device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=args.amp and device.type == 'cuda'):
        context = model.prefix_context(*prefix)
        if model.trainable_codec:
            z1, _, _, kl = model.codec.encode_future(*future)
        else:
            with torch.no_grad():
                z1, _, _, kl = model.codec.encode_future(*future)
    target = z1.detach().float()
    noise = torch.randn(args.batch, model.objects, model.latent_dim, generator=noise_rng).to(device)
    t = torch.rand(args.batch, generator=noise_rng).to(device)
    zt = (1 - t[:, None, None]) * noise + t[:, None, None] * target
    velocity = model.flow(zt, t, context).float()
    flow_loss = F.mse_loss(velocity, target - noise)
    total = flow_loss
    zero = flow_loss.new_zeros(())
    ae_loss = match_loss = kl_loss = zero
    if model.trainable_codec:
        coords, truth = sampled_queries(bank, ids, args.points, query_rng, device)
        predicted = model.codec.render_queries(z1.float(), coords)
        ae_loss = F.mse_loss(predicted, truth)
        kl_loss = kl.float()
        if args.lambda_match > 0:
            estimate = zt + (1 - t[:, None, None]) * velocity
            rendered = model.codec.render_queries(estimate, coords)
            per_clip = (rendered - truth).square().mean((1, 2))
            match_loss = (t * per_clip).mean()
        total = flow_loss + args.lambda_ae * ae_loss + args.lambda_match * match_loss + args.kl * kl_loss
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite total loss')
    opt_flow.zero_grad(set_to_none=True)
    if opt_codec is not None:
        opt_codec.zero_grad(set_to_none=True)
    total.backward()
    grad_flow = torch.nn.utils.clip_grad_norm_(flow_params, 1., error_if_nonfinite=True)
    grad_codec = zero
    opt_flow.step()
    if opt_codec is not None:
        grad_codec = torch.nn.utils.clip_grad_norm_(codec_params, 1., error_if_nonfinite=True)
        opt_codec.step()
    return {'total': total.detach(), 'flow': flow_loss.detach(), 'ae': ae_loss.detach(),
            'match': match_loss.detach(), 'kl': kl_loss.detach(),
            'grad_flow': grad_flow.detach(), 'grad_codec': grad_codec.detach()}


@torch.no_grad()
def posterior_means(model, bank, ids, device):
    return [model.codec.encode_future(*bank.batch([i], device))[1][0].detach().float().cpu() for i in ids]


@torch.no_grad()
def codec_drift(model, bank, ids, reference, device):
    """Relative posterior-mean drift of the showcase clips from the starting codec."""
    values = []
    for identifier, ref in zip(ids, reference):
        mu = model.codec.encode_future(*bank.batch([identifier], device))[1][0].float().cpu()
        values.append(float((mu - ref).norm() / ref.norm().clamp_min(1e-12)))
    return values


def checkpoint_payload(model, opt_flow, opt_codec, gens, extra, fixed, signature, device):
    return {
        'format': FORMAT, 'signature_sha256': signature, 'protocol': fixed,
        'flow': flow_state(model.flow), 'optimizer_flow': opt_flow.state_dict(),
        'codec_state': model.codec.state_dict() if model.trainable_codec else None,
        'optimizer_codec': opt_codec.state_dict() if opt_codec is not None else None,
        'rng': rng_state(device),
        'data_rng': gens[0].get_state(), 'query_rng': gens[1].get_state(), 'noise_rng': gens[2].get_state(),
        **extra,
    }


def save_checkpoint(folder, name, payload, keep):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{name}.pt'
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)
    link = folder / f'latest.{os.getpid()}.tmp'
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(path.name, link)
    os.replace(link, folder / 'latest.pt')
    atomic_json(folder / 'latest.json', {'name': name, 'step': payload['step'], 'sha256': digest(path),
                                         'train_seconds': payload['train_seconds'],
                                         'allocated_seconds': payload['allocated_seconds']})
    for old in sorted(folder.glob('step_*.pt'))[:-keep]:
        old.unlink()


def load_resume(folder, model, opt_flow, opt_codec, gens, signature):
    ckpt = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
    if ckpt.get('format') != FORMAT or ckpt.get('signature_sha256') != signature:
        raise ValueError('Resume checkpoint does not match this protocol signature')
    load_flow_state(model.flow, ckpt['flow'])
    opt_flow.load_state_dict(ckpt['optimizer_flow'])
    if opt_codec is not None:
        model.codec.load_state_dict(ckpt['codec_state'], strict=True)
        opt_codec.load_state_dict(ckpt['optimizer_codec'])
    for generator, key in zip(gens, ['data_rng', 'query_rng', 'noise_rng']):
        generator.set_state(ckpt[key])
    restore_rng(ckpt['rng'], next(model.parameters()).device)
    return ckpt


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['frozen', 'joint'], required=True)
    for name in ['data', 'cache', 'out', 'codec-checkpoint', 'init-flow', 'cohorts']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--codec-sha', required=True)
    p.add_argument('--init-flow-sha', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--seed', type=int, default=713)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--points', type=int, default=512)
    p.add_argument('--lr-flow', type=float, default=1e-4)
    p.add_argument('--lr-encoder', type=float, default=1e-6)
    p.add_argument('--lr-decoder', type=float, default=1e-5)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--kl', type=float, default=1e-5)
    p.add_argument('--lambda-ae', type=float, default=1.)
    p.add_argument('--lambda-match', type=float, default=1.)
    p.add_argument('--small-width', type=int, default=384)
    p.add_argument('--small-depth', type=int, default=8)
    p.add_argument('--small-heads', type=int, default=6)
    p.add_argument('--train-cohort', choices=['all', 'overfit_train4', 'pilot_train1024'], default='all')
    p.add_argument('--train-limit', type=int, default=0)
    p.add_argument('--eval-cohort', choices=['pilot32', 'confirmation64'], default='pilot32')
    p.add_argument('--eval-count', type=int, default=32)
    p.add_argument('--eval-size', type=int, default=64)
    p.add_argument('--target-train-hours', type=float, required=True,
                   help='Cumulative training-loop hours including the inherited 900.009 s')
    p.add_argument('--allocated-hour-cap', type=float, required=True,
                   help='Cap on this process: setup, train, eval, checkpoints')
    p.add_argument('--wall-deadline-unix', type=float, required=True)
    p.add_argument('--quick-eval-minutes', type=float, default=60.)
    p.add_argument('--save-minutes', type=float, default=30.)
    p.add_argument('--keep-checkpoints', type=int, default=3)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--milestones', default=','.join(str(x) for x in MILESTONE_HOURS))
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args()
    if a.mode == 'frozen' and (a.lambda_ae or a.lambda_match):
        p.error('frozen mode has no codec terms; pass --lambda-ae 0 --lambda-match 0 for a clean control')
    a.milestone_hours = tuple(float(x) for x in a.milestones.split(',') if x.strip())
    out = a.out
    if not a.resume and (out / 'checkpoints').exists() and any((out / 'checkpoints').iterdir()):
        p.error('Nonempty checkpoint directory without --resume')
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.set_num_threads(a.threads)

    started = time.time()
    source_sha = {name: digest(Path(__file__).with_name(name)) for name in SOURCE_FILES}
    bank = CacheBank(a.data, a.cache)
    cohort_args = argparse.Namespace(mode='train', cohorts=a.cohorts, eval_cohort=a.eval_cohort,
                                     train_cohort=a.train_cohort, train_limit=a.train_limit,
                                     eval_count=a.eval_count)
    train_ids, eval_ids, showcase, cohorts = pick_cohorts(cohort_args, bank)
    fixed = protocol_fields(a, source_sha, cohorts['cohorts_sha256'])
    signature = canonical_sha(fixed)
    model, codec_identity = build(a, bank, device)
    flow_params = [q for q in model.flow.parameters() if q.requires_grad]
    opt_flow = torch.optim.AdamW(flow_params, lr=a.lr_flow, weight_decay=a.weight_decay)
    if model.trainable_codec:
        groups, codec_params = codec_groups(model, a.lr_encoder, a.lr_decoder)
        opt_codec = torch.optim.AdamW(groups, weight_decay=a.weight_decay)
    else:
        codec_params, opt_codec = [], None
    gens = (torch.Generator().manual_seed(a.seed + 101), torch.Generator().manual_seed(a.seed + 202),
            torch.Generator().manual_seed(a.seed + 303))
    folder = out / 'checkpoints'
    # Starting point: fresh from the inherited flow, or exact resume of this protocol.
    state = {'step': 0, 'samples_seen': 0, 'train_seconds': INHERITED_TRAIN_SECONDS, 'eval_seconds': 0.,
             'allocated_seconds': 0., 'milestones_done': []}
    if a.resume:
        ckpt = load_resume(folder, model, opt_flow, opt_codec, gens, signature)
        for key in ['step', 'samples_seen', 'train_seconds', 'eval_seconds', 'allocated_seconds', 'milestones_done']:
            state[key] = ckpt[key]
        reference = torch.load(out / 'reference_mu_showcase.pt', map_location='cpu', weights_only=False)
        del ckpt
    else:
        if digest(a.init_flow) != a.init_flow_sha:
            raise ValueError('Init flow checkpoint SHA mismatch')
        init = torch.load(a.init_flow, map_location='cpu', weights_only=False)
        if init.get('format') != INIT_FORMAT or init.get('protocol_sha256') != INIT_PROTOCOL_SHA:
            raise ValueError('Init checkpoint is not the bound 900 s small-flow protocol')
        load_flow_state(model.flow, init['flow'])
        opt_flow.load_state_dict(init['optimizer'])
        for generator, key in zip(gens, ['data_rng', 'query_rng', 'noise_rng']):
            generator.set_state(init[key])
        restore_rng(init['rng'], device)
        state.update(step=int(init['step']), samples_seen=int(init['samples_seen']),
                     train_seconds=float(init['train_seconds']))
        del init
        reference = posterior_means(model, bank, showcase, device)
        torch.save(reference, out / 'reference_mu_showcase.pt')
        atomic_json(out / 'protocol.json', {**fixed, 'signature_sha256': signature,
                    'codec_identity': codec_identity,
                    'parameter_counts': {'flow_trainable': sum(q.numel() for q in flow_params),
                                         'codec_trainable': sum(q.numel() for q in codec_params)},
                    'history': {'inherited_flow_train_seconds': INHERITED_TRAIN_SECONDS,
                                'inherited_flow_source': a.init_flow_sha,
                                'codec_pretraining_history_seconds': 2688.118,
                                'note': 'history is not charged to the 100 H200h allocation'},
                    'torch_version': torch.__version__,
                    'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})
    setup_seconds = time.time() - started
    session_start = time.perf_counter()
    prior_allocated = state['allocated_seconds']
    allocated_now = lambda: prior_allocated + (time.perf_counter() - session_start)  # noqa: E731
    target_seconds = a.target_train_hours * 3600.
    cap_seconds = a.allocated_hour_cap * 3600.
    stop_file, pause_file = out / 'REQUEST_STOP', out / 'REQUEST_PAUSE'
    last_quick = last_save = time.perf_counter()
    reason = None
    ledger = out / 'budget_ledger.jsonl'
    pid_started = time.ctime()

    def ledger_row(event):
        with ledger.open('a') as stream:
            stream.write(json.dumps({'event': event, 'unix': time.time(), 'step': state['step'],
                                     'train_hours_cumulative': state['train_seconds'] / 3600,
                                     'allocated_hours_this_run': allocated_now() / 3600}) + '\n')

    def status(extra):
        atomic_json(out / 'status.json', {
            'pid': os.getpid(), 'pid_started': pid_started, 'mode': a.mode, 'signature_sha256': signature,
            'step': state['step'], 'samples_seen': state['samples_seen'],
            'train_hours_cumulative': state['train_seconds'] / 3600,
            'eval_hours_cumulative': state['eval_seconds'] / 3600,
            'allocated_hours_this_run': allocated_now() / 3600,
            'milestones_done': state['milestones_done'], 'updated_unix': time.time(),
            **memory_peaks(device), **extra})

    def run_eval(ids, tag):
        began = time.perf_counter()
        try:
            return evaluate(model, bank, ids, showcase, out / 'eval' / f'{tag}_{state["step"]:07d}',
                            a.eval_size, state['step'], device, a.amp, a.wall_deadline_unix)
        finally:
            sync(device)
            state['eval_seconds'] += time.perf_counter() - began

    def save(name):
        state['allocated_seconds'] = allocated_now()
        payload = checkpoint_payload(model, opt_flow, opt_codec, gens, {
            'step': state['step'], 'samples_seen': state['samples_seen'],
            'train_seconds': state['train_seconds'], 'eval_seconds': state['eval_seconds'],
            'allocated_seconds': state['allocated_seconds'], 'setup_seconds': setup_seconds,
            'milestones_done': state['milestones_done']}, fixed, signature, device)
        save_checkpoint(folder, name, payload, a.keep_checkpoints)

    if not a.resume:
        ledger_row('start_from_inherited_flow')
        quick0 = run_eval(showcase, 'quick_step0')
        with (out / 'quick_history.jsonl').open('a') as stream:
            stream.write(json.dumps({'step': state['step'], 'means': quick0['means'],
                                     'codec_drift_showcase': [0.] * len(showcase),
                                     'train_hours': state['train_seconds'] / 3600}, allow_nan=False) + '\n')
    else:
        ledger_row('resume')

    model.train()
    try:
        while True:
            if stop_file.exists():
                reason = 'requested_stop'
                break
            if pause_file.exists():
                reason = 'paused'
                break
            if state['train_seconds'] >= target_seconds:
                reason = 'target_train_hours_reached'
                break
            if allocated_now() >= cap_seconds:
                reason = 'allocated_hour_cap'
                break
            if time.time() >= a.wall_deadline_unix:
                reason = 'wall_deadline'
                break
            train_hours = state['train_seconds'] / 3600
            due = [h for h in a.milestone_hours if h <= train_hours and h not in state['milestones_done']]
            if due:
                h = due[0]
                run_eval(eval_ids, f'milestone_{h:g}h')
                state['milestones_done'].append(h)
                save(f'milestone_{h:g}h')
                ledger_row(f'milestone_{h:g}h')
            ids = [train_ids[i] for i in torch.randint(len(train_ids), (a.batch,),
                                                        generator=gens[0]).tolist()]
            sync(device)
            began = time.perf_counter()
            stats = training_step(model, bank, ids, a, device, gens, opt_flow, opt_codec,
                                  flow_params, codec_params)
            sync(device)
            duration = time.perf_counter() - began
            state['step'] += 1
            state['samples_seen'] += a.batch
            state['train_seconds'] += duration
            if state['step'] % a.log_every == 0 or state['step'] == 1:
                row = {'step': state['step'], 'samples_seen': state['samples_seen'],
                       'loss_total': float(stats['total']), 'flow': float(stats['flow']),
                       'ae': float(stats['ae']), 'match': float(stats['match']), 'kl': float(stats['kl']),
                       'grad_flow': float(stats['grad_flow']), 'grad_codec': float(stats['grad_codec']),
                       'step_seconds': duration, 'train_hours_cumulative': state['train_seconds'] / 3600,
                       'eval_hours_cumulative': state['eval_seconds'] / 3600, 'unix_time': time.time()}
                with (out / 'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                status({'phase': 'training'})
            if time.perf_counter() - last_quick >= a.quick_eval_minutes * 60:
                result = run_eval(showcase, 'quick')
                drift = codec_drift(model, bank, showcase, reference, device) if model.trainable_codec \
                    else [0.] * len(showcase)
                with (out / 'quick_history.jsonl').open('a') as stream:
                    stream.write(json.dumps({'step': state['step'], 'means': result['means'],
                                             'codec_drift_showcase': drift,
                                             'train_hours': state['train_seconds'] / 3600,
                                             'unix_time': time.time()}, allow_nan=False) + '\n')
                ledger_row('quick_eval')
                last_quick = time.perf_counter()
            if time.perf_counter() - last_save >= a.save_minutes * 60:
                save(f'step_{state["step"]:07d}')
                ledger_row('checkpoint')
                last_save = time.perf_counter()
            del stats, ids
    except Exception as exc:
        status({'phase': 'failed', 'error': type(exc).__name__ + ': ' + str(exc)})
        ledger_row('failed:' + type(exc).__name__)
        raise

    save('final' if reason in {'target_train_hours_reached', 'requested_stop'} else 'interrupted')
    final_eval_done = False
    if reason in {'target_train_hours_reached', 'requested_stop', 'allocated_hour_cap'} \
            and time.time() < a.wall_deadline_unix:
        run_eval(eval_ids, 'final')
        save('final')
        final_eval_done = True
    drift = codec_drift(model, bank, showcase, reference, device) if model.trainable_codec else []
    status({'phase': 'stopped', 'reason': reason,
            'status': 'complete' if reason in {'target_train_hours_reached', 'requested_stop'} else reason,
            'final_eval_done': final_eval_done, 'codec_drift_showcase': drift,
            **memory_peaks(device)})
    ledger_row('exit:' + reason)
    print(json.dumps({'reason': reason, 'step': state['step'], 'train_hours': state['train_seconds'] / 3600,
                      'allocated_hours_this_run': allocated_now() / 3600}), flush=True)


if __name__ == '__main__':
    main()

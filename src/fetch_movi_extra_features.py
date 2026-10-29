"""Stream official MOVi-C/D/F 128 train shards and keep only Wan VAE features (amendment G3).

New scenes for the "more data" question: MOVi-D is MOVi-E with a static camera,
MOVi-F is MOVi-E with motion blur, MOVi-C has 3-10 objects and a static camera.
Only the train split of each is read; test and validation splits are never
downloaded. Each shard is fetched whole into memory, every record's 24 PNG
frames are decoded to RGB, and the same frozen VAE, encode function,
coordinates and file layout as the 10-27 cache (code_v5/cache_vae_features.py,
imported, SHA checked) produce the features. RGB is not kept; each record
stores its shard URL, index and record SHA256 so it can be re-fetched.

Clip IDs are '<dataset>_s<shard>_r<index>'. Control: REQUEST_STOP in --out
stops after the current batch; status.json is the heartbeat. Each shard gets at
most --retries attempts and failures are recorded, never retried forever;
--max-gib caps total network bytes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import queue
import struct
import sys
import threading
import time

import numpy as np
from PIL import Image
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_movi_segmentations import fields  # noqa: E402

ORIGINAL_SCRIPT_SHA = '15d4221ee6192e0ea6583fef51927dcf8b22c6a2183a844810d4dd3da7da4081'
BUCKET = 'https://storage.googleapis.com/kubric-public/tfds/{0}/128x128/1.0.0/'
LISTING = 'https://storage.googleapis.com/storage/v1/b/kubric-public/o'


def shard_urls(dataset):
    prefix = f'tfds/{dataset}/128x128/1.0.0/{dataset}-train.tfrecord-'
    names, token = [], None
    while True:
        params = {'prefix': prefix, 'fields': 'items(name,size),nextPageToken', 'maxResults': 1000}
        if token:
            params['pageToken'] = token
        page = requests.get(LISTING, params=params, timeout=30).json()
        names += [(i['name'], int(i['size'])) for i in page.get('items', [])]
        token = page.get('nextPageToken')
        if not token:
            break
    names.sort()
    return [('https://storage.googleapis.com/kubric-public/' + n, s) for n, s in names]


def video_frames(example):
    """Only the 'video' bytes list of a tf.train.Example."""
    feature_map = next(v for f, w, v in fields(example) if f == 1 and w == 2)
    for _, _, entry in fields(feature_map):
        parts = {f: v for f, w, v in fields(entry)}
        if parts[1] == b'video':
            (kind, _, payload), = list(fields(parts[2]))
            if kind != 1:
                raise ValueError('video is not a bytes list')
            return [v for _, _, v in fields(payload)]
    raise ValueError('Missing video feature')


def records(data):
    pos = 0
    while pos < len(data):
        length = struct.unpack('<Q', data[pos:pos + 8])[0]
        yield data[pos + 12:pos + 12 + length]
        pos += 16 + length
    if pos != len(data):
        raise ValueError('TFRecord framing mismatch')


def fetch_shard(url, size, retries, meter):
    last = None
    for attempt in range(retries):
        try:
            meter(size)
            response = requests.get(url, timeout=(10, 300))
            response.raise_for_status()
            if len(response.content) != size:
                raise ValueError('Shard size differs from listing')
            return response.content
        except Exception as exc:  # network; bounded attempts
            last = exc
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f'{type(last).__name__}: {last}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['data', 'pretrained', 'original-cache', 'original-script', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--datasets', default='movi_d,movi_f,movi_c')
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--retries', type=int, default=3)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--max-gib', type=float, default=260.)
    p.add_argument('--limit-shards', type=int, default=0)
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location('cache_vae_features', a.original_script)
    cvf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cvf)
    if cvf.digest(a.original_script) != ORIGINAL_SCRIPT_SHA:
        raise ValueError('Original cache script differs from the one that built the 10-27 cache')
    original = json.loads((a.original_cache / 'protocol.json').read_text())
    if cvf.vae_fingerprint(a.pretrained) != original['vae_fingerprint']:
        raise ValueError('VAE weights differ from the original cache')
    protocol = {**original, 'version': 'wan-vae-feature-cache-extra-v1',
                'extra_source': 'official kubric-public tfds <dataset>/128x128/1.0.0 train shards only; RGB not kept',
                'extra_script_sha256': cvf.digest(Path(__file__)),
                'original_cache_protocol_sha256': cvf.canonical_sha(original)}
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Existing extra cache protocol differs')
    cvf.atomic_json(path, protocol)
    protocol_sha = cvf.canonical_sha(protocol)

    jobs = []
    for dataset in a.datasets.split(','):
        urls = shard_urls(dataset)
        if not urls:
            raise ValueError('No train shards listed for ' + dataset)
        jobs += [(dataset, i, u, s) for i, (u, s) in enumerate(urls[:a.limit_shards or None])]
    done_marks = a.out / 'shards_done'
    done_marks.mkdir(exist_ok=True)
    jobs = [j for j in jobs if not (done_marks / f'{j[0]}_{j[1]:05d}').exists()]

    lock, used = threading.Lock(), [0]

    def meter(n):
        with lock:
            if used[0] + n > a.max_gib * 2 ** 30:
                raise RuntimeError('Network byte cap reached')
            used[0] += n

    clips = queue.Queue(maxsize=512)
    failures = (a.out / 'failures.jsonl').open('a')

    def produce(job):
        dataset, shard, url, size = job
        if (a.out / 'REQUEST_STOP').exists():
            return
        try:
            data = fetch_shard(url, size, a.retries, meter)
            items = []
            for index, example in enumerate(records(data)):
                frames = video_frames(example)
                rgb = np.stack([np.asarray(Image.open(io.BytesIO(f)).convert('RGB')) for f in frames])
                if rgb.shape != (24, 128, 128, 3):
                    raise ValueError(f'Unexpected video shape {rgb.shape}')
                items.append({'id': f'{dataset}_s{shard:05d}_r{index:03d}', 'dataset': dataset, 'shard_url': url,
                              'record_index': index, 'record_sha256': hashlib.sha256(example).hexdigest(), 'rgb': rgb})
            for item in items:
                clips.put(item)
            clips.put(('shard_done', dataset, shard, len(items)))
        except Exception as exc:  # recorded, not retried beyond fetch_shard's bound
            with lock:
                failures.write(json.dumps({'dataset': dataset, 'shard': shard, 'url': url,
                                           'error': f'{type(exc).__name__}: {exc}'}) + '\n')
                failures.flush()
            clips.put(('shard_failed', dataset, shard, 0))

    import torch
    from diffusers import AutoencoderKLWan
    torch.cuda.set_device(0)
    vae = AutoencoderKLWan.from_pretrained(a.pretrained, subfolder='vae', torch_dtype=torch.float32,
                                           local_files_only=True).cuda().float().eval()
    vae.requires_grad_(False)
    mean = torch.tensor(vae.config.latents_mean, device='cuda', dtype=torch.float32).reshape(1, 48, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device='cuda', dtype=torch.float32).reshape(1, 48, 1, 1, 1)
    started = time.time()
    counts = {'clips': 0, 'shards_done': 0, 'shards_failed': 0}
    pending_shards = {}

    def status(state):
        cvf.atomic_json(a.out / 'status.json', {'state': state, **counts, 'shards_total': len(jobs),
                                                 'network_gib': used[0] / 2 ** 30, 'elapsed_s': time.time() - started,
                                                 'pid': os.getpid(), 'unix': time.time()})

    def flush(batch):
        video = torch.from_numpy(np.stack([b['rgb'] for b in batch])).permute(0, 1, 4, 2, 3).cuda().float() / 255
        with torch.inference_mode():
            prefix = cvf.encode(vae, video[:, :4], mean, std)
            future = cvf.encode(vae, video[:, 4:24], mean, std)
        pc, _ = cvf.coordinates(prefix.shape, 4, 0)
        fc, flc = cvf.coordinates(future.shape, 20, 4)
        pf = prefix.flatten(2).transpose(1, 2).cpu().numpy().astype(np.float32)
        ff = future.flatten(2).transpose(1, 2).cpu().numpy().astype(np.float32)
        if not np.isfinite(pf).all() or not np.isfinite(ff).all():
            raise ValueError('Nonfinite cached feature values')
        for i, b in enumerate(batch):
            relative = Path('features') / 'train' / (b['id'] + '.npz')
            target = a.out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            arrays = {'prefix': pf[i], 'future': ff[i], 'prefix_coords': pc, 'future_coords': fc,
                      'future_local_coords': flc}
            tmp = target.with_name(target.name + f'.{os.getpid()}.tmp')
            with tmp.open('wb') as stream:
                np.savez(stream, **arrays)
            os.replace(tmp, target)
            item = {'id': b['id'], 'split': 'train', 'dataset': b['dataset'], 'source_url': b['shard_url'],
                    'record_index': b['record_index'], 'record_sha256': b['record_sha256'],
                    'features': str(relative), 'features_sha256': cvf.digest(target), 'protocol_sha256': protocol_sha,
                    'arrays': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in arrays.items()}}
            cvf.atomic_json(a.out / 'records' / 'train' / (b['id'] + '.json'), item)
            counts['clips'] += 1
            key = (b['dataset'], int(b['id'].split('_s')[1][:5]))
            pending_shards[key] -= 1

    def settle():
        for key in [k for k, v in pending_shards.items() if v == 0 and k in finished]:
            (done_marks / f'{key[0]}_{key[1]:05d}').touch()
            del pending_shards[key]
            finished.discard(key)
            counts['shards_done'] += 1

    finished = set()
    pool = ThreadPoolExecutor(a.workers)
    futures = [pool.submit(produce, j) for j in jobs]
    outstanding, batch = len(jobs), []
    status('running')
    while outstanding:
        item = clips.get()
        if isinstance(item, tuple):
            kind, dataset, shard, n = item
            outstanding -= 1
            if kind == 'shard_failed':
                counts['shards_failed'] += 1
            else:
                finished.add((dataset, shard))
                pending_shards.setdefault((dataset, shard), 0)
        else:
            key = (item['dataset'], int(item['id'].split('_s')[1][:5]))
            pending_shards[key] = pending_shards.get(key, 0) + 1
            batch.append(item)
        if len(batch) >= a.batch or (not outstanding and batch):
            flush(batch)
            batch = []
            settle()
            status('running')
        if (a.out / 'REQUEST_STOP').exists() and not batch:
            break
    if batch:
        flush(batch)
    settle()
    pool.shutdown(wait=True, cancel_futures=True)
    for f in futures:
        if f.exception():
            raise f.exception()
    status('stopped' if (a.out / 'REQUEST_STOP').exists() else 'finished')
    print(json.dumps({**counts, 'network_gib': used[0] / 2 ** 30, 'elapsed_s': time.time() - started}))


if __name__ == '__main__':
    main()

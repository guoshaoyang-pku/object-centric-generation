"""Stream official MOVi-E 128 records and keep only instance segmentation.

For every train/validation clip of the immutable snapshot manifest, the full
official TFRecord byte range is fetched into memory, its SHA256 must equal the
manifest's source_record_sha256, and its decoded RGB must equal the snapshot
array exactly. Only the masks and a few small instance fields are written; the
record itself is discarded. Annotations feed the oracle-codec experiment
(amendment F); prefix-only sampling must use masks of the 4 observed frames only.

Control: REQUEST_STOP in --out stops after in-flight clips; status.json is the
heartbeat. Each clip gets at most --retries attempts; failures are recorded,
never retried forever. --max-gib caps total network bytes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import threading
import time

import numpy as np
from PIL import Image
import requests

BASE = 'https://storage.googleapis.com/kubric-public/tfds/movi_e/128x128/1.0.0/'


def varint(data, pos):
    result = shift = 0
    while True:
        value = data[pos]
        pos += 1
        result |= (value & 127) << shift
        if value < 128:
            return result, pos
        shift += 7
        if shift > 70:
            raise ValueError('Invalid protobuf varint')


def fields(data):
    pos = 0
    while pos < len(data):
        tag, pos = varint(data, pos)
        field, wire = tag >> 3, tag & 7
        if wire == 2:
            count, pos = varint(data, pos)
            value = data[pos:pos + count]
            pos += count
        elif wire == 0:
            value, pos = varint(data, pos)
        elif wire in (1, 5):
            count = 8 if wire == 1 else 4
            value = data[pos:pos + count]
            pos += count
        else:
            raise ValueError(f'Unsupported protobuf wire {wire}')
        yield field, wire, value


WANTED = {'video', 'segmentations', 'metadata/num_instances', 'instances/is_dynamic',
          'instances/visibility', 'instances/image_positions'}


def parse_example(example):
    """Only the wanted TF features; same decoding as joint_video/fetch_movi_annotations.py."""
    feature_map = next(v for f, w, v in fields(example) if f == 1 and w == 2)
    result = {}
    for field, wire, entry in fields(feature_map):
        parts = {f: v for f, w, v in fields(entry)}
        name = parts[1].decode('utf-8')
        if name not in WANTED:
            continue
        (kind, wire, payload), = list(fields(parts[2]))
        values = []
        for f, w, v in fields(payload):
            if kind == 1:
                values.append(v)
            elif kind == 2 and w == 2:
                values.extend(np.frombuffer(v, dtype='<f4').tolist())
            elif kind == 2 and w == 5:
                values.append(struct.unpack('<f', v)[0])
            elif kind == 3:
                if w == 0:
                    values.append(v)
                else:
                    pos = 0
                    while pos < len(v):
                        number, pos = varint(v, pos)
                        values.append(number)
            else:
                raise ValueError('Unsupported TF list wire type: ' + name)
        result[name] = (kind, values)
    missing = WANTED - set(result)
    if missing:
        raise ValueError('Missing TF features: ' + ', '.join(sorted(missing)))
    return result


class Meter:
    def __init__(self, cap_bytes):
        self.cap, self.used, self.lock = cap_bytes, 0, threading.Lock()

    def reserve(self, n):
        with self.lock:
            if self.used + n > self.cap:
                raise RuntimeError('Network byte cap reached')
            self.used += n


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def fetch_one(a, record, meter, session):
    target = a.out / record['split'] / (record['id'] + '.npz')
    if target.exists():
        return {'id': record['id'], 'status': 'present'}
    if not record['source_url'].startswith(BASE):
        raise ValueError('Unexpected source URL')
    start, end = record['byte_range']
    size = end - start + 1
    if size != record['source_record_bytes'] or size > 24 * 2 ** 20:
        raise ValueError('Invalid record bounds')
    last = None
    for attempt in range(a.retries):
        try:
            meter.reserve(size)
            response = session.get(record['source_url'], headers={'Range': f'bytes={start}-{end}'}, timeout=(10, 60))
            response.raise_for_status()
            data = response.content
            if len(data) != size or sha256(data) != record['source_record_sha256']:
                raise ValueError('Record bytes differ from manifest SHA')
            break
        except Exception as exc:  # network or checksum; bounded attempts
            last = exc
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f'{type(last).__name__}: {last}')
    length = struct.unpack('<Q', data[:8])[0]
    if length + 16 != len(data):
        raise ValueError('TFRecord framing mismatch')
    feats = parse_example(data[12:-4])
    rgb = np.stack([np.asarray(Image.open(io.BytesIO(p)).convert('RGB')) for p in feats['video'][1]])
    array_path = a.data / record['array']
    raw = array_path.read_bytes()
    if sha256(raw) != record['array_sha256']:
        raise ValueError('Snapshot RGB checksum mismatch')
    if not np.array_equal(rgb, np.load(io.BytesIO(raw), allow_pickle=False)):
        raise ValueError('Official RGB differs from snapshot array')
    masks = np.stack([np.asarray(Image.open(io.BytesIO(p))) for p in feats['segmentations'][1]])
    if masks.ndim == 4:
        masks = masks[..., 0]
    n = int(feats['metadata/num_instances'][1][0])
    if masks.shape != (24, 128, 128) or masks.dtype != np.uint8 or not 1 <= n <= 255 or masks.max() > n:
        raise ValueError('Unexpected segmentation layout')
    arrays = {'segmentations': masks,
              'is_dynamic': np.asarray(feats['instances/is_dynamic'][1], np.int64).reshape(n),
              'visibility': np.asarray(feats['instances/visibility'][1], np.int64).reshape(n, 24),
              'image_positions': np.asarray(feats['instances/image_positions'][1], np.float32).reshape(n, 24, 2)}
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, target)
    return {'id': record['id'], 'status': 'complete', 'instances': n,
            'dynamic': int(arrays['is_dynamic'].sum()), 'npz_sha256': sha256(target.read_bytes())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--retries', type=int, default=3)
    p.add_argument('--max-gib', type=float, default=100.)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--splits', default='validation,train')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((a.data / 'manifest.json').read_text())
    if manifest.get('test_split_used'):
        raise SystemExit('Test split is not admitted')
    order = a.splits.split(',')
    records = sorted((r for r in manifest['samples'] if r['split'] in order),
                     key=lambda r: (order.index(r['split']), r['id']))
    if a.limit:
        records = records[:a.limit]
    meter = Meter(int(a.max_gib * 2 ** 30))
    log = (a.out / 'clips.jsonl').open('a')
    done = failed = 0
    started = time.time()

    def status(state):
        tmp = a.out / 'status.json.tmp'
        tmp.write_text(json.dumps({'state': state, 'done': done, 'failed': failed, 'total': len(records),
                                   'network_gib': meter.used / 2 ** 30, 'elapsed_s': time.time() - started,
                                   'pid': os.getpid(), 'unix': time.time()}) + '\n')
        os.replace(tmp, a.out / 'status.json')

    local = threading.local()

    def worker(record):
        if (a.out / 'REQUEST_STOP').exists():
            return {'id': record['id'], 'status': 'skipped_stop'}
        if not hasattr(local, 'session'):
            local.session = requests.Session()
        try:
            return fetch_one(a, record, meter, local.session)
        except Exception as exc:
            return {'id': record['id'], 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}

    status('running')
    with ThreadPoolExecutor(a.workers) as pool:
        for future in as_completed([pool.submit(worker, r) for r in records]):
            row = future.result()
            if row['status'] == 'failed':
                failed += 1
            elif row['status'] in {'complete', 'present'}:
                done += 1
            if row['status'] != 'present':
                log.write(json.dumps(row) + '\n')
                log.flush()
            if (done + failed) % 100 == 0:
                status('running')
    status('stopped' if (a.out / 'REQUEST_STOP').exists() else 'finished')
    print(json.dumps({'done': done, 'failed': failed, 'network_gib': meter.used / 2 ** 30}))


if __name__ == '__main__':
    main()

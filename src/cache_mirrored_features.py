"""Wan VAE feature cache of horizontally mirrored train clips (amendment F2).

Same frozen VAE, encode function, coordinates and file layout as the original
cache (research_20261027/code_v5/cache_vae_features.py, imported, SHA checked);
the only change is that the source RGB is flipped along width before encoding.
Records keep the original clip IDs and source SHA; protocol.json says mirror.
Validation clips are never mirrored or cached here.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ORIGINAL_SCRIPT_SHA = '15d4221ee6192e0ea6583fef51927dcf8b22c6a2183a844810d4dd3da7da4081'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['data', 'pretrained', 'original-cache', 'original-script', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location('cache_vae_features', a.original_script)
    cvf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cvf)
    if cvf.digest(a.original_script) != ORIGINAL_SCRIPT_SHA:
        raise ValueError('Original cache script differs from the one that built the 10-27 cache')
    original = json.loads((a.original_cache / 'protocol.json').read_text())
    if original['source_script_sha256'] != ORIGINAL_SCRIPT_SHA:
        raise ValueError('Original cache was not built by the expected script')
    manifest = json.loads((a.data / 'manifest.json').read_text())
    records = sorted((r for r in manifest['samples'] if r['split'] == 'train'), key=lambda r: r['id'])
    if a.limit:
        records = records[:a.limit]
    protocol = {**original, 'version': 'wan-vae-feature-cache-mirror-v1',
                'mirror': 'source RGB flipped along width (x -> -x) before encoding; train split only',
                'mirror_script_sha256': cvf.digest(Path(__file__)),
                'original_cache_protocol_sha256': cvf.canonical_sha(original)}
    if cvf.vae_fingerprint(a.pretrained) != original['vae_fingerprint']:
        raise ValueError('VAE weights differ from the original cache')
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Existing mirror cache protocol differs')
    cvf.atomic_json(path, protocol)
    protocol_sha = cvf.canonical_sha(protocol)
    pending = [r for r in records if not (a.out / 'records' / 'train' / (r['id'] + '.json')).exists()]

    import torch
    from diffusers import AutoencoderKLWan
    torch.cuda.set_device(0)
    vae = AutoencoderKLWan.from_pretrained(a.pretrained, subfolder='vae', torch_dtype=torch.float32,
                                           local_files_only=True).cuda().float().eval()
    vae.requires_grad_(False)
    mean = torch.tensor(vae.config.latents_mean, device='cuda', dtype=torch.float32).reshape(1, 48, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device='cuda', dtype=torch.float32).reshape(1, 48, 1, 1, 1)
    started, done = time.time(), 0
    for offset in range(0, len(pending), a.batch):
        if (a.out / 'REQUEST_STOP').exists():
            break
        batch = pending[offset:offset + a.batch]
        values = []
        for r in batch:
            if cvf.digest(a.data / r['array']) != r['array_sha256']:
                raise ValueError('Source array SHA mismatch: ' + r['id'])
            values.append(np.load(a.data / r['array'], allow_pickle=False)[:, :, ::-1])
        video = torch.from_numpy(np.ascontiguousarray(np.stack(values))).permute(0, 1, 4, 2, 3).cuda().float() / 255
        with torch.inference_mode():
            prefix = cvf.encode(vae, video[:, :4], mean, std)
            future = cvf.encode(vae, video[:, 4:24], mean, std)
        pc, _ = cvf.coordinates(prefix.shape, 4, 0)
        fc, flc = cvf.coordinates(future.shape, 20, 4)
        pf = prefix.flatten(2).transpose(1, 2).cpu().numpy().astype(np.float32)
        ff = future.flatten(2).transpose(1, 2).cpu().numpy().astype(np.float32)
        if not np.isfinite(pf).all() or not np.isfinite(ff).all():
            raise ValueError('Nonfinite cached feature values')
        for i, r in enumerate(batch):
            relative = Path('features') / 'train' / (r['id'] + '.npz')
            target = a.out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            arrays = {'prefix': pf[i], 'future': ff[i], 'prefix_coords': pc, 'future_coords': fc,
                      'future_local_coords': flc}
            tmp = target.with_name(target.name + f'.{os.getpid()}.tmp')
            with tmp.open('wb') as stream:
                np.savez(stream, **arrays)
            os.replace(tmp, target)
            item = {'id': r['id'], 'split': 'train', 'source_array': r['array'],
                    'source_array_sha256': r['array_sha256'], 'source_url': r['source_url'],
                    'features': str(relative), 'features_sha256': cvf.digest(target), 'protocol_sha256': protocol_sha,
                    'mirrored': True, 'arrays': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in arrays.items()}}
            cvf.atomic_json(a.out / 'records' / 'train' / (r['id'] + '.json'), item)
            done += 1
        cvf.atomic_json(a.out / 'status.json', {'done_this_run': done, 'pending_at_start': len(pending),
                                                 'elapsed_s': time.time() - started, 'pid': os.getpid(),
                                                 'unix': time.time()})
    print(json.dumps({'done_this_run': done, 'pending_at_start': len(pending), 'elapsed_s': time.time() - started}))


if __name__ == '__main__':
    main()

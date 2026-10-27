"""Representation-independent foreground/background metrics with verified GT masks."""
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def _verified(root, bank, identifier, required):
    if root is None:
        if required:
            raise ValueError('Required annotation root is missing')
        return None
    folder = Path(root) / identifier
    metadata_path, mask_path = folder / 'metadata.json', folder / 'segmentations.npy'
    if not metadata_path.exists() and not mask_path.exists():
        if required:
            raise FileNotFoundError('Missing required mask: ' + identifier)
        return None
    if not metadata_path.is_file() or not mask_path.is_file():
        raise ValueError('Incomplete annotation pair: ' + identifier)
    metadata_sha = _sha(metadata_path)
    meta = json.loads(metadata_path.read_text())
    record = bank.records.get(identifier)
    if record is None or meta.get('id') != identifier or meta.get('status') != 'complete':
        raise ValueError('Annotation id/status differs: ' + identifier)
    if meta.get('rgb_array_sha256') != record['array_sha256'] or meta.get('source_rgb_exactly_equal') is not True:
        raise ValueError('Annotation RGB identity differs: ' + identifier)
    for key in ['source_record_sha256', 'source_url']:
        if record.get(key) is not None and meta.get(key) != record[key]:
            raise ValueError('Annotation source identity differs: ' + identifier + '/' + key)
    mask_sha = _sha(mask_path)
    if meta.get('segmentations_npy_sha256') != mask_sha:
        raise ValueError('Annotation mask SHA differs: ' + identifier)
    mask = np.load(mask_path, allow_pickle=False)
    if mask.shape != (24, 128, 128) or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError('Expected integer segmentation [24,128,128]')
    if mask.min() < 0 or (meta.get('num_instances') is not None and mask.max() > meta['num_instances']):
        raise ValueError('Invalid instance label range: ' + identifier)
    if _sha(metadata_path) != metadata_sha or _sha(mask_path) != mask_sha:
        raise ValueError('Annotation changed while reading: ' + identifier)
    return mask, {'metadata_sha256': metadata_sha, 'segmentations_npy_sha256': mask_sha,
                  'rgb_array_sha256': meta['rgb_array_sha256'], 'shape': list(mask.shape),
                  'source_record_sha256': meta.get('source_record_sha256')}


def annotation_binding(root, bank):
    if root is None:
        return {'status': 'none', 'root': None, 'records': {}, 'foreground_metrics_available': False}
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    records = {}
    for metadata in sorted(root.glob('*/metadata.json')):
        identifier = metadata.parent.name
        _, binding = _verified(root, bank, identifier, required=True)
        records[identifier] = binding
    return {'status': 'bound', 'root': str(root.resolve()), 'records': records,
            'foreground_metrics_available': bool(records),
            'definition': 'Foreground is positive official instance ID; future frames4:24. No inferred parent weights.'}


def load_fg_mask(root, bank, identifier, size, required=False):
    if not isinstance(size, int) or size < 1:
        raise ValueError('Mask size must be a positive integer')
    result = _verified(root, bank, identifier, required)
    if result is None:
        return None
    mask = result[0][4:24] > 0
    if size != 128:
        # Matches nearest-neighbor resizing with the source pixel at floor(i*128/size).
        indexes = np.minimum(np.arange(size) * 128 // size, 127)
        mask = mask[:, indexes][:, :, indexes]
    return mask.astype(bool, copy=False)


def add_fg_bg(pred, truth, mask):
    p, y = np.asarray(pred), np.asarray(truth)
    if p.shape != y.shape or p.ndim != 4 or p.shape[0] != 20:
        raise ValueError('Expected matching 20-frame RGB videos')
    if p.shape[1] == 3:
        error = ((p.astype(np.float64) - y.astype(np.float64)) ** 2).mean(1)
    elif p.shape[-1] == 3:
        error = ((p.astype(np.float64) - y.astype(np.float64)) ** 2).mean(-1)
    else:
        raise ValueError('Expected TCHW or THWC RGB layout')
    if not np.isfinite(error).all():
        raise ValueError('Nonfinite RGB values')
    result = {'scalar': {'foreground_mse': None, 'background_mse': None},
              'time': {'foreground_frame_mse': [None] * 20, 'background_frame_mse': [None] * 20}}
    if mask is None:
        return result
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape != error.shape:
        raise ValueError('Expected boolean mask [20,H,W] aligned with RGB')
    for name, selection in [('foreground', mask), ('background', ~mask)]:
        result['scalar'][name + '_mse'] = float(error[selection].mean()) if selection.any() else None
        result['time'][name + '_frame_mse'] = [float(e[m].mean()) if m.any() else None
                                               for e, m in zip(error, selection)]
    return result

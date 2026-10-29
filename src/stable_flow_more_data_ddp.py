"""stable_flow_ddp.py, unchanged, on MOVi-E train plus new-scene train clips (amendment G3).

Extra flags:
  --extra-cache DIR   fetch_movi_extra_features.py output (MOVi-C/D/F train
                      features, same frozen VAE and layout)
  --extra-datasets L  comma list of dataset prefixes to use, e.g. movi_d,movi_f
  --mirror-cache DIR  optional: also add the F2 mirrored MOVi-E clips
Train IDs are the 9,749 MOVi-E originals in their usual order (so the fixed
gain-calibration batch train_ids[:8] is the main run's), then mirrored copies,
then the extra clips sorted by ID. Batches sample uniformly over all of them.
Validation, showcase and every evaluation read only MOVi-E validation clips.
The extra cache protocol, the clip counts and this file's SHA enter the
protocol signature.
"""
from collections import OrderedDict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stable_flow_ddp as sfd  # noqa: E402
from codec_research import CacheBank, digest, canonical_sha  # noqa: E402

MIRROR_SUFFIX = '__mirror'


def take(flag, default=None):
    if flag not in sys.argv:
        return default
    i = sys.argv.index(flag)
    value = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    return value


EXTRA = Path(take('--extra-cache'))
DATASETS = take('--extra-datasets').split(',')
MIRROR = take('--mirror-cache')
MIRROR = Path(MIRROR) if MIRROR else None


class ExtraView:
    """Duck-typed CacheBank state so CacheBank.features loads extra clips with the same checks."""
    def __init__(self, cache, protocol_sha):
        self.cache, self.protocol_sha, self.metadata, self.feature_lru = cache, protocol_sha, {}, OrderedDict()

    def check_metadata(self, identifier):
        if identifier in self.metadata:
            return self.metadata[identifier]
        if Path(identifier).name != identifier:
            raise ValueError('Unsafe clip identity')
        meta = json.loads((self.cache / 'records' / 'train' / (identifier + '.json')).read_text())
        if meta['id'] != identifier or meta['split'] != 'train' or meta['protocol_sha256'] != self.protocol_sha:
            raise ValueError('Extra cache record provenance differs: ' + identifier)
        if meta['features'] != str(Path('features') / 'train' / (identifier + '.npz')):
            raise ValueError('Unexpected feature path: ' + identifier)
        self.metadata[identifier] = meta
        return meta

    features = CacheBank.features


class MoreDataBank(CacheBank):
    def __init__(self, data, cache):
        super().__init__(data, cache)
        protocol = json.loads((EXTRA / 'protocol.json').read_text())
        if protocol.get('version') != 'wan-vae-feature-cache-extra-v1' or \
                protocol['original_cache_protocol_sha256'] != self.protocol_sha:
            raise ValueError('Extra cache does not derive from this feature cache')
        self.extra = ExtraView(EXTRA, canonical_sha(protocol))
        names = sorted(p.stem for p in (EXTRA / 'records' / 'train').glob('*.json'))
        self.extra_ids = [n for n in names if any(n.startswith(d + '_s') for d in DATASETS)]
        if not self.extra_ids:
            raise ValueError('No extra clips for ' + ','.join(DATASETS))
        self.mirror_ids = []
        if MIRROR:
            self.mirror = CacheBank(data, MIRROR)
            if self.mirror.protocol.get('version') != 'wan-vae-feature-cache-mirror-v1' or \
                    self.mirror.protocol['original_cache_protocol_sha256'] != self.protocol_sha:
                raise ValueError('Mirror cache does not derive from this feature cache')
            self.mirror_ids = [i + MIRROR_SUFFIX for i in super().ids('train')]
        self.counts = {'movi_e': len(super().ids('train')), 'mirror': len(self.mirror_ids),
                       **{d: sum(n.startswith(d + '_s') for n in self.extra_ids) for d in DATASETS}}

    def ids(self, split):
        base = super().ids(split)
        return base + self.mirror_ids + self.extra_ids if split == 'train' else base

    def check_metadata(self, identifier):
        if identifier.endswith(MIRROR_SUFFIX):
            return self.mirror.check_metadata(identifier[:-len(MIRROR_SUFFIX)])
        if identifier.startswith('movi_'):
            return self.extra.check_metadata(identifier)
        return super().check_metadata(identifier)

    def features(self, identifier):
        if identifier.endswith(MIRROR_SUFFIX):
            return self.mirror.features(identifier[:-len(MIRROR_SUFFIX)])
        if identifier.startswith('movi_'):
            return self.extra.features(identifier)
        return super().features(identifier)


_protocol_fields = sfd.protocol_fields
_bank = {}


class RecordingBank(MoreDataBank):
    def __init__(self, data, cache):
        super().__init__(data, cache)
        _bank['counts'] = self.counts


def protocol_fields(a, world, source_sha, cohort_sha, branch_sha):
    fixed = _protocol_fields(a, world, source_sha, cohort_sha, branch_sha)
    fixed['training_data'] = {
        'clips': 'MOVi-E train' + (' + mirrored copies' if MIRROR else '') + ' + new-scene train: ' + ','.join(DATASETS),
        'counts': _bank['counts'],
        'extra_cache_protocol_sha256': canonical_sha(json.loads((EXTRA / 'protocol.json').read_text())),
        'mirror_cache_protocol_sha256': canonical_sha(json.loads((MIRROR / 'protocol.json').read_text()))
        if MIRROR else None,
        'wrapper_sha256': digest(Path(__file__))}
    return fixed


sfd.protocol_fields = protocol_fields
sfd.CacheBank = RecordingBank

if __name__ == '__main__':
    sfd.main()

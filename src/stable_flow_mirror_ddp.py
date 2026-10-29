"""stable_flow_ddp.py, unchanged, on the train set doubled by mirrored clips (amendment F2).

Extra flag: --mirror-cache DIR (cache_mirrored_features.py output). Train IDs are
the 9,749 originals in their usual order followed by '<id>__mirror' copies, so
the fixed gain-calibration batch (train_ids[:8]) is the main run's. Validation,
showcase and every evaluation read only original clips. The mirror cache
protocol and this file's SHA enter the protocol signature.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stable_flow_ddp as sfd  # noqa: E402
from codec_research import CacheBank, digest, canonical_sha  # noqa: E402

SUFFIX = '__mirror'


def take(flag):
    i = sys.argv.index(flag)
    value = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    return value


MIRROR = Path(take('--mirror-cache'))


class MirrorBank(CacheBank):
    def __init__(self, data, cache):
        super().__init__(data, cache)
        self.mirror = CacheBank(data, MIRROR)
        if self.mirror.protocol.get('version') != 'wan-vae-feature-cache-mirror-v1' or \
                self.mirror.protocol['original_cache_protocol_sha256'] != self.protocol_sha:
            raise ValueError('Mirror cache does not derive from this feature cache')
        originals = super().ids('train')
        missing = [i for i in originals if not (MIRROR / 'records' / 'train' / (i + '.json')).exists()]
        if missing:
            raise ValueError(f'Mirror cache incomplete: {len(missing)} train clips missing')
        self.mirror_ids = [i + SUFFIX for i in originals]

    def ids(self, split):
        base = super().ids(split)
        return base + self.mirror_ids if split == 'train' else base

    def check_metadata(self, identifier):
        if identifier.endswith(SUFFIX):
            return self.mirror.check_metadata(identifier[:-len(SUFFIX)])
        return super().check_metadata(identifier)

    def features(self, identifier):
        if identifier.endswith(SUFFIX):
            return self.mirror.features(identifier[:-len(SUFFIX)])
        return super().features(identifier)


_protocol_fields = sfd.protocol_fields


def protocol_fields(a, world, source_sha, cohort_sha, branch_sha):
    fixed = _protocol_fields(a, world, source_sha, cohort_sha, branch_sha)
    fixed['training_data'] = {
        'clips': 'official train (9,749) + horizontally mirrored copies (VAE re-encoded); validation unmirrored',
        'mirror_cache_protocol_sha256': canonical_sha(json.loads((MIRROR / 'protocol.json').read_text())),
        'wrapper_sha256': digest(Path(__file__))}
    return fixed


sfd.protocol_fields = protocol_fields
sfd.CacheBank = MirrorBank

if __name__ == '__main__':
    sfd.main()

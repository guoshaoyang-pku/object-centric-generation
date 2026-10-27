"""CPU-only raw-pixel contracts; full CLI resume tests live separately."""
import copy
import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from codec_research import Codec, canonical_sha, digest
from codec_flow_research import state_sha, flow_state, load_flow_state
from eval_validation import clip_noise as object_clip_noise
from raw_pixel_codec import (pack_rgb, unpack_rgb, preprocess_rgb, FixedPixelDecoder,
                             RawPixelFlow, load_statistics)
from raw_pixel_flow_research import (RawPixelModel, fm_loss, training_noise,
                                     clip_noise, target_rgb_batch)

STATS_PATH = Path(__file__).resolve().parent.parent / 'control' / 'build_raw_pixel_stats.py'
spec = importlib.util.spec_from_file_location('raw_pixel_stats_builder_for_tests', STATS_PATH)
stats_builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats_builder)


class RawPixelContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(713)

    def model(self):
        codec = Codec('trajectory', objects=4, latent_dim=8, feature_width=64,
                      texture_kernels=0, encoder_depth=0, freeze_decoder=True)
        return RawPixelModel(codec, RawPixelFlow(64, 32, 1, 4),
                             FixedPixelDecoder([.42, .45, .48], [.21, .22, .23]))

    def test_pack_inverse_index_order_and_gradient(self):
        values = torch.arange(20 * 3 * 64 * 64, dtype=torch.float32).reshape(1, 20, 3, 64, 64)
        values.requires_grad_()
        packed = pack_rgb(values)
        self.assertEqual(packed.shape, (1, 1280, 192))
        self.assertTrue(torch.equal(unpack_rgb(packed), values))
        expected = values[0, 0, :, :8, :8].permute(1, 2, 0).reshape(-1)
        self.assertTrue(torch.equal(packed[0, 0], expected))
        self.assertTrue(torch.equal(packed[0, 1], values[0, 0, :, :8, 8:16].permute(1, 2, 0).reshape(-1)))
        self.assertTrue(torch.equal(packed[0, 64], values[0, 1, :, :8, :8].permute(1, 2, 0).reshape(-1)))
        unpack_rgb(packed).sum().backward()
        self.assertTrue(torch.equal(values.grad, torch.ones_like(values)))
        with self.assertRaises(ValueError):
            pack_rgb(values[..., :32, :])
        with self.assertRaises(ValueError):
            unpack_rgb(packed[:, :-1])

    def test_preprocess_exactly_matches_train_statistics(self):
        array = np.random.default_rng(713).integers(0, 256, (24, 128, 128, 3), dtype=np.uint8)
        self.assertTrue(torch.equal(preprocess_rgb(array), stats_builder.preprocess_rgb(array)))
        with self.assertRaises(ValueError):
            preprocess_rgb(array.astype(np.float32))

    def test_fixed_decoder_zero_parameters_and_roundtrip(self):
        decoder = FixedPixelDecoder([.42, .45, .48], [.21, .22, .23])
        self.assertEqual(sum(p.numel() for p in decoder.parameters()), 0)
        video = torch.rand(1, 20, 3, 64, 64)
        video[0, 0, :, 0, 0] = 0.
        video[0, 0, :, 0, 1] = 1.
        decoded = decoder.decode_tokens(decoder.encode_rgb(video))
        self.assertLessEqual(float((decoded - video).abs().max()), 2e-7)
        decoder.train()
        self.assertFalse(decoder.training)
        with self.assertRaises(ValueError):
            decoder.decode_tokens(decoder.encode_rgb(video), size=128)

    def test_noise_preserves_object_stream_and_time_draw(self):
        primary = torch.Generator().manual_seed(1016)
        reference = torch.Generator().manual_seed(1016)
        extra = torch.Generator().manual_seed(2027)
        noise, t = training_noise(2, primary, extra, 'cpu')
        expected_common = torch.randn(2, 250, 128, generator=reference).flatten(1)
        expected_t = torch.rand(2, generator=reference)
        self.assertTrue(torch.equal(noise.flatten(1)[:, :32000], expected_common))
        self.assertTrue(torch.equal(t, expected_t))
        self.assertTrue(torch.equal(primary.get_state(), reference.get_state()))
        other, other_t = training_noise(2, torch.Generator().manual_seed(1016),
                                        torch.Generator().manual_seed(2028), 'cpu')
        self.assertTrue(torch.equal(noise.flatten(1)[:, :32000], other.flatten(1)[:, :32000]))
        self.assertTrue(torch.equal(t, other_t))
        self.assertFalse(torch.equal(noise.flatten(1)[:, 32000:], other.flatten(1)[:, 32000:]))
        fixed = clip_noise(['heldout-fixed'], 17, 1280, 192, 'cpu', torch.float32)
        object_fixed = object_clip_noise(['heldout-fixed'], 17, 250, 128, 'cpu', torch.float32)
        self.assertTrue(torch.equal(fixed.flatten(1)[:, :32000], object_fixed.flatten(1)))

    def test_prefix_sample_and_fm_do_not_access_future_encoder_or_decoder(self):
        model = self.model().train()
        before = state_sha(model.codec)
        prefix, coords = torch.randn(1, 128, 48), torch.randn(1, 128, 3)
        self.assertEqual(list(inspect.signature(model.prefix_context).parameters), ['features', 'coords'])
        self.assertEqual(list(inspect.signature(model.sample).parameters), ['context', 'initial_noise', 'steps'])
        optimizer = torch.optim.AdamW(model.flow.parameters(), lr=1e-3)
        target = model.target_latent(torch.rand(1, 20, 3, 64, 64))[0]
        self.assertFalse(target.requires_grad)
        with patch.object(model.codec, 'encode_future', side_effect=AssertionError('Future posterior accessed')), \
             patch.object(model.decoder, 'decode_tokens', side_effect=AssertionError('Decoder accessed')), \
             patch.object(model.decoder, 'encode_rgb', side_effect=AssertionError('Target accessed during generation')):
            for _ in range(2):
                context = model.prefix_context(prefix, coords)
                self.assertFalse(context.requires_grad)
                optimizer.zero_grad(set_to_none=True)
                loss, _, _ = fm_loss(model, context, target, torch.randn_like(target), torch.tensor([.5]))
                loss.backward()
                optimizer.step()
            grad = model.flow.blocks[0].attn.in_proj_weight.grad
            self.assertTrue(bool(torch.isfinite(grad).all()))
            self.assertGreater(float(grad.square().sum()), 0.)
            with torch.inference_mode():
                sample = model.sample(context, torch.randn(1, 1280, 192), 2)
            self.assertTrue(bool(torch.isfinite(sample).all()))
        self.assertEqual(before, state_sha(model.codec))
        model.assert_frozen()

    def test_raw_target_path_never_queries_cached_future_features(self):
        array = np.random.default_rng(1).integers(0, 256, (24, 128, 128, 3), dtype=np.uint8)
        class Bank:
            def __init__(self): self.calls = []
            def rgb(self, identifier): self.calls.append(identifier); return array
            def batch(self, *args): raise AssertionError('Future VAE feature target used')
        bank = Bank()
        target = target_rgb_batch(bank, ['raw-id'], 'cpu')
        self.assertTrue(torch.equal(target[0], preprocess_rgb(array)))
        self.assertEqual(bank.calls, ['raw-id'])

    def test_default_parameter_count_and_state_binding(self):
        flow = RawPixelFlow()
        self.assertEqual(sum(p.numel() for p in flow.parameters()), 23_515_584)
        del flow
        model = self.model()
        state = flow_state(model.flow)
        target = self.model()
        load_flow_state(target.flow, state)
        self.assertEqual(state_sha(model.flow), state_sha(target.flow))
        invalid = copy.deepcopy(state)
        invalid['_extra_state']['target_contract']['tokens'] = 96
        with self.assertRaises(ValueError):
            load_flow_state(self.model().flow, invalid)

    def test_statistics_cover_all_train_only_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = []
            for index, split in enumerate(['train', 'train', 'validation']):
                array = np.random.default_rng(index).integers(0, 256, (24, 128, 128, 3), dtype=np.uint8)
                path = root / f'clip{index}.npy'
                np.save(path, array)
                records.append({'id': f'clip{index}', 'split': split, 'array': path.name,
                                'array_sha256': digest(path), 'shape': list(array.shape), 'dtype': 'uint8'})
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'samples': records}))
            bank = SimpleNamespace(manifest_sha=digest(manifest), records={r['id']: r for r in records})
            bank.ids = lambda split: sorted(r['id'] for r in records if r['split'] == split)
            valid = stats_builder.collect_stats(root, bank.manifest_sha)
            path = root / 'stats.json'
            def save(document, rebind=True):
                document = copy.deepcopy(document)
                if rebind:
                    document['stats_sha256'] = canonical_sha({k: v for k, v in document.items() if k != 'stats_sha256'})
                path.write_text(json.dumps(document))
                return digest(path)
            actual, identity = load_statistics(path, save(valid), bank)
            self.assertEqual(actual['count'], 2)
            self.assertEqual(identity['count'], 2)
            with self.assertRaisesRegex(ValueError, 'file SHA'):
                load_statistics(path, '0' * 64, bank)
            altered = copy.deepcopy(valid)
            altered['channel_mean'][0] += .1
            with self.assertRaisesRegex(ValueError, 'canonical SHA'):
                load_statistics(path, save(altered, rebind=False), bank)
            altered = copy.deepcopy(valid)
            altered['records'].pop()
            altered['count'] = 1
            with self.assertRaisesRegex(ValueError, 'exactly all train'):
                load_statistics(path, save(altered), bank)
            altered = copy.deepcopy(valid)
            altered['records'].append(records[-1])
            altered['count'] = 3
            with self.assertRaisesRegex(ValueError, 'exactly all train'):
                load_statistics(path, save(altered), bank)
            altered = copy.deepcopy(valid)
            altered['source_code_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'source implementation'):
                load_statistics(path, save(altered), bank)


if __name__ == '__main__':
    unittest.main(verbosity=2)

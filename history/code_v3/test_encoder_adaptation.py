"""Independent CPU admission tests for the frozen-decoder encoder experiment.

The source fixture uses the actual unmodified v1 Codec implementation and its
five bound source files. No VAE, Wan backbone, remote process or GPU is loaded.
"""
import copy
import importlib.util
import io
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

import codec_research as new


ROOT = Path(__file__).resolve().parents[3]
OLD_SOURCE = Path(os.environ.get('ENCODER_SOURCE_DIR', str(ROOT / 'slotformer-demo' / 'joint_video'))).resolve()
SOURCE_NAMES = ['codec_research.py', 'model.py', 'trajectory_codec.py',
                'wan_model.py', 'diagnose_persistence.py']
spec = importlib.util.spec_from_file_location('encoder_trial_v1_source', OLD_SOURCE / 'codec_research.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return set(a) == set(b) and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(tree_equal(x, y) for x, y in zip(a, b))
    return a == b


class EncoderAdaptationContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(713)
        self.temp = tempfile.TemporaryDirectory(prefix='encoder-adaptation-test-')
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.sources = self.folder / 'source'
        self.sources.mkdir()
        for name in SOURCE_NAMES:
            shutil.copyfile(OLD_SOURCE / name, self.sources / name)
        self.baseline = old.Codec('trajectory', objects=8, latent_dim=8,
                                 feature_width=64, subkernels=2, texture_kernels=0)
        self.bank = SimpleNamespace(manifest_sha='source-data-test-sha', protocol_sha='source-cache-test-sha')
        bound = {'model_config': self.baseline.config,
                 'data_manifest_sha256': self.bank.manifest_sha,
                 'cache_protocol_sha256': self.bank.protocol_sha,
                 'source_sha256': {name: new.digest(self.sources / name) for name in SOURCE_NAMES}}
        self.payload = {'format': 'codec_research_v1', 'step': 37,
                        'protocol': bound, 'protocol_sha256': new.canonical_sha(bound),
                        'model': self.baseline.state_dict(), **bound}
        self.checkpoint = self.folder / 'source.pt'
        torch.save(self.payload, self.checkpoint)
        self.sha = new.digest(self.checkpoint)
        self.features = torch.randn(2, 11, 48)
        self.local = torch.rand(11, 3) * 2 - 1
        self.absolute = self.local.clone()
        self.absolute[:, 2] = (self.absolute[:, 2] + 1) * (19 / 23) + 8 / 23 - 1
        self.queries = torch.rand(2, 17, 3) * 2 - 1
        self.truth = torch.rand(2, 17, 3)

    def model(self, depth):
        model = new.Codec(**self.baseline.config, encoder_depth=depth, freeze_decoder=True)
        identity = new.import_source_codec(model, self.checkpoint, self.bank, self.sources, self.sha)
        self.assertEqual(identity['checkpoint_step'], 37)
        self.assertEqual(identity['checkpoint_sha256'], self.sha)
        self.assertFalse(identity['optimizer_restored'])
        self.assertEqual(identity['loaded_modules'], ['visual_adapter', 'posterior', 'decoder'])
        return model

    def encode(self, model, seed=911):
        torch.manual_seed(seed)
        return model.encode_future(self.features, self.absolute, self.local)

    def update(self, model, optimizer):
        model.train()
        self.assertFalse(model.decoder.training)
        optimizer.zero_grad(set_to_none=True)
        z, mu, logvar, kl = self.encode(model)
        predicted = model.render_queries(z, self.queries)
        loss = F.mse_loss(predicted, self.truth) + 1e-5 * kl
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        for layer in model.encoder_mixer.layers:
            for projection in [layer.self_attn.out_proj, layer.linear2]:
                gradient = projection.weight.grad
                self.assertIsNotNone(gradient)
                self.assertTrue(bool(torch.isfinite(gradient).all()))
                self.assertGreater(float(gradient.square().sum()), 0)
        optimizer.step()
        return float(loss.detach())

    def test_identity_import_matches_real_v1_in_train_and_eval(self):
        plain, mixed = self.model(0), self.model(2)
        for key, value in self.baseline.state_dict().items():
            self.assertTrue(torch.equal(value, plain.state_dict()[key]), key)
            self.assertTrue(torch.equal(value, mixed.state_dict()[key]), key)
        for training in [True, False]:
            for model in [self.baseline, plain, mixed]:
                model.train(training)
            context = torch.enable_grad() if training else torch.inference_mode()
            with context:
                reference = self.encode(self.baseline)
                for model in [plain, mixed]:
                    result = self.encode(model)
                    for expected, actual in zip(reference, result):
                        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
                    torch.testing.assert_close(self.baseline.render_queries(reference[0], self.queries),
                                               model.render_queries(result[0], self.queries), rtol=0, atol=0)
                torch.testing.assert_close(plain.adapt_features(self.features, self.absolute),
                                           mixed.adapt_features(self.features, self.absolute), rtol=0, atol=0)

    def test_render_gradient_reaches_mixer_and_decoder_stays_identical(self):
        model = self.model(2)
        before = new.module_state_sha(model.decoder)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        self.update(model, optimizer)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.decoder.parameters()))
        self.assertEqual(before, model.assert_decoder_unchanged(before))
        self.assertTrue(any(bool(p.detach().abs().sum() > 0)
                            for layer in model.encoder_mixer.layers
                            for p in [layer.self_attn.out_proj.weight, layer.linear2.weight]))
        with torch.no_grad():
            next(model.decoder.parameters()).flatten()[0] += 1
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            model.assert_decoder_unchanged(before)

    def test_config_checkpoint_optimizer_roundtrip_and_next_update(self):
        model = self.model(2)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        self.update(model, optimizer)
        checkpoint = io.BytesIO()
        torch.save({'config': model.config, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint)
        checkpoint.seek(0)
        loaded = torch.load(checkpoint, map_location='cpu', weights_only=False)
        restored = new.Codec(**loaded['config'])
        restored.load_state_dict(loaded['model'], strict=True)
        restored_optimizer = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=1e-3)
        restored_optimizer.load_state_dict(loaded['optimizer'])
        self.assertTrue(tree_equal(model.state_dict(), restored.state_dict()))
        self.assertTrue(tree_equal(optimizer.state_dict(), restored_optimizer.state_dict()))
        for expected, actual in zip(self.encode(model), self.encode(restored)):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        self.assertEqual(self.update(model, optimizer), self.update(restored, restored_optimizer))
        self.assertTrue(tree_equal(model.state_dict(), restored.state_dict()))
        self.assertTrue(tree_equal(optimizer.state_dict(), restored_optimizer.state_dict()))
        restored.assert_decoder_unchanged()

    def test_import_rejects_checkpoint_data_source_and_state_mismatch(self):
        model = new.Codec(**self.baseline.config, encoder_depth=2, freeze_decoder=True)
        with self.assertRaises(ValueError):
            new.import_source_codec(model, self.checkpoint, self.bank, self.sources, 'bad-sha')
        bad_bank = SimpleNamespace(manifest_sha='other-data', protocol_sha=self.bank.protocol_sha)
        with self.assertRaisesRegex(ValueError, 'data/cache'):
            new.import_source_codec(model, self.checkpoint, bad_bank, self.sources, self.sha)
        copied = self.sources / 'model.py'
        copied.write_bytes(copied.read_bytes() + b'\n# deliberate test-only change\n')
        with self.assertRaisesRegex(ValueError, 'implementation'):
            new.import_source_codec(model, self.checkpoint, self.bank, self.sources, self.sha)
        shutil.copyfile(OLD_SOURCE / 'model.py', copied)
        missing = copy.deepcopy(self.payload)
        missing['model'].pop('visual_adapter.0.weight')
        missing_path = self.folder / 'missing.pt'
        torch.save(missing, missing_path)
        with self.assertRaisesRegex(ValueError, 'missing or adding'):
            new.import_source_codec(model, missing_path, self.bank, self.sources, new.digest(missing_path))
        corrupt = copy.deepcopy(self.payload)
        corrupt['protocol_sha256'] = 'wrong-protocol'
        corrupt_path = self.folder / 'corrupt.pt'
        torch.save(corrupt, corrupt_path)
        with self.assertRaisesRegex(ValueError, 'protocol'):
            new.import_source_codec(model, corrupt_path, self.bank, self.sources, new.digest(corrupt_path))

    def test_actual_256_width_counts_and_configuration_gate(self):
        plain = new.Codec('trajectory', encoder_depth=0, freeze_decoder=True)
        mixed = new.Codec('trajectory', encoder_depth=2, freeze_decoder=True)
        counts0, counts2 = plain.parameter_counts(), mixed.parameter_counts()
        # Two standard width256 / FF1024 pre-norm TransformerEncoder layers.
        self.assertEqual(counts2['encoder_mixer'], 1_579_520)
        self.assertEqual(counts2['total'] - counts0['total'], counts2['encoder_mixer'])
        self.assertEqual(counts2['trainable'] - counts0['trainable'], counts2['encoder_mixer'])
        self.assertEqual(counts0['frozen'], counts2['frozen'])
        self.assertEqual(counts2['frozen'], counts2['decoder'])
        self.assertEqual(mixed.config['encoder_depth'], 2)
        self.assertTrue(mixed.config['freeze_decoder'])
        for kwargs in [{'encoder_depth': 1}, {'feature_width': 62}]:
            with self.assertRaises(ValueError):
                new.Codec('trajectory', **kwargs)


if __name__ == '__main__':
    unittest.main(verbosity=2)

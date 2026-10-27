"""CPU contract tests using actual WanObjectFlow/LoRA with a tiny Wan mock.

These test routing and checkpoint semantics, not pretrained model quality.
The separate CLI smoke also checks optimizer/RNG resume and rendered GIFs.
"""
import copy
import unittest

import torch

from codec_research import Codec
from codec_flow_research import (ConditionalObjectFlow, fm_loss, flow_state,
                                 load_flow_state, make_wan_flow, position_policy, state_sha)
from wan_model import WanObjectFlow
from wan_token_adapter import WanTokenAdapter
from test_wan_token_adapter import TinyBackbone


class CodecFlowContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(713)

    def build(self, variant='trajectory', objects=250, subkernels=4):
        codec = Codec(variant, objects=objects, latent_dim=7, feature_width=64,
                      subkernels=subkernels, texture_kernels=0)
        flow = make_wan_flow(TinyBackbone(), codec, lora_rank=2, lora_alpha=2.)
        return ConditionalObjectFlow(codec, flow)

    def test_c3_64_by_16_full_codec_shapes_gradients_and_counts(self):
        codec = Codec('trajectory3d', objects=64, latent_dim=128,
                      feature_width=256, subkernels=16, texture_kernels=64)
        counts = codec.parameter_counts()
        self.assertEqual(counts['geometry_fields_per_object'], 115)
        self.assertEqual(counts['texture_fields_per_object'], 768)
        self.assertEqual(counts['geometry_fields_per_clip'] + counts['texture_fields_per_clip'], 56512)
        self.assertEqual(counts['latent_values_per_clip'], 8192)
        self.assertEqual(counts['total'], 1276531)
        features = torch.randn(1, 24, 48)
        coords = torch.rand(1, 24, 3) * 2 - 1
        _, mu, logvar, _ = codec.encode_future(features, coords, coords)
        self.assertEqual(tuple(mu.shape), (1, 64, 128))
        self.assertEqual(tuple(logvar.shape), tuple(mu.shape))
        rendered = codec.render_queries(mu, coords[:, :17])
        self.assertEqual(tuple(rendered.shape), (1, 17, 3))
        self.assertTrue(torch.isfinite(rendered).all())
        rendered.square().mean().backward()
        grads = [p.grad for p in codec.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(g.abs().sum().item() for g in grads), 0.)

    def test_variable_count_uses_exact_direct_token_count_without_volume_path(self):
        for count in [64, 250]:
            model = self.build('trajectory3d', objects=count, subkernels=16)
            flow = model.flow
            self.assertEqual(flow.position_mode, 'object_xy')
            self.assertEqual(tuple(flow.object_anchors.shape), (count, 2))
            observed_counts = []
            hooks = [block.register_forward_pre_hook(
                lambda module, args: observed_counts.append(args[0].shape[1]))
                for block in flow.backbone.blocks]
            def forbidden(*args, **kwargs):
                raise AssertionError('Direct object flow entered native volume/grid path')
            flow.backbone.forward = forbidden
            flow.backbone.patch_embedding.forward = forbidden
            flow.backbone.rope.forward = forbidden
            z = torch.randn(1, count, 7, requires_grad=True)
            output = flow(z, torch.tensor([.4]), torch.randn(1, 8, 64))
            self.assertEqual(tuple(output.shape), (1, count, 7))
            output.square().mean().backward()
            self.assertTrue(observed_counts)
            self.assertEqual(set(observed_counts), {count})
            self.assertTrue(torch.isfinite(z.grad).all())
            self.assertGreater(z.grad.abs().sum().item(), 0.)
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for n, p in flow.named_parameters() if 'lora_B' in n))
            for hook in hooks:
                hook.remove()
            with self.assertRaises(ValueError):
                flow(torch.randn(1, 250 if count == 64 else 64, 7),
                     torch.tensor([.4]), torch.randn(1, 8, 64))

    def test_64_token_checkpoint_roundtrip_and_legacy_count_rejection(self):
        flow = self.build('trajectory3d', objects=64, subkernels=16).flow.eval()
        saved = copy.deepcopy(flow_state(flow))
        copied = copy.deepcopy(flow)
        with torch.no_grad():
            for p in copied.parameters():
                if p.requires_grad:
                    p.add_(.2)
        load_flow_state(copied, saved)
        z, t, context = torch.randn(1, 64, 7), torch.tensor([.6]), torch.randn(1, 8, 64)
        torch.testing.assert_close(copied(z, t, context), flow(z, t, context), rtol=0, atol=0)
        with self.assertRaises((ValueError, RuntimeError)):
            load_flow_state(self.build('trajectory3d', objects=250).flow, saved)
        for variant in ['xyt_t', 'xyt_xy']:
            with self.assertRaisesRegex(ValueError, '250-token'):
                self.build(variant, objects=64)

    def test_time_partitioned_codec_keeps_distinct_temporal_positions(self):
        for variant in ['xyt_t', 'xyt_xy']:
            model = self.build(variant)
            self.assertEqual(len(torch.unique(model.codec.decoder.anchors[:, :2], dim=0)), 50)
            self.assertEqual(position_policy(model.codec), 'legacy_grid')
            self.assertIsNone(model.flow.object_anchors)
            self.assertIsNotNone(model.flow._position(torch.randn(1, 250, 7)))
        model = self.build('trajectory')
        self.assertEqual(len(torch.unique(model.codec.decoder.anchors, dim=0)), 250)
        self.assertEqual(position_policy(model.codec), 'object_xy')
        self.assertIsNone(model.flow._position(torch.randn(1, 250, 7)))
        model.codec.variant = 'trajectory3d'
        self.assertEqual(position_policy(model.codec), 'object_xy')
        with torch.no_grad():
            model.codec.decoder.anchors[1].copy_(model.codec.decoder.anchors[0])
        with self.assertRaises(ValueError):
            position_policy(model.codec)

    def test_checkpoint_retains_every_adapter_but_no_frozen_weights(self):
        model = self.build()
        saved = flow_state(model.flow)
        names = {n for n, p in model.flow.named_parameters() if p.requires_grad}
        self.assertTrue(names.issubset(saved))
        for key in ['input.weight', 'input.bias', 'output.weight', 'output.bias',
                    'condition.0.weight', 'condition.1.weight', 'object_anchors', '_extra_state']:
            self.assertIn(key, saved)
        self.assertTrue(any('lora_A' in n for n in saved))
        self.assertTrue(any('lora_B' in n for n in saved))
        self.assertNotIn('backbone.patch_embedding.weight', saved)
        self.assertFalse(any(n.endswith('.base.weight') for n in saved))
        copied = copy.deepcopy(model.flow)
        with torch.no_grad():
            for p in copied.parameters():
                if p.requires_grad:
                    p.add_(.3)
        load_flow_state(copied, saved)
        for name, p in copied.named_parameters():
            if name in names:
                torch.testing.assert_close(p, dict(model.flow.named_parameters())[name], rtol=0, atol=0)
        for key in ['input.weight', 'output.weight', '_extra_state']:
            incomplete = dict(saved)
            del incomplete[key]
            with self.assertRaises(ValueError):
                load_flow_state(copy.deepcopy(model.flow), incomplete)
        extra = dict(saved)
        extra['backbone.patch_embedding.weight'] = model.flow.backbone.patch_embedding.weight.detach().clone()
        with self.assertRaises(ValueError):
            load_flow_state(copy.deepcopy(model.flow), extra)

    def test_checkpoint_rejects_anchor_buffer_and_policy_mutation(self):
        model = self.build()
        for key in ['object_anchors', '_extra_state']:
            saved = copy.deepcopy(flow_state(model.flow))
            if key == 'object_anchors':
                saved[key][0, 0] += .1
            else:
                saved[key]['position_mode'] = 'legacy_grid'
            with self.assertRaises(ValueError):
                load_flow_state(copy.deepcopy(model.flow), saved)
        grid = self.build('xyt_t').flow
        saved = flow_state(grid)
        self.assertNotIn('object_anchors', saved)
        load_flow_state(copy.deepcopy(grid), saved)

    def test_actual_lora_backprop_and_frozen_codec_mode(self):
        model = self.build().train()
        self.assertFalse(model.codec.training)
        self.assertFalse(any(p.requires_grad for p in model.codec.parameters()))
        old_codec = state_sha(model.codec)
        frozen = {n: p.detach().clone() for n, p in model.flow.named_parameters() if not p.requires_grad}
        features = torch.randn(1, 8, 48, requires_grad=True)
        coords = torch.rand(1, 8, 3) * 2 - 1
        context = model.prefix_context(features, coords)
        self.assertFalse(context.requires_grad)
        target = torch.randn(1, 250, 7, requires_grad=True)
        noise = torch.randn_like(target)
        t = torch.tensor([.6])
        loss, _, _ = fm_loss(model, context, target, noise, t)
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertIsNone(features.grad)
        self.assertTrue(all(p.grad is None for p in model.codec.parameters()))
        for name in ['input.weight', 'output.weight', 'condition.0.weight']:
            grad = dict(model.flow.named_parameters())[name].grad
            self.assertIsNotNone(grad)
            self.assertTrue(bool(torch.isfinite(grad).all()))
            self.assertGreater(grad.abs().sum().item(), 0.)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for n, p in model.flow.named_parameters() if 'lora_B' in n))
        optimizer = torch.optim.AdamW([p for p in model.flow.parameters() if p.requires_grad], lr=1e-4)
        optimizer.step()
        self.assertEqual(state_sha(model.codec), old_codec)
        for name, old in frozen.items():
            torch.testing.assert_close(dict(model.flow.named_parameters())[name], old, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()

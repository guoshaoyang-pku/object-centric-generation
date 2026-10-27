"""CPU tests for native object-flow contracts, not video quality."""
import copy
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from codec_research import Codec
from codec_flow_research import (ConditionalObjectFlow, ToyFlow, build_model, fm_loss,
                                 flow_state, load_flow_state, state_sha)
from small_object_flow import SmallObjectFlow


class SmallFlowContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(713)

    def build(self):
        codec = Codec('trajectory', objects=8, latent_dim=8, feature_width=64,
                      subkernels=2, texture_kernels=0, encoder_depth=2, freeze_decoder=True)
        flow = SmallObjectFlow(8, 8, 64, codec.decoder.anchors, width=64, depth=2, heads=4)
        return ConditionalObjectFlow(codec, flow)

    def train_step(self, model, optimizer, context):
        torch.manual_seed(911)
        target = torch.randn(2, 8, 8)
        noise = torch.randn_like(target)
        time = torch.tensor([.2, .8])
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = fm_loss(model, context, target, noise, time)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                            for p in model.flow.parameters()))
        optimizer.step()

    def test_actual_default_parameter_count_and_shape(self):
        codec = Codec('trajectory', texture_kernels=0)
        flow = SmallObjectFlow(250, 128, 256, codec.decoder.anchors)
        self.assertEqual(sum(p.numel() for p in flow.parameters()), 23_070_848)
        self.assertTrue(all(p.requires_grad and p.dtype == torch.float32 for p in flow.parameters()))
        output = flow(torch.randn(1, 250, 128), torch.tensor([.5]), torch.randn(1, 12, 256))
        self.assertEqual(output.shape, (1, 250, 128))
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        self.assertEqual(flow.get_extra_state()['backend'], 'small')
        with self.assertRaises(ValueError):
            flow(torch.randn(1, 249, 128), torch.tensor([.5]), torch.randn(1, 12, 256))

    def test_all_object_and_prefix_interactions_with_frozen_codec(self):
        model = self.build().train()
        before = state_sha(model.codec)
        context = torch.randn(2, 11, 64)
        optimizer = torch.optim.AdamW(model.flow.parameters(), lr=1e-3)
        # Output starts at zero. A second step exercises every hidden block.
        self.train_step(model, optimizer, context)
        self.train_step(model, optimizer, context)
        for key in ['input.weight', 'slots', 'condition.weight', 'blocks.0.attn.in_proj_weight',
                    'blocks.0.cross.in_proj_weight', 'blocks.1.ff.0.weight', 'out.weight']:
            self.assertGreater(float(dict(model.flow.named_parameters())[key].grad.square().sum()), 0, key)
        z = torch.randn(1, 8, 8, requires_grad=True)
        c = torch.randn(1, 11, 64, requires_grad=True)
        y = model.flow(z, torch.tensor([.5]), c)[0, 0].square().sum()
        gz, gc = torch.autograd.grad(y, (z, c))
        self.assertGreater(float(gz[0, -1].square().sum()), 0)  # remote object reaches object 0
        self.assertGreater(float(gc[0, -1].square().sum()), 0)  # remote prefix token reaches object 0
        self.assertEqual(before, state_sha(model.codec))
        self.assertFalse(model.codec.training)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.codec.parameters()))

    def test_sample_has_no_future_path_and_prefix_adapter_matches(self):
        model = self.build().eval()
        prefix = torch.randn(1, 11, 48)
        coords = torch.randn(1, 11, 3)
        with patch.object(model.codec, 'encode_future', side_effect=AssertionError('future read')):
            context = model.prefix_context(prefix, coords)
            torch.testing.assert_close(context, model.codec.adapt_features(prefix, coords), rtol=0, atol=0)
            sample = model.sample(context, torch.randn(1, 8, 8), 4)
        self.assertTrue(bool(torch.isfinite(sample).all()))
        self.assertEqual(list(inspect.signature(model.sample).parameters), ['context', 'initial_noise', 'steps'])

    def test_checkpoint_contract_rejects_other_backend_config_or_anchors(self):
        model = self.build()
        saved = flow_state(model.flow)
        same = self.build()
        load_flow_state(same.flow, saved)
        self.assertEqual(state_sha(model.flow), state_sha(same.flow))
        for mutate in ['backend', 'config', 'anchors', 'missing']:
            invalid = copy.deepcopy(saved)
            if mutate == 'backend':
                invalid['_extra_state']['backend'] = 'wan'
            elif mutate == 'config':
                invalid['_extra_state']['model_config']['heads'] = 2
            elif mutate == 'anchors':
                invalid['slot_anchors'][0, 0] += .1
            else:
                invalid.pop('out.weight')
            with self.assertRaises((ValueError, RuntimeError)):
                load_flow_state(self.build().flow, invalid)
        toy = ToyFlow(8, 64, model.codec.decoder.anchors)
        with self.assertRaises(ValueError):
            load_flow_state(model.flow, flow_state(toy))

    def test_small_rejects_wan_inputs_before_any_codec_or_backbone_load(self):
        for kwargs in [{'pretrained': Path('/unused/Wan'), 'flow_warmstart': None},
                       {'pretrained': None, 'flow_warmstart': Path('/unused/wan.pt')}]:
            args = SimpleNamespace(backend='small', **kwargs)
            with patch('codec_flow_research.load_codec', side_effect=AssertionError('loaded codec')):
                with self.assertRaisesRegex(ValueError, 'Wan weights'):
                    build_model(args, None, torch.device('cpu'))

    def test_small_build_never_touches_pretrained_loader(self):
        codec = self.build().codec
        args = SimpleNamespace(backend='small', pretrained=None, flow_warmstart=None,
                               codec_checkpoint=Path('/fixture/codec.pt'), codec_sha='fixture',
                               seed=713, small_width=64, small_depth=2, small_heads=4)
        with patch('codec_flow_research.load_codec', return_value=(codec, {'fixture': True})), \
             patch('codec_flow_research.transformer_fingerprint', side_effect=AssertionError('Wan weights read')), \
             patch('codec_flow_research.make_wan_flow', side_effect=AssertionError('Wan constructed')):
            model, identity = build_model(args, None, torch.device('cpu'))
        self.assertIsInstance(model.flow, SmallObjectFlow)
        self.assertEqual(identity['backend'], 'small')
        self.assertEqual(identity['parameter_counts']['lora'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)

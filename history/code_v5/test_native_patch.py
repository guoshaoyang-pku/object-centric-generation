"""CPU contracts for lossless patch packing and the future-free predictor."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from codec_research import Codec
from codec_flow_research import state_sha, flow_state, load_flow_state
from native_patch_codec import (pack_features, unpack_tokens, NativePatchFlow, FrozenVAEDecoder,
                                MockVAE, check_future_grid)
from cache_vae_features import coordinates
from patch_flow_research import NativePatchModel, fm_loss, clip_noise
from eval_validation import clip_noise as object_clip_noise


class NativePatchContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(713)

    def model(self):
        codec = Codec('trajectory', objects=4, latent_dim=8, feature_width=64,
                      texture_kernels=0, encoder_depth=0, freeze_decoder=True)
        return NativePatchModel(codec, NativePatchFlow(64,64,2,4), FrozenVAEDecoder(MockVAE()))

    def test_pack_inverse_exact_index_order_and_gradient(self):
        values = torch.arange(2*384*48,dtype=torch.float32).reshape(2,384,48).requires_grad_()
        packed = pack_features(values)
        self.assertEqual(packed.shape,(2,96,192))
        self.assertTrue(torch.equal(unpack_tokens(packed),values))
        grid=values.reshape(2,6,8,8,48)
        expected=torch.cat([grid[0,0,0,0],grid[0,0,0,1],grid[0,0,1,0],grid[0,0,1,1]])
        self.assertTrue(torch.equal(packed[0,0],expected))
        unpack_tokens(packed).sum().backward()
        self.assertTrue(torch.equal(values.grad,torch.ones_like(values)))

    def test_regular_coordinates_fail_closed(self):
        absolute,local=coordinates((1,48,6,8,8),20,4)
        a=torch.from_numpy(absolute)[None];l=torch.from_numpy(local)[None]
        check_future_grid(a,l)
        with self.assertRaises(ValueError):check_future_grid(a.flip(1),l)
        with self.assertRaises(ValueError):pack_features(torch.randn(1,383,48))

    def test_noise_uses_same_object_stream_prefix(self):
        expected=object_clip_noise(['fixed-id'],17,250,128,'cpu',torch.float32).flatten(1)[:,:18432]
        actual=clip_noise(['fixed-id'],17,96,192,'cpu',torch.float32).flatten(1)
        self.assertTrue(torch.equal(expected,actual))

    def test_mock_reference_decode_exact_and_frozen(self):
        decoder=FrozenVAEDecoder(MockVAE())
        features=torch.randn(1,384,48)
        expected=decoder.reference_decode_cached(features)
        actual=decoder.decode_tokens(pack_features(features),128)
        self.assertEqual(actual.shape,(1,20,3,128,128))
        self.assertTrue(torch.equal(actual,expected))
        decoder.train()
        self.assertFalse(decoder.training)
        self.assertTrue(all(not p.requires_grad for p in decoder.parameters()))

    def test_training_never_decodes_or_reads_future_encoder(self):
        model=self.model().train();before=state_sha(model.codec)
        prefix=torch.randn(2,128,48);coords=torch.randn(2,128,3)
        absolute,local=coordinates((1,48,6,8,8),20,4)
        absolute=torch.from_numpy(absolute)[None].expand(2,-1,-1);local=torch.from_numpy(local)[None].expand(2,-1,-1)
        optimizer=torch.optim.AdamW(model.flow.parameters(),lr=1e-3)
        with patch.object(model.codec,'encode_future',side_effect=AssertionError('object posterior used')), \
             patch.object(model.decoder,'decode_tokens',side_effect=AssertionError('decoder entered training')):
            for _ in range(2):
                context=model.prefix_context(prefix,coords)
                target,*_=model.target_latent(torch.randn(2,384,48),absolute,local)
                self.assertFalse(target.requires_grad)
                optimizer.zero_grad(set_to_none=True)
                loss,*_=fm_loss(model,context,target,torch.randn_like(target),torch.tensor([.2,.7]))
                loss.backward();optimizer.step()
            self.assertGreater(float(model.flow.blocks[0].attn.in_proj_weight.grad.square().sum()),0)
            generated=model.sample(context[:1],torch.randn(1,96,192),4)
        self.assertTrue(bool(torch.isfinite(generated).all()))
        self.assertEqual(before,state_sha(model.codec));model.assert_frozen()

    def test_patch_state_contract_and_default_count(self):
        flow=NativePatchFlow();self.assertEqual(sum(p.numel() for p in flow.parameters()),23_060_928)
        small=self.model().flow;state=flow_state(small);other=self.model().flow
        load_flow_state(other,state)
        self.assertEqual(state_sha(small),state_sha(other))
        invalid=copy.deepcopy(state);invalid['_extra_state']['target_contract']['token_width']=128
        with self.assertRaises(ValueError):load_flow_state(self.model().flow,invalid)


if __name__=='__main__':unittest.main(verbosity=2)

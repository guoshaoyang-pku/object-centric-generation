"""CPU contracts for direct tokens; no pretrained downloads or GPU required.

The mock backbone implements Wan's public tensor contract independently of the
new adapter and includes position-dependent self attention, cross attention,
time modulation and nontrivial output normalization. A real Wan GPU equivalence
check is still required before training deployment.

Run: python -m unittest discover -s slotformer-demo/joint_video -p test_wan_token_adapter.py
"""
import copy
import math
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from wan_model import WanObjectFlow
from wan_token_adapter import WanTokenAdapter, fused_projection_tensors, patch_projection_weight


class TinyRope(nn.Module):
    def forward(self, volume):
        _, _, t, h, w = volume.shape
        coords = torch.cartesian_prod(torch.arange(t), torch.arange(h // 2), torch.arange(w // 2))
        phase = (coords * torch.tensor([.2, .3, .7])).sum(-1).to(volume.device)
        return phase.cos()[None, :, None], phase.sin()[None, :, None]


class TinyAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.to_q = nn.Linear(width, width)
        self.to_k = nn.Linear(width, width)
        self.to_v = nn.Linear(width, width)
        self.to_out = nn.Sequential(nn.Linear(width, width))

    def forward(self, x, context=None, rotary=None):
        context = x if context is None else context
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        if rotary is not None:
            c, s = rotary
            def rotate(a):
                first, second = a.chunk(2, -1)
                return torch.cat([first * c - second * s, first * s + second * c], -1)
            q, k = rotate(q), rotate(k)
        weights = (q @ k.transpose(-1, -2) / math.sqrt(x.shape[-1])).softmax(-1)
        return self.to_out(weights @ v)


class TinyBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(width)
        self.norm3 = nn.LayerNorm(width, elementwise_affine=False)
        self.attn1, self.attn2 = TinyAttention(width), TinyAttention(width)
        self.ffn = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, width) / width**.5)

    def forward(self, hidden, context, modulation, rotary):
        if modulation.ndim == 4:
            terms = (self.scale_shift_table.unsqueeze(0) + modulation).chunk(6, 2)
            a, b, c, d, e, f = [x.squeeze(2) for x in terms]
        else:
            a, b, c, d, e, f = (self.scale_shift_table + modulation).chunk(6, 1)
        hidden = hidden + self.attn1(self.norm1(hidden) * (1 + b) + a, rotary=rotary) * c
        hidden = hidden + self.attn2(self.norm2(hidden), context)
        return hidden + self.ffn(self.norm3(hidden) * (1 + e) + d) * f


class TinyCondition(nn.Module):
    def __init__(self, text_width, width):
        super().__init__()
        self.text = nn.Linear(text_width, width)
        self.time = nn.Linear(2, width)
        self.modulation = nn.Linear(width, width * 6)

    def forward(self, timestep, context, encoder_hidden_states_image=None, timestep_seq_len=None):
        phase = timestep / 1000
        temb = self.time(torch.stack([phase, phase.square()], -1))
        if timestep_seq_len is not None:
            temb = temb.unflatten(0, (-1, timestep_seq_len))
        return temb, self.modulation(temb), self.text(context), None


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        channels, width, text_width = 3, 12, 9
        self.config = SimpleNamespace(in_channels=channels, patch_size=(1, 2, 2), text_dim=text_width)
        self.patch_embedding = nn.Conv3d(channels, width, (1, 2, 2), stride=(1, 2, 2))
        self.rope = TinyRope()
        self.condition_embedder = TinyCondition(text_width, width)
        self.blocks = nn.ModuleList([TinyBlock(width), TinyBlock(width)])
        self.norm_out = nn.LayerNorm(width, elementwise_affine=False)
        self.proj_out = nn.Linear(width, channels * 4)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, width) / width**.5)
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def _gradient_checkpointing_func(self, fn, *args):
        return checkpoint(fn, *args, use_reentrant=False)

    def forward(self, hidden_states, encoder_hidden_states, timestep, return_dict=False):
        # Independent native volume -> patches -> blocks -> volume reference.
        b, _, t, h, w = hidden_states.shape
        pt, ph, pw = self.config.patch_size
        nt, nh, nw = t // pt, h // ph, w // pw
        rotary = self.rope(hidden_states)
        hidden = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
        seq_len = timestep.shape[1] if timestep.ndim == 2 else None
        if seq_len is not None:
            timestep = timestep.flatten()
        temb, modulation, context, _ = self.condition_embedder(
            timestep, encoder_hidden_states, timestep_seq_len=seq_len)
        modulation = modulation.unflatten(2 if seq_len is not None else 1, (6, -1))
        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden = self._gradient_checkpointing_func(block, hidden, context, modulation, rotary)
            else:
                hidden = block(hidden, context, modulation, rotary)
        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, 2)
            shift, scale = shift.squeeze(2), scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, 1)
        hidden = self.proj_out(self.norm_out(hidden) * (1 + scale) + shift)
        hidden = hidden.reshape(b, nt, nh, nw, pt, ph, pw, -1)
        hidden = hidden.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return (hidden.flatten(6, 7).flatten(4, 5).flatten(2, 3),)


class WanTokenAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(20261027)
        self.legacy = WanObjectFlow(TinyBackbone(), latent_dim=7, feature_width=5, objects=250, rank=2, alpha=2.)
        # Nonzero biases and nonzero LoRA B catch lost affine terms/adapters.
        with torch.no_grad():
            self.legacy.input.bias.normal_()
            self.legacy.output.bias.normal_()
            for n, p in self.legacy.named_parameters():
                if "lora_B" in n:
                    p.normal_(std=.07)
        self.z = torch.randn(2, 250, 7)
        self.t = torch.tensor([.23, .71])
        self.context = torch.randn(2, 11, 5)

    def assert_close(self, a, b, **kwargs):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6, **kwargs)

    def test_legacy_grid_matches_native_volume_path(self):
        adapter = WanTokenAdapter(self.legacy, "legacy_grid")
        with torch.no_grad():
            self.assert_close(adapter(self.z, self.t, self.context), self.legacy(self.z, self.t, self.context))

    def test_channel_kernel_order_and_fused_affine_biases(self):
        backbone = self.legacy.backbone
        projected = self.legacy.input(self.z)
        native = backbone.patch_embedding(self.legacy.unpatchify(projected)).flatten(2).transpose(1, 2)
        direct = F.linear(projected, patch_projection_weight(backbone), backbone.patch_embedding.bias)
        self.assert_close(direct, native)
        wrong = F.linear(projected, backbone.patch_embedding.weight.flatten(1), backbone.patch_embedding.bias)
        self.assertGreater((wrong - native).abs().max().item(), .1)
        fused = fused_projection_tensors(self.legacy)
        self.assert_close(F.linear(self.z, fused["input_weight"], fused["input_bias"]), native)
        h = torch.randn(2, 250, 12)
        self.assert_close(F.linear(h, fused["output_weight"], fused["output_bias"]),
                          self.legacy.output(backbone.proj_out(h)))
        self.assertTrue(all(not v.requires_grad for v in fused.values()))

    def test_original_parameter_identity_count_and_gradients_preserved(self):
        adapter = WanTokenAdapter(self.legacy, "legacy_grid")
        original = dict(self.legacy.named_parameters())
        self.assertEqual(list(original), list(dict(adapter.named_parameters())))
        self.assertTrue(all(p is dict(adapter.named_parameters())[n] for n, p in original.items()))
        def gradients(module):
            module.zero_grad(set_to_none=True)
            z, context = self.z.clone().requires_grad_(), self.context.clone().requires_grad_()
            result = module(z, self.t, context)
            (result.square().sum() / result.numel()).backward()
            return result.detach(), z.grad, context.grad, {
                n: p.grad.clone() for n, p in module.named_parameters() if p.grad is not None}
        a, b = gradients(self.legacy), gradients(adapter)
        for x, y in zip(a[:3], b[:3]):
            self.assert_close(x, y)
        self.assertEqual(a[3].keys(), b[3].keys())
        for name in a[3]:
            self.assert_close(a[3][name], b[3][name])
        self.assertTrue(all(p.grad is None for n, p in adapter.named_parameters() if not p.requires_grad))

    def test_none_supports_variable_count_and_permutation_equivariance(self):
        adapter = WanTokenAdapter(self.legacy, "none")
        with torch.no_grad():
            for n in [1, 7, 63, 251]:
                z = torch.randn(2, n, 7)
                perm = torch.randperm(n)
                result = adapter(z, self.t, self.context)
                self.assertEqual(result.shape, z.shape)
                self.assert_close(adapter(z[:, perm], self.t, self.context), result[:, perm])
            with self.assertRaises(ValueError):
                WanTokenAdapter(self.legacy, "legacy_grid")(torch.randn(2, 7, 7), self.t, self.context)

    def test_removing_rope_changes_predictions(self):
        with torch.no_grad():
            grid = WanTokenAdapter(self.legacy, "legacy_grid")(self.z, self.t, self.context)
            no_grid = WanTokenAdapter(self.legacy, "none")(self.z, self.t, self.context)
        self.assertGreater((grid - no_grid).abs().max().item(), 1e-5)

    def test_object_xy_anchors_bind_slot_but_joint_permutation_is_equivariant(self):
        anchors = torch.tensor([[-.8, -.7], [-.4, .2], [.1, -.5], [.6, .8], [.9, -.1]])
        z = self.z[:, :len(anchors)]
        perm = torch.tensor([3, 0, 4, 1, 2])
        adapter = WanTokenAdapter.from_legacy(self.legacy, "object_xy", object_anchors=anchors, anchor_scale=.4)
        permuted = WanTokenAdapter.from_legacy(self.legacy, "object_xy", object_anchors=anchors[perm], anchor_scale=.4)
        with torch.no_grad():
            result = adapter(z, self.t, self.context)
            changed_slots = adapter(z[:, perm], self.t, self.context)
            self.assertGreater((changed_slots-result[:, perm]).abs().max().item(), 1e-6)
            self.assert_close(permuted(z[:, perm], self.t, self.context), result[:, perm])
        original = dict(self.legacy.named_parameters())
        self.assertEqual(list(original), list(dict(adapter.named_parameters())))
        self.assertTrue(all(p is dict(adapter.named_parameters())[n] for n, p in original.items()))
        self.assertIn("object_anchors", dict(adapter.named_buffers()))
        self.assertFalse(adapter.object_anchors.requires_grad)
        self.assertIsNone(adapter._position(z))

    def test_object_xy_checkpoint_rejects_wrong_anchor_identity_and_scale(self):
        anchors = torch.tensor([[-.8, -.7], [.1, -.5], [.6, .8]])
        adapter = WanTokenAdapter(self.legacy, "object_xy", object_anchors=anchors)
        state = copy.deepcopy(adapter.state_dict())
        WanTokenAdapter(copy.deepcopy(self.legacy), "object_xy", object_anchors=anchors).load_state_dict(state, strict=True)
        for changed, scale in [(anchors.flip(0), .1), (anchors, .2)]:
            with self.assertRaises(ValueError):
                WanTokenAdapter(copy.deepcopy(self.legacy), "object_xy", object_anchors=changed, anchor_scale=scale).load_state_dict(state)
        broken = copy.deepcopy(state)
        broken['object_anchors'][0, 0] += .1
        with self.assertRaises(ValueError):
            WanTokenAdapter(copy.deepcopy(self.legacy), "object_xy", object_anchors=anchors).load_state_dict(broken)
        explicit = WanTokenAdapter(copy.deepcopy(self.legacy), "object_xy", object_anchors=anchors)
        explicit.load_legacy_state_dict(self.legacy.state_dict())

    def test_object_xy_rejects_implicit_or_invalid_coordinates(self):
        for anchors in [None, torch.empty(0,2), torch.zeros(4,3), torch.tensor([[float('nan'),0.]])]:
            with self.assertRaises(ValueError):
                WanTokenAdapter(self.legacy, "object_xy", object_anchors=anchors)
        with self.assertRaises(ValueError):
            WanTokenAdapter(self.legacy, "none", object_anchors=torch.zeros(3,2))
        adapter = WanTokenAdapter(self.legacy, "object_xy", object_anchors=torch.zeros(3,2))
        with self.assertRaises(ValueError):
            adapter(self.z, self.t, self.context)

    def test_per_token_noise_levels_follow_native_modulation(self):
        t = torch.rand(2, 250)
        adapter = WanTokenAdapter(self.legacy, "legacy_grid")
        self.assert_close(adapter(self.z, t, self.context), self.legacy(self.z, t, self.context))

    def test_checkpoint_and_ordinary_gradients_match_without_position(self):
        adapter = WanTokenAdapter(self.legacy, "none")
        values = []
        for enabled in [False, True]:
            adapter.backbone.gradient_checkpointing = enabled
            adapter.zero_grad(set_to_none=True)
            z = self.z[:, :9].clone().requires_grad_()
            result = adapter(z, self.t, self.context)
            result.square().mean().backward()
            values.append((result.detach(), z.grad, adapter.input.weight.grad.clone()))
        for x, y in zip(*values):
            self.assert_close(x, y)

    def test_strict_legacy_import_and_position_metadata(self):
        adapter = WanTokenAdapter(copy.deepcopy(self.legacy), "none")
        adapter.load_legacy_state_dict(self.legacy.state_dict())
        state = adapter.state_dict()
        self.assertEqual(state["_extra_state"]["position_mode"], "none")
        WanTokenAdapter(copy.deepcopy(self.legacy), "none").load_state_dict(state, strict=True)
        with self.assertRaises(ValueError):
            WanTokenAdapter(copy.deepcopy(self.legacy), "legacy_grid").load_state_dict(state)

    def test_invalid_inputs_fail_explicitly(self):
        adapter = WanTokenAdapter(self.legacy, "none")
        for z, t, context in [(self.z[:, :0], self.t, self.context),
                              (self.z, self.t[:1], self.context),
                              (self.z, self.t, self.context[:1])]:
            with self.assertRaises(ValueError):
                adapter(z, t, context)
        with self.assertRaises(ValueError):
            WanTokenAdapter(self.legacy, "made_up")


if __name__ == "__main__":
    unittest.main()

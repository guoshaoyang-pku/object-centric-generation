"""Exact packing of cached regular VAE latents and frozen FP32 decoding.

This is a 96x192 regular spacetime VAE-patch target, not raw RGB diffusion.
The first twenty decoded frames reproduce evaluate_vae_floor.py alignment.
"""
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from cache_vae_features import coordinates, vae_fingerprint
from small_object_flow import SmallObjectFlow


TARGET_CONTRACT = {'cached_features': [384, 48], 'latent_grid_t_h_w': [6, 8, 8],
                   'patch_t_h_w': [1, 2, 2], 'tokens': 96, 'token_width': 192,
                   'packed_order': 't,patch_y,patch_x,dy,dx,channel',
                   'target_kind': 'cached normalized VAE posterior mode',
                   'decoded_alignment': 'FP32 unnormalize; VAE decode21; take first20; clampRGB01'}


def pack_features(features):
    if features.ndim != 3 or features.shape[1:] != (384, 48):
        raise ValueError('Regular patch baseline requires future features [B,384,48]')
    return features.reshape(-1, 6, 4, 2, 4, 2, 48).permute(0, 1, 2, 4, 3, 5, 6).reshape(-1, 96, 192)


def unpack_tokens(tokens):
    if tokens.ndim != 3 or tokens.shape[1:] != (96, 192):
        raise ValueError('Regular patch baseline requires tokens [B,96,192]')
    return tokens.reshape(-1, 6, 4, 4, 2, 2, 48).permute(0, 1, 2, 4, 3, 5, 6).reshape(-1, 384, 48)


def patch_anchors():
    axis = torch.linspace(-1, 1, 8).reshape(4, 2).mean(1)
    t, y, x = torch.meshgrid(torch.linspace(-1, 1, 6), axis, axis, indexing='ij')
    return torch.stack((x, y, t), -1).reshape(96, 3)


def check_future_grid(absolute, local):
    expected_absolute, expected_local = coordinates((1, 48, 6, 8, 8), 20, 4)
    for actual, expected in [(absolute, expected_absolute), (local, expected_local)]:
        if actual.ndim != 3 or actual.shape[1:] != (384, 3):
            raise ValueError('Cached future coordinate grid differs')
        value = torch.as_tensor(expected, device=actual.device, dtype=actual.dtype)
        if not torch.allclose(actual, value[None].expand_as(actual), rtol=0, atol=1e-6):
            raise ValueError('Cached future tokens are not in the admitted regular t/y/x order')


class NativePatchFlow(SmallObjectFlow):
    def __init__(self, encoder_width=256, width=384, depth=8, heads=6):
        super().__init__(96, 192, encoder_width, patch_anchors(), width, depth, heads)

    def get_extra_state(self):
        result = super().get_extra_state()
        result.update(backend='regular_vae_patch', target_contract=TARGET_CONTRACT)
        return result


class MockVAE(nn.Module):
    """CPU execution fixture only; never a quality-producing fallback."""
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(latents_mean=[0.] * 48, latents_std=[1.] * 48)

    def decode(self, z, return_dict=True):
        return SimpleNamespace(sample=F.interpolate(z[:, :3], (21, 128, 128), mode='trilinear',
                                                    align_corners=False).tanh() * self.scale)


class FrozenVAEDecoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae.float().requires_grad_(False).eval()
        mean = torch.tensor(vae.config.latents_mean).reshape(1, 48, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std).reshape(1, 48, 1, 1, 1)
        if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
            raise ValueError('Invalid VAE normalization configuration')
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def decode_tokens(self, tokens, size=128):
        features = unpack_tokens(tokens.float())
        with torch.autocast(device_type=features.device.type, enabled=False):
            z = features.transpose(1, 2).reshape(-1, 48, 6, 8, 8) * self.std + self.mean
            decoded = self.vae.decode(z, return_dict=True).sample
            if decoded.shape[1:] != (3, 21, 128, 128):
                raise ValueError('Wan VAE must return the admitted native 21x128x128 video')
            video = ((decoded[:, :, :20].permute(0, 2, 1, 3, 4).float() + 1) / 2).clamp(0, 1)
            if size != 128:
                video = F.interpolate(video.flatten(0, 1), (size, size), mode='bilinear',
                                      align_corners=False).reshape(tokens.shape[0], 20, 3, size, size)
        return video

    @torch.no_grad()
    def reference_decode_cached(self, features):
        """Independent literal original-floor route, without pack/unpack calls."""
        if features.shape != (1, 384, 48):
            raise ValueError('Reference parity checks one native cached clip at a time')
        with torch.autocast(device_type=features.device.type, enabled=False):
            z = features[0].T.reshape(1, 48, 6, 8, 8).float() * self.std + self.mean
            decoded = self.vae.decode(z, return_dict=True).sample
            return ((decoded[0, :, :20].permute(1, 0, 2, 3).float() + 1) / 2).clamp(0, 1)[None]


def load_vae(pretrained, bank, device, backend):
    if backend == 'cpu_mock':
        if device.type != 'cpu' or pretrained is not None:
            raise ValueError('Mock VAE requires explicit CPU mode and no pretrained path')
        vae = MockVAE()
        identity = {'backend': 'cpu_mock_fixture_only', 'quality_claims_admitted': False}
    elif backend == 'wan_vae':
        if pretrained is None:
            raise ValueError('Native patch evaluation requires --pretrained VAE root')
        fingerprint = vae_fingerprint(Path(pretrained))
        if fingerprint != bank.protocol.get('vae_fingerprint'):
            raise ValueError('VAE weights differ from the cache encoder fingerprint')
        import diffusers
        from diffusers import AutoencoderKLWan
        if diffusers.__version__ != '0.35.2':
            raise ValueError('VAE baseline is pinned to Diffusers0.35.2')
        vae = AutoencoderKLWan.from_pretrained(pretrained, subfolder='vae', torch_dtype=torch.float32,
                                               local_files_only=True)
        identity = {'backend': 'wan_vae', 'fingerprint': fingerprint, 'diffusers': diffusers.__version__,
                    'dtype': 'float32', 'native_shape': [21, 128, 128], 'kept_frames': [0, 20]}
    else:
        raise ValueError('Unknown VAE backend')
    decoder = FrozenVAEDecoder(vae).to(device)
    identity['parameters'] = sum(p.numel() for p in decoder.parameters())
    return decoder, identity

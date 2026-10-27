"""Randomly initialized native object-flow cost baseline.

ObjectFlow supplies full global object attention, prefix cross-attention and
learned slot identities. The immutable codec anchors bind each learned slot's
meaning; they are metadata, not an additional Fourier position input. No Wan
or VAE constructor is imported or called here.
"""
import hashlib

import torch

from model import ObjectFlow


def anchors_contract(anchors):
    value = anchors.detach().to(device='cpu', dtype=torch.float32).contiguous()
    return {'shape': list(value.shape), 'float32_sha256': hashlib.sha256(value.numpy().tobytes()).hexdigest()}


class SmallObjectFlow(ObjectFlow):
    def __init__(self, objects, latent_dim, encoder_width, anchors,
                 width=384, depth=8, heads=6):
        if min(objects, latent_dim, encoder_width, width, depth, heads) < 1 or width % heads:
            raise ValueError('Positive small-flow dimensions and width divisible by heads are required')
        if anchors.ndim != 2 or anchors.shape[0] != objects or anchors.shape[1] not in {2, 3}:
            raise ValueError('Anchors must bind every codec slot with xy or xyt coordinates')
        if not bool(torch.isfinite(anchors).all()):
            raise ValueError('Nonfinite codec anchors')
        super().__init__(objects, latent_dim, encoder_width, width, depth, heads)
        self.model_config = {'objects': int(objects), 'latent_dim': int(latent_dim),
                             'encoder_width': int(encoder_width), 'width': int(width),
                             'depth': int(depth), 'heads': int(heads)}
        self.position_mode = 'learned_object_slots'
        self.register_buffer('slot_anchors', anchors.detach().to(dtype=torch.float32).clone())
        self._expected_anchors = anchors_contract(self.slot_anchors)

    def get_extra_state(self):
        return {'version': 1, 'backend': 'small', 'architecture': 'model.ObjectFlow',
                'model_config': self.model_config, 'position_mode': self.position_mode,
                'slot_anchor_contract': self._expected_anchors,
                'attention': 'full global self attention plus prefix cross attention',
                'initialization': 'random weights; zero output projection; no pretrained backbone'}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('Small-flow architecture or positional contract differs')
        if anchors_contract(self.slot_anchors) != self._expected_anchors:
            raise ValueError('Small-flow codec slot anchors differ')

    def forward(self, z, t, context):
        cfg = self.model_config
        if z.ndim != 3 or z.shape[1:] != (cfg['objects'], cfg['latent_dim']):
            raise ValueError('Object latent shape differs from the fixed small-flow configuration')
        if context.ndim != 3 or context.shape[0] != z.shape[0] or context.shape[-1] != cfg['encoder_width']:
            raise ValueError('Prefix context shape differs from small-flow configuration')
        if t.shape != (z.shape[0],):
            raise ValueError('Flow time must be one scalar per clip')
        # Match WanTokenAdapter's BF16 compute / FP32 master-parameter policy.
        with torch.autocast(device_type=z.device.type, dtype=torch.bfloat16,
                            enabled=z.device.type == 'cuda'):
            return super().forward(z, t, context)

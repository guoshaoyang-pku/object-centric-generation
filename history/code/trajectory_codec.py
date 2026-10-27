"""Persistent parents with shared 2-D motion and object-local Gaussian children.

This is an experimental decoder, independent of the live ``CompositeDecoder``.
The third query coordinate is video time in [-1, 1], not physical depth. A
parent exists throughout that interval; it has no time-position anchor or
parent lifetime gate. Geometry and texture share its moving local coordinates.
Rendering retains normalized density mixing, without introducing depth layers.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def bezier_basis(time):
    """Cubic Bernstein basis for normalized video time [-1, 1]."""
    u = (time.float() + 1) * .5
    v = 1 - u
    return torch.stack((v.pow(3), 3 * v.square() * u,
                        3 * v * u.square(), u.pow(3)), dim=-1)


def xy_anchors(objects):
    """A deterministic spatial initialization, with no time partition."""
    if objects < 1:
        raise ValueError('objects must be positive')
    nx = math.ceil(math.sqrt(objects))
    ny = math.ceil(objects / nx)
    x = torch.linspace(-.9, .9, nx) if nx > 1 else torch.zeros(1)
    y = torch.linspace(-.9, .9, ny) if ny > 1 else torch.zeros(1)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    return torch.stack((xx, yy), -1).reshape(-1, 2)[:objects], (nx, ny)


def disk_points(count, radius):
    if count == 0:
        return torch.empty(0, 2)
    if count == 1:
        return torch.zeros(1, 2)
    i = torch.arange(count, dtype=torch.float32)
    angle = i * (math.pi * (3 - math.sqrt(5)))
    r = radius * ((i + .5) / count).sqrt()
    return torch.stack((r * angle.cos(), r * angle.sin()), -1)


class TrajectoryDecoder(nn.Module):
    """Decode each parent latent into a shared trajectory and local children.

    ``decode(z)`` accepts [B, objects, latent_dim]. ``render(p, coords)`` accepts
    [B, Q, 3] x/y/time queries and returns [B, Q, 3] RGB in [0, 1].

    Each object has a cubic Bezier center c(t), and an affine transform
    A(t) = R(theta(t)) @ diag(exp(log_scale(t))) @ shear(h(t)). The four
    affine coordinates vary linearly in normalized time. Every child and every
    texture kernel shares this transform. Child geometry is a static local
    ellipse; optional local time envelopes add two fields per child.

    Object/geometry fields: 2 center + 6 curve offsets + 8 affine + 6M child
    fields + 3 base RGB logits (+ 2M optional child-time fields).
    Texture fields: 7K = 2 center + 2 scales + 3 RGB residuals per 2-D RBF.
    Counts describe decoded values, not the number of network weights.
    """
    def __init__(self, objects=250, latent_dim=128, subkernels=4,
                 texture_kernels=16, temporal_support=False,
                 motion_init='static', hidden_width=256):
        super().__init__()
        if subkernels < 1 or texture_kernels < 0:
            raise ValueError('Require subkernels >= 1 and texture_kernels >= 0')
        if motion_init not in {'static', 'small'}:
            raise ValueError("motion_init must be 'static' or 'small'")
        self.objects = int(objects)
        self.latent_dim = int(latent_dim)
        self.subkernels = int(subkernels)
        self.texture_kernels = int(texture_kernels)
        self.temporal_support = bool(temporal_support)
        anchors, (nx, ny) = xy_anchors(self.objects)
        self.register_buffer('anchors', anchors)
        self.register_buffer('base_log_scale', torch.tensor([
            1.4 / max(nx - 1, 1), 1.4 / max(ny - 1, 1)]).log())
        self.register_buffer('child_centers', disk_points(subkernels, .65))
        self.register_buffer('texture_centers', disk_points(texture_kernels, 1.2))
        self.geometry_dim = 19 + 6 * subkernels + (2 * subkernels if temporal_support else 0)
        self.texture_dim = 7 * texture_kernels
        self.output_dim = self.geometry_dim + self.texture_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, self.output_dim))
        nn.init.normal_(self.net[-1].weight, std=.002)
        nn.init.zeros_(self.net[-1].bias)
        if motion_init == 'static':
            # c1=c2=c3=c0 and affine slopes zero, while motion rows remain
            # trainable. This does not freeze object positions or appearances.
            with torch.no_grad():
                for span in (slice(2, 8), slice(12, 16)):
                    self.net[-1].weight[span].zero_()
                    self.net[-1].bias[span].zero_()

    def parameter_counts(self):
        return {
            'network_parameters': sum(p.numel() for p in self.parameters()),
            'geometry_fields_per_object': self.geometry_dim,
            'texture_fields_per_object': self.texture_dim,
            'fields_per_object': self.output_dim,
            'geometry_fields_per_clip': self.objects * self.geometry_dim,
            'texture_fields_per_clip': self.objects * self.texture_dim,
            'latent_values_per_clip': self.objects * self.latent_dim,
        }

    def decode(self, z):
        if z.ndim != 3 or z.shape[1:] != (self.objects, self.latent_dim):
            raise ValueError('Expected latent shape [B, objects, latent_dim]')
        raw = self.net(z).float()
        b, n, _ = raw.shape
        m, k = self.subkernels, self.texture_kernels
        center = self.anchors[None] + raw[..., :2].tanh()
        offsets = 1.25 * raw[..., 2:8].reshape(b, n, 3, 2).tanh()
        controls = torch.cat((center[..., None, :], center[..., None, :] + offsets), -2)
        affine_base = torch.cat((
            math.pi * raw[..., 8:9].tanh(),
            self.base_log_scale[None, None] + raw[..., 9:11].clamp(-2.5, 2.5),
            .8 * raw[..., 11:12].tanh()), -1)
        affine_slope = torch.cat((
            math.pi * raw[..., 12:13].tanh(),
            .75 * raw[..., 13:15].tanh(),
            .4 * raw[..., 15:16].tanh()), -1)
        j = 16
        children = raw[..., j:j + 6 * m].reshape(b, n, m, 6)
        j += 6 * m
        child_center = self.child_centers[None, None] + .8 * children[..., :2].tanh()
        child_scale = (math.log(.65) + .8 * children[..., 2:4].tanh()).exp()
        child_angle = math.pi * children[..., 4].tanh()
        # Log-softmax is a relative mixture amplitude: changing all amplitudes
        # equally must not create or destroy a parent's apparent existence.
        child_log_amplitude = F.log_softmax(children[..., 5], dim=-1)
        p = {'controls': controls, 'affine_base': affine_base,
             'affine_slope': affine_slope, 'child_center': child_center,
             'child_scale': child_scale, 'child_angle': child_angle,
             'child_log_amplitude': child_log_amplitude,
             'color': raw[..., j:j + 3]}
        j += 3
        if self.temporal_support:
            temporal = raw[..., j:j + 2 * m].reshape(b, n, m, 2)
            j += 2 * m
            p.update(child_time_center=temporal[..., 0].tanh(),
                     child_time_scale=(math.log(1.5) + temporal[..., 1].clamp(-2, 1)).exp())
        if j != self.geometry_dim:
            raise RuntimeError('Geometry field accounting mismatch')
        if k:
            tex = raw[..., j:].reshape(b, n, k, 7)
            p.update(texture_center=self.texture_centers[None, None] + .75 * tex[..., :2].tanh(),
                     texture_scale=(math.log(.45) + .8 * tex[..., 2:4].tanh()).exp(),
                     texture_rgb=.75 * tex[..., 4:7].tanh())
        return p

    @staticmethod
    def motion_at(p, time):
        """Return centers [B,Q,N,2] and affine coordinates [B,Q,N,4]."""
        if time.ndim != 2 or time.shape[0] != p['controls'].shape[0]:
            raise ValueError('Expected time shape [B, Q]')
        basis = bezier_basis(time)
        # Small fixed degree, explicit accumulation avoids autocast of einsum.
        centers = sum(basis[..., i, None, None] * p['controls'][:, None, :, i, :]
                      for i in range(4))
        affine = p['affine_base'][:, None] + time.float()[..., None, None] * p['affine_slope'][:, None]
        return centers, affine

    @staticmethod
    def affine_matrix(affine):
        """A=R @ diag(sx,sy) @ [[1,h],[0,1]], with positive determinant."""
        angle, log_sx, log_sy, shear = affine.unbind(-1)
        c, s = angle.cos(), angle.sin()
        sx, sy = log_sx.exp(), log_sy.exp()
        return torch.stack((c * sx, c * sx * shear - s * sy,
                            s * sx, s * sx * shear + c * sy), -1).reshape(*affine.shape[:-1], 2, 2)

    @staticmethod
    def local_coordinates(p, coords):
        centers, affine = TrajectoryDecoder.motion_at(p, coords[..., 2])
        delta = coords[..., :2, None].transpose(-1, -2) - centers
        angle, log_sx, log_sy, shear = affine.unbind(-1)
        c, s = angle.cos(), angle.sin()
        dx, dy = delta.unbind(-1)
        rx, ry = c * dx + s * dy, -s * dx + c * dy
        ly = ry * (-log_sy).exp()
        lx = rx * (-log_sx).exp() - shear * ly
        return torch.stack((lx, ly), -1)

    def render(self, p, coords):
        if coords.ndim != 3 or coords.shape[-1] != 3 or coords.shape[0] != p['controls'].shape[0]:
            raise ValueError('Expected coordinates [B, Q, 3] matching latent batch')
        # Geometry is FP32 under an outer BF16 training autocast, including the
        # log-density normalization. No inverse of a poorly conditioned matrix.
        with torch.autocast(device_type=coords.device.type, enabled=False):
            coords = coords.float()
            local = self.local_coordinates(p, coords)
            delta = local[..., None, :] - p['child_center'][:, None]
            angle = p['child_angle'][:, None]
            c, s = angle.cos(), angle.sin()
            dx, dy = delta.unbind(-1)
            child = torch.stack((c * dx + s * dy, -s * dx + c * dy), -1)
            distance = (child / p['child_scale'][:, None]).square().sum(-1)
            log_density = -.5 * distance + p['child_log_amplitude'][:, None]
            if self.temporal_support:
                dt = coords[..., 2, None, None] - p['child_time_center'][:, None]
                log_density = log_density - .5 * (dt / p['child_time_scale'][:, None]).square()
            parent_log_density = torch.logsumexp(log_density, dim=-1)
            weights = parent_log_density.softmax(dim=-1)
            color = p['color'][:, None]
            if self.texture_kernels:
                d = (local[..., None, :] - p['texture_center'][:, None]) / p['texture_scale'][:, None]
                rbf = (-.5 * d.square().sum(-1)).exp()
                # A bounded sum, rather than a softmax, lets texture vanish
                # outside its actual support in object-local coordinates.
                texture = (rbf[..., None] * p['texture_rgb'][:, None]).sum(-2)
                color = color + texture
            return (weights[..., None] * color.sigmoid()).sum(-2)

    def forward(self, z, coords):
        return self.render(self.decode(z), coords)
